from absl import app, flags
from functools import partial
import numpy as np
import jax
import tqdm
import gym
import wandb
from ml_collections import config_flags
from flax.training import checkpoints
import ml_collections
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from typing import Any

from utils.wandb import setup_wandb, default_wandb_config, get_flag_dict
from utils.evaluation import evaluate, flatten
from utils.train_state import TrainState, target_update, supply_rng
from utils.networks import Policy, ValueCritic, Critic, ensemblize
from utils.rms import RunningMeanStd
from utils.dataset import ReplayBuffer
from envs.env_helper import make_env, make_vec_env

###############################
#  Configs
###############################


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('video_interval', 250000, 'Video interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')


flags.DEFINE_integer('minibatch_size', 64, 'Mini batch size.') # Multiplied by num_envs. Batch will be [batch_size, num_envs, ...]
flags.DEFINE_integer('num_envs', 16, 'Number of environments to run in parallel.')
flags.DEFINE_integer('num_steps_per_update', 128, 'Number of steps per epoch.') # Total-Batch = num_envs * num_steps_per_update
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs per update.')

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'rlbase_default',
    'name': 'ppo_{env_name}',
})

agent_config = ml_collections.ConfigDict({
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'hidden_dims': (512, 512, 512),
    'final_fc_init_scale': 0.01,
    'gamma': 0.99,
    'lam': 0.95,
    'ent_coef': 0,
    'clip_grad_norm': 1.5,
    'clip_ratio': 0.2,
    'use_clipping': 1,
    'recompute_advantages': 1,
    'use_tanh': 0,
    'state_dependent_std': 0,
    'normalize_reward': 1,
    'normalize_advantage': 1,
    'normalize_input': 1,
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('agent', agent_config, lock_config=False)

###############################
#  Agent. Contains the neural networks, training logic, and sampling.
###############################

class PPOAgent(flax.struct.PyTreeNode):
    rng: Any
    critic: TrainState
    actor: TrainState
    rms_obs: RunningMeanStd
    config: dict = flax.struct.field(pytree_node=False)

    @functools.partial(jax.jit)
    def compute_gae(agent, batch):
        batch_value = agent.critic(batch['observations']) # [batch_size, num_envs]
        batch_next_value = agent.critic(batch['next_observations'])

        # GAE Advantages
        def scan_fn(lastgaelam, inputs):
            reward, mask, value, next_value = inputs
            delta = reward + mask * agent.config['gamma'] * next_value - value
            advantage = delta + mask * agent.config['gamma'] * agent.config['lam'] * lastgaelam
            return advantage, advantage
        zeros = jnp.zeros((FLAGS.num_envs,))
        _, batch_advantages = jax.lax.scan(scan_fn, zeros, 
                                           (batch['rewards'], batch['masks'], batch_value, batch_next_value),
                                           reverse=True)
        batch_normalized_adv = (batch_advantages - jnp.mean(batch_advantages)) / (jnp.std(batch_advantages) + 1e-7)
        batch_values_gae = batch_value + batch_advantages
        assert batch_advantages.shape == batch_value.shape, (batch_advantages.shape, batch_value.shape)
        return batch_values_gae, batch_normalized_adv


    @functools.partial(jax.jit)
    def update(agent, batch):
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)
        assert len(batch['observations'].shape) == 3 # [batch_size, num_envs, obs_dim]

        # reshape into [batch_size * num_envs, ...]
        batch = {k: v.reshape((-1, *v.shape[2:])) for k, v in batch.items()}

        batch_values_gae = batch['values_gae']
        v_gae_est = batch_values_gae.mean()
        batch_normalized_adv = batch['normalized_advantages']

        def critic_loss_fn(critic_params):
            values = agent.critic(batch['observations'], params=critic_params)

            if values.shape != batch_values_gae.shape:
                raise ValueError(f'Values shape {values.shape} does not match batch_values shape {batch_values_gae.shape}')
            critic_loss = ((values - batch_values_gae) ** 2).mean()

            return critic_loss, {
                'critic_loss': critic_loss,
                'v_pred': values.mean(),
                'v_gae_est': v_gae_est,
            }        

        def actor_loss_fn(actor_params):
            if agent.config['use_tanh']:
                safe_actions = jnp.clip(batch['actions'], -1.0 + 1e-5, 1.0 - 1e-5)
            else:
                safe_actions = batch['actions']

            action_dist = agent.actor(batch['observations'], params=actor_params)
            log_probs = action_dist.log_prob(safe_actions)

            logratio = log_probs - batch['old_log_probs']
            ratio = jnp.exp(logratio) # p_new / p_old

            assert batch_normalized_adv.shape == ratio.shape, (batch_normalized_adv.shape, ratio.shape)

            if agent.config['use_clipping']:
                pg_loss1 = -batch_normalized_adv * ratio
                pg_loss2 = -batch_normalized_adv * jnp.clip(ratio, 1 - agent.config['clip_ratio'], 
                                                            1 + agent.config['clip_ratio'])
                pg_loss = jnp.maximum(pg_loss1, pg_loss2)
            else:
                pg_loss = -batch_normalized_adv * log_probs
            approx_kl = ((ratio - 1) - logratio).mean()
            clip_fracs = (jnp.abs(ratio - 1.0) > agent.config['clip_ratio']).mean()

            # Entropy must be computed on logprobs that pass gradient through the action.
            actions_live, log_probs_live = action_dist.sample_and_log_prob(seed=curr_key)
            entropy = -1 * log_probs_live

            # Only for debugging
            if agent.config['use_tanh']:
                action_std = action_dist._distribution.stddev()
            else:
                action_std = action_dist.stddev().mean()

            loss = pg_loss.mean() - agent.config['ent_coef'] * entropy.mean()
            
            return loss, {
                'actor_loss': loss.mean(),
                'pg_loss': pg_loss.mean(),
                'entropy': entropy.mean(),
                'ratio': ratio.mean(),
                'action_std': action_std,
                'approx_kl': approx_kl,
                'clip_fracs': clip_fracs,
            }
        
        info = {
            'normalized_advantages': jnp.abs(batch_normalized_adv).mean(),
        }
        
        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        return agent.replace(rng=new_rng, critic=new_critic, actor=new_actor), {
            **critic_info, **actor_info, **info}

    @jax.jit
    def sample_actions(agent, observations: np.ndarray, *, seed: Any, temperature: float = 1.0) -> jnp.ndarray:
        if type(observations) is dict:
            observations = jnp.concatenate([observations['observation'], observations['goal']], axis=-1)
        observations = observations[None]
        if agent.config['normalize_input']:
            observations = agent.rms_obs.norm(observations)
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)[0]
        return actions
    
    @jax.jit
    def sample_actions_and_logprobs(agent, observations: np.ndarray, *, seed: Any, temperature: float = 1.0):
        if type(observations) is dict:
            observations = jnp.concatenate([observations['observation'], observations['goal']], axis=-1)
        observations = observations[None]
        if agent.config['normalize_input']:
            observations = agent.rms_obs.norm(observations)
        actions_dist = agent.actor(observations, temperature=temperature)[0]
        actions = actions_dist.sample(seed=seed)
        log_probs = actions_dist.log_prob(actions)
        return actions, log_probs


def create_agent(seed: int,
                observations: jnp.ndarray,
                actions: jnp.ndarray,
            **kwargs):

        print('Extra kwargs:', kwargs)
        config = kwargs

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        action_dim = actions.shape[-1]
        actor_def = Policy(config['hidden_dims'], action_dim=action_dim, 
            log_std_min=-10.0, state_dependent_std=config['state_dependent_std'], tanh_squash_distribution=config['use_tanh'], final_fc_init_scale=config['final_fc_init_scale'])

        actor_params = actor_def.init(actor_key, observations)['params']
        actor_tx = optax.chain(optax.clip_by_global_norm(config['clip_grad_norm']), optax.adam(learning_rate=config['actor_lr']))
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

        critic_def = ValueCritic(config['hidden_dims'])
        critic_params = critic_def.init(critic_key, observations)['params']
        critic_tx = optax.chain(optax.clip_by_global_norm(config['clip_grad_norm']), optax.adam(learning_rate=config['critic_lr']))
        critic = TrainState.create(critic_def, critic_params, tx=critic_tx)

        config_dict = flax.core.FrozenDict(**config)
        rms_obs = RunningMeanStd()
        
        return PPOAgent(rng, critic=critic, actor=actor, rms_obs=rms_obs, config=config_dict)

###############################
#  Run Script. Loads data, logs to wandb, and runs the training loop.
###############################

def main(_):
    np.random.seed(FLAGS.seed)

    # Create wandb logger
    setup_wandb(FLAGS.agent.to_dict(), **FLAGS.wandb)

    # PPO trains on vectorized environments.
    env = make_vec_env(FLAGS.env_name, num_envs=FLAGS.num_envs)
    if FLAGS.agent.normalize_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=FLAGS.agent.gamma)
    eval_env = make_env(FLAGS.env_name)
    
    ones = np.ones((FLAGS.num_envs,))
    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        old_log_probs=ones,
        rewards=ones,
        masks=ones,
        next_observations=env.observation_space.sample(),
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(1e6))

    agent = create_agent(FLAGS.seed,
                    example_transition['observations'][None],
                    example_transition['actions'][None],
                    max_steps=FLAGS.max_steps,
                    **FLAGS.agent)
    
    exploration_metrics = dict()
    obs = env.reset()    
    exploration_rng = jax.random.PRNGKey(0)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        exploration_rng, key = jax.random.split(exploration_rng)
        action, log_probs = agent.sample_actions_and_logprobs(obs, seed=key) # Range [-1, 1].

        next_obs, reward, done, info = env.step(action)
        truncated = np.array(['TimeLimit.truncated' in i for i in info])
        mask = ((~done) | truncated).astype(np.float32)

        replay_buffer.add_transition(dict(
            observations=obs,
            actions=action,
            rewards=reward,
            masks=mask,
            next_observations=next_obs,
            old_log_probs=log_probs,
        ))
        obs = next_obs

        # Log metrics from the first environment.
        if done[0]:
            exploration_metrics = {f'exploration/{k}': v for k, v in flatten(info[0]).items()}

        # Train model
        if i % FLAGS.num_steps_per_update == 0:
            batch = replay_buffer.get_subset(np.arange(replay_buffer.size))

            if FLAGS.agent.normalize_input:
                agent = agent.replace(
                    rms_obs=agent.rms_obs.update(batch['observations']),
                )
                batch['observations'] = agent.rms_obs.norm(batch['observations'])
                batch['next_observations'] = agent.rms_obs.norm(batch['next_observations'])

            values_gae, advantages = agent.compute_gae(batch)
            batch['values_gae'] = values_gae
            batch['normalized_advantages'] = advantages

            for n in range(FLAGS.num_epochs):
                if FLAGS.agent.recompute_advantages and n != 0:
                    values_gae, advantages = agent.compute_gae(batch)
                    batch['values_gae'] = values_gae
                    batch['normalized_advantages'] = advantages
                all_indicies = np.arange(replay_buffer.size)
                np.random.shuffle(all_indicies)
                for start in range(0, len(all_indicies), FLAGS.minibatch_size):
                    indicies = all_indicies[start:start+FLAGS.minibatch_size]
                    minibatch = {k: v[indicies] for k, v in batch.items()}
                    agent, update_info = agent.update(minibatch)

            replay_buffer.clear()

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)
            wandb.log(exploration_metrics, step=i)
            exploration_metrics = dict()

        if (i % FLAGS.eval_interval == 0) or i == 0:
            record_video = (i % FLAGS.video_interval == 0) or i == 0
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=1.0)
            eval_info = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes, record_video=record_video)
            eval_metrics = {}
            for k in ['success', 'episode.return', 'episode.length']:
                eval_metrics[f'evaluation/{k}'] = eval_info[k]
            try:
                if record_video:
                    wandb.log({'video': eval_info['video']}, step=i)
            except:
                print('Failed to log video')
            wandb.log(eval_metrics, step=i)

if __name__ == '__main__':
    app.run(main)