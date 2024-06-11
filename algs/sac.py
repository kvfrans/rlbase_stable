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
import flax.linen as nn
from typing import Any

from utils.wandb import setup_wandb, default_wandb_config, get_flag_dict
from utils.evaluation import evaluate, flatten
from utils.train_state import TrainState, target_update, supply_rng
from utils.networks import Policy, ValueCritic, Critic, ensemblize
from utils.dataset import ReplayBuffer
from envs.env_helper import make_env

###############################
#  Configs
###############################


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('video_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_steps', int(1e4), 'Number of initial exploration steps.')

agent_config = ml_collections.ConfigDict({
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'temp_lr': 3e-4,
    'hidden_dims': '(512, 512, 512)',
    'use_layer_norm': 1,
    'activation': 'mish',
    'discount': 0.99,
    'tau': 0.005,
    'target_entropy': ml_collections.config_dict.placeholder(float),
    'backup_entropy': 1,
    'use_tanh': 1,
    'state_dependent_std': 1,
    'target_entropy_multiplier': 0.5,
    'final_fc_init_scale': 1.0,
    'utd': 1,
    'actor_utd': 1,
    'num_q': 2,
    'num_q_subsample': 2, # Must be <= num_q
})

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'rlbase_default',
    'name': 'sac_{env_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('agent', agent_config, lock_config=False)

###############################
#  Agent. Contains the neural networks, training logic, and sampling.
###############################

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)

# Operates over a flattened [obs, goal].
class SACAgent(flax.struct.PyTreeNode):
    rng: Any
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    temp: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @partial(jax.jit, static_argnames=('update_actor_and_temp',))
    def update(agent, batch, update_critic: bool = True, update_actor_and_temp: bool = True):
        new_rng, curr_key, next_key, subsample_key = jax.random.split(agent.rng, 4)

        def critic_loss_fn(critic_params):
            next_dist = agent.actor(batch['next_observations'])
            next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=next_key)

            next_qs = agent.target_critic(batch['next_observations'], next_actions) # (num_q, batch_size)
            if agent.config['num_q'] == agent.config['num_q_subsample']:
                next_qs_subset = next_qs
            else:
                subsample_idcs = jax.random.randint(subsample_key, (agent.config["num_q_subsample"],), 0, agent.config["num_q"],)
                next_qs_subset = next_qs[subsample_idcs]
            next_q = jnp.min(next_qs_subset, axis=0) # (batch_size)
            
            target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
            if agent.config['backup_entropy']:
                entropy = -1 * next_log_probs
                target_q = target_q + agent.config['discount'] * batch['masks'] * entropy * agent.temp()
            
            q = agent.critic(batch['observations'], batch['actions'], params=critic_params) # (num_q, batch_size)
            target_q_expand = jnp.expand_dims(target_q, axis=0) # (1, batch_size)
            critic_loss = jnp.square(q - target_q_expand).mean()
            
            return critic_loss, {
                'critic_loss': critic_loss,
                'q': q.mean(),
            }        

        def actor_loss_fn(actor_params):
            dist = agent.actor(batch['observations'], params=actor_params)
            actions, log_probs = dist.sample_and_log_prob(seed=curr_key)
            
            qs = agent.critic(batch['observations'], actions)
            q = jnp.min(qs, axis=0)

            # Only for debugging
            if agent.config['use_tanh']:
                action_std = dist._distribution.stddev()
            else:
                action_std = dist.stddev().mean()

            actor_loss = (log_probs * agent.temp() - q).mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': -1 * log_probs.mean(),
                'action_std': action_std 
            }
        
        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp(params=temp_params)
            temp_loss = (temperature * (entropy - target_entropy)).mean()
            return temp_loss, {
                'temp_loss': temp_loss,
                'temperature': temperature,
            }
        
        if update_actor_and_temp:
            new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)
            temp_loss_fn = functools.partial(temp_loss_fn, entropy=actor_info['entropy'], target_entropy=agent.config['target_entropy'])
            new_temp, temp_info = agent.temp.apply_loss_fn(loss_fn=temp_loss_fn, has_aux=True)
        else:
            new_actor, actor_info = agent.actor, {}
            new_temp, temp_info = agent.temp, {}

        if update_critic:
            new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
            new_target_critic = target_update(agent.critic, agent.target_critic, agent.config['tau'])
        else:
            new_critic, critic_info = agent.critic, {}
            new_target_critic = agent.target_critic

        return agent.replace(rng=new_rng, critic=new_critic, target_critic=new_target_critic, actor=new_actor, temp=new_temp), {
            **critic_info, **actor_info, **temp_info}
    
    @jax.jit
    def update_high_utd(agent, batch):
        batch_size = batch["rewards"].shape[0]
        assert batch_size % agent.config['utd'] == 0, f"Batch {batch_size} must divide by UTD {agent.config['utd']}"
        assert batch_size % agent.config['actor_utd'] == 0, f"Batch {batch_size} must divide by Actor UTD {agent.config['actor_utd']}"

        # Take `utd_ratio` gradient descent steps on the critic
        minibatch_size = batch_size // agent.config['utd']
        def scan_body(carry, data):
            (agent,) = carry
            (minibatch,) = data
            agent, info = agent.update(minibatch, update_actor_and_temp=False)
            return (agent,), info
        def make_minibatch(data: jnp.ndarray):
            return jnp.reshape(data, (agent.config['utd'], minibatch_size) + data.shape[1:])
        minibatches = jax.tree_map(make_minibatch, batch)
        (agent,), critic_infos = jax.lax.scan(scan_body, (agent,), (minibatches,))
        critic_infos = jax.tree_map(lambda x: jnp.mean(x, axis=0), critic_infos)

        # Take 'actor_utd' gradient descent step on the actor and temperature
        minibatch_size = batch_size // agent.config['actor_utd']
        def scan_body(carry, data):
            (agent,) = carry
            (minibatch,) = data
            agent, info = agent.update(minibatch, update_actor_and_temp=True)
            return (agent,), info
        def make_minibatch(data: jnp.ndarray):
            return jnp.reshape(data, (agent.config['actor_utd'], minibatch_size) + data.shape[1:])
        minibatches = jax.tree_map(make_minibatch, batch)
        (agent,), actor_temp_infos = jax.lax.scan(scan_body, (agent,), (minibatches,))
        actor_temp_infos = jax.tree_map(lambda x: jnp.mean(x, axis=0), actor_temp_infos)

        infos = {**critic_infos, **actor_temp_infos}

        return agent, infos

    @jax.jit
    def sample_actions(agent, observations: np.ndarray, *, seed: Any, temperature: float = 1.0) -> jnp.ndarray:
        if type(observations) is dict:
            observations = jnp.concatenate([observations['observation'], observations['goal']], axis=-1)
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

def create_agent(
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
            **kwargs):

        print('Extra kwargs:', kwargs)
        config = kwargs

        if type(config['hidden_dims']) is str:
            config['hidden_dims'] = eval(config['hidden_dims'])

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        mlp_kwargs = dict(activation=config['activation'], use_layer_norm=config['use_layer_norm'])

        action_dim = actions.shape[-1]
        actor_def = Policy(config['hidden_dims'], action_dim=action_dim, 
            log_std_min=-10.0, state_dependent_std=config['state_dependent_std'], tanh_squash_distribution=config['use_tanh'], final_fc_init_scale=config['final_fc_init_scale'], mlp_kwargs=mlp_kwargs)

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=config['actor_lr']))

        critic_def = ensemblize(Critic, num_qs=config['num_q'])(config['hidden_dims'])
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=config['critic_lr']))
        target_critic = TrainState.create(critic_def, critic_params)

        temp_def = Temperature()
        temp_params = temp_def.init(rng)['params']
        temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=config['temp_lr']))

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        config_dict = flax.core.FrozenDict(**config)
        return SACAgent(rng, critic=critic, target_critic=target_critic, actor=actor, temp=temp, config=config_dict)

###############################
#  Run Script. Loads data, logs to wandb, and runs the training loop.
###############################

def main(_):

    # Create wandb logger
    setup_wandb(FLAGS.agent.to_dict(), **FLAGS.wandb)

    env = make_env(FLAGS.env_name)
    eval_env = make_env(FLAGS.env_name)
    
    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
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

    for i in tqdm.tqdm(range(0, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        if i < FLAGS.start_steps:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action = agent.sample_actions(obs, seed=key)

        next_obs, reward, done, info = env.step(action)
        mask = float(not done or 'TimeLimit.truncated' in info)

        replay_buffer.add_transition(dict(
            observations=obs,
            actions=action,
            rewards=reward,
            masks=mask,
            next_observations=next_obs,
        ))
        obs = next_obs

        if done:
            exploration_metrics = {f'exploration/{k}': v for k, v in flatten(info).items()}
            obs = env.reset()

        if replay_buffer.size < FLAGS.start_steps:
            continue

        batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.agent.utd)
        agent, update_info = agent.update_high_utd(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)
            wandb.log(exploration_metrics, step=i)
            exploration_metrics = dict()

        if (i % FLAGS.eval_interval == 0):
            record_video = (i % FLAGS.video_interval == 0)
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info, trajectories = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes, record_video=record_video, return_trajectories=True)
            eval_metrics = {}
            for k in ['success', 'episode.return', 'episode.length']:
                eval_metrics[f'evaluation/{k}'] = eval_info[k]
            if record_video:
                wandb.log({'video': eval_info['video']}, step=i)

            # Measure Q-Function Bias.
            q_bias = []
            q_bias_std = []
            for t in trajectories:
                observations = np.array(t['observation'])
                actions = np.array(t['action'])
                rewards = np.array(t['reward'])
                masks = np.array(t['mask'])
                final_obs_q = agent.critic(observations[-1:], actions[-1:])[0]

                if agent.config['backup_entropy']:
                    action_logprobs = agent.actor(observations).log_prob(actions)
                    rewards += agent.temp() * -1 * action_logprobs

                discounted_mc_returns = np.zeros_like(rewards)
                discounted_mc_returns[-1] = final_obs_q
                for r in reversed(range(len(rewards) - 1)):
                    discounted_mc_returns[r] = rewards[r] + FLAGS.agent.discount * masks[r] * discounted_mc_returns[r + 1]

                q_values = agent.critic(observations[:-1], actions[:-1])
                q_values = np.min(q_values, axis=0)
                q_bias.append((q_values - discounted_mc_returns[:-1]).mean())
                q_bias_std.append(np.std(q_values - discounted_mc_returns[:-1]).mean())
            eval_metrics['evaluation/q_bias'] = np.mean(q_bias)
            eval_metrics['evaluation/q_bias_std'] = np.mean(q_bias_std)

            wandb.log(eval_metrics, step=i)

if __name__ == '__main__':
    app.run(main)