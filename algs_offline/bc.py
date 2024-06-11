from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import flax
import wandb
from ml_collections import config_flags
import ml_collections
from typing import Any

from utils.gc_dataset import GCDataset
from utils.wandb import setup_wandb, default_wandb_config, get_flag_dict
from utils.evaluation import evaluate
from utils.train_state import TrainState, target_update, supply_rng
from utils.networks import Policy
from envs.env_helper import make_env, get_dataset


###############################
#  Configs
###############################

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'gc-antmaze-large-diverse-v2', 'Environment name.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('video_interval', 50000, 'Video interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('use_validation', 0, 'Whether to use validation or not.')

# These variables are passed to the BCAgent class.
agent_config = ml_collections.ConfigDict({
    'goal_conditioned': 0,
    'actor_lr': 3e-4,
    'hidden_dims': (512, 512, 512),
    'opt_decay_schedule': 'none',
    'use_tanh': 0,
    'state_dependent_std': 0,
    'use_layer_norm': 1,
    'fixed_std': 0,
    'actor_tau': 0.0005,
    'eval_target_actor': 0,
})

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'rlbase_default',
    'name': 'bc_{env_name}',
})


config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('agent', agent_config, lock_config=False)
config_flags.DEFINE_config_dict('gcdataset', GCDataset.get_default_config(), lock_config=False)

###############################
#  Agent. Contains the neural networks, training logic, and sampling.
###############################

class BCAgent(flax.struct.PyTreeNode):
    rng: Any
    actor: TrainState
    target_actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @jax.jit
    def update(agent, batch):
        observations = batch['observations']
        actions = batch['actions']

        def actor_loss_fn(actor_params):
            dist = agent.actor(observations, params=actor_params)
            log_probs = dist.log_prob(actions)
            actor_loss = -(log_probs).mean()
            mse_error = jnp.square(dist.loc - actions).mean()

            return actor_loss, {
                'actor_loss': actor_loss,
                'action_std': dist.stddev().mean(),
                'mse_error': mse_error,
            }
    
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)
        new_target_actor = target_update(agent.actor, agent.target_actor, agent.config['actor_tau'])

        return agent.replace(actor=new_actor, target_actor=new_target_actor), {
            **actor_info
        }

    @jax.jit
    def sample_actions(agent, observations: np.ndarray, *, seed: Any, temperature: float = 1.0) -> jnp.ndarray:
        if type(observations) is dict:
            observations = jnp.concatenate([observations['observation'], observations['goal']], axis=-1)
        if agent.config['eval_target_actor']:
            actions = agent.target_actor(observations, temperature=temperature).sample(seed=seed)
        else:
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

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = Policy(config['hidden_dims'], action_dim=action_dim, 
            fixed_std=config['fixed_std'] if config['fixed_std'] != 0 else None,
            log_std_min=-5.0, state_dependent_std=config['state_dependent_std'], tanh_squash_distribution=config['use_tanh'], mlp_kwargs=dict(use_layer_norm=False, activations=nn.relu))

        if config['opt_decay_schedule'] == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-config['actor_lr'], config['max_steps'])
            actor_tx = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            actor_tx = optax.adam(learning_rate=config['actor_lr'])

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)
        target_actor = TrainState.create(actor_def, actor_params)
        config_dict = flax.core.FrozenDict(**config)
        return BCAgent(rng, actor=actor, target_actor=target_actor, config=config_dict)

###############################
#  Run Script. Loads data, logs to wandb, and runs the training loop.
###############################

def main(_):
    if FLAGS.agent.goal_conditioned:
        assert 'gc' in FLAGS.env_name
    else:
        assert 'gc' not in FLAGS.env_name

    # Create wandb logger
    setup_wandb(FLAGS.agent.to_dict(), **FLAGS.wandb)
    
    env = make_env(FLAGS.env_name)
    eval_env = make_env(FLAGS.env_name)

    dataset = get_dataset(env, FLAGS.env_name)

    if FLAGS.agent.goal_conditioned:
        dataset = GCDataset(dataset, **FLAGS.gcdataset.to_dict())
        example_batch = dataset.sample(1)
        example_obs = np.concatenate([example_batch['observations'], example_batch['goals']], axis=-1)
        debug_batch = dataset.sample(100)
        print("Masks Look Like", debug_batch['masks'])
        print("Rewards Look Like", debug_batch['rewards'])
    else:
        example_batch = dataset.sample(1)
        example_obs = dataset.sample(1)['observations']

    if FLAGS.use_validation:
        dataset, dataset_valid = dataset.train_valid_split(0.9)

    agent = create_agent(FLAGS.seed,
                    example_obs,
                    example_batch['actions'],
                    max_steps=FLAGS.max_steps,
                    **FLAGS.agent)
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        batch = dataset.sample(FLAGS.batch_size)
        if FLAGS.agent.goal_conditioned:
            batch['observations'] = np.concatenate([batch['observations'], batch['policy_goals']], axis=-1)

        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

            if FLAGS.use_validation:
                batch = dataset_valid.sample(FLAGS.batch_size)
                if FLAGS.agent.goal_conditioned:
                    batch['observations'] = np.concatenate([batch['observations'], batch['policy_goals']], axis=-1)
                _, valid_update_info = agent.update(batch)
                valid_metrics = {f'validation/{k}': v for k, v in valid_update_info.items()}
                wandb.log(valid_metrics, step=i)

                wandb.log({'training/actor_valid_difference': (valid_update_info['actor_loss'] - update_info['actor_loss'])}, step=i)

        if i % FLAGS.eval_interval == 0:
            record_video = i % FLAGS.video_interval == 0
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info, trajs = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes, record_video=record_video, return_trajectories=True)
            eval_metrics = {}
            for k in ['episode.return', 'episode.length']:
                eval_metrics[f'evaluation/{k}'] = eval_info[k]
                print(f'evaluation/{k}: {eval_info[k]}')
            try:
                eval_metrics['evaluation/episode.return.normalized'] = eval_env.get_normalized_score(eval_info['episode.return'])
                print(f'evaluation/episode.return.normalized: {eval_metrics["evaluation/episode.return.normalized"]}')
            except:
                pass
            if record_video:
                wandb.log({'video': eval_info['video']}, step=i)

            # Antmaze Specific Logging
            if 'antmaze-large' in FLAGS.env_name or 'maze2d-large' in FLAGS.env_name:
                import rlbase.common.envs.d4rl.d4rl_ant as d4rl_ant
                # Make an image of the trajectories.
                traj_image = d4rl_ant.trajectory_image(eval_env, trajs)
                eval_metrics['trajectories'] = wandb.Image(traj_image)

                # Maze2d Action Distribution
                if 'maze2d-large' in FLAGS.env_name:
                    # Make a plot of the actions.
                    traj_actions = np.concatenate([t['action'] for t in trajs], axis=0) # (T, A)
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.scatter(traj_actions[::100, 0], traj_actions[::100, 1], alpha=0.4)
                    wandb.log({'actions_traj': wandb.Image(plt)}, step=i)

                    data_actions = batch['actions']
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.scatter(data_actions[:, 0], data_actions[:, 1], alpha=0.2)
                    wandb.log({'actions_data': wandb.Image(plt)}, step=i)

            wandb.log(eval_metrics, step=i)

if __name__ == '__main__':
    app.run(main)