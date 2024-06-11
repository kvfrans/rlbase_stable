import d4rl
import d4rl.gym_mujoco
import gym
import numpy as np
from jax import tree_util

import d4rl_ant
from utils.dataset import Dataset


# Note on AntMaze. Reward = 1 at the goal, and Terminal = 1 at the goal.
# Masks = Does the episode end due to final state?
# Dones_float = Does the episode end due to time limit? OR does the episode end due to final state?
def get_dataset(env: gym.Env, env_name: str, clip_to_eps: bool = True,
                eps: float = 1e-5, dataset=None, filter_terminals=False, obs_dtype=np.float32):
    if dataset is None:
        dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

    # Mask everything that is marked as a terminal state.
    # For AntMaze, this should mask the end of each trajectory.
    masks = 1.0 - dataset['terminals']

    # In the AntMaze data, terminal is 1 when at the goal. But the episode doesn't end. 
    # This just ensures that we treat AntMaze trajectories as non-ending.
    if "antmaze" in env_name or "maze2d" in env_name:
        dataset['terminals'] = np.zeros_like(dataset['terminals'])

    if 'antmaze' in env_name and 'discrete' in env_name:
        print("Discretizing AntMaze observations.")
        print("Raw observations looks like", dataset['observations'].shape[1:])
        dataset['observations'] = d4rl_ant.discretize_obs(dataset['observations'])
        dataset['next_observations'] = d4rl_ant.discretize_obs(dataset['next_observations'])
        print("Discretized observations looks like", dataset['observations'].shape[1:])

    # Compute dones if terminal OR orbservation jumps.
    dones_float = np.zeros_like(dataset['rewards'])

    imputed_next_observations = np.roll(dataset['observations'], -1, axis=0)
    same_obs = np.all(np.isclose(imputed_next_observations, dataset['next_observations'], atol=1e-5), axis=-1)
    dones_float = 1.0 - same_obs.astype(np.float32)
    dones_float += dataset['terminals']
    dones_float[-1] = 1.0
    dones_float = np.clip(dones_float, 0.0, 1.0)

    observations = dataset['observations'].astype(obs_dtype)
    next_observations = dataset['next_observations'].astype(obs_dtype)

    return Dataset.create(
        observations=observations,
        actions=dataset['actions'].astype(np.float32),
        rewards=dataset['rewards'].astype(np.float32),
        masks=masks.astype(np.float32),
        dones_float=dones_float.astype(np.float32),
        next_observations=next_observations,
    )

def get_normalization(dataset):
    returns = []
    ret = 0
    for r, term in zip(dataset['rewards'], dataset['dones_float']):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000

def normalize_dataset(env_name, dataset):
    print("Normalizing", env_name)
    if 'antmaze' in env_name or 'maze2d' in env_name:
        return dataset.copy({'rewards': dataset['rewards']- 1.0})
    else:
        normalizing_factor = get_normalization(dataset)
        print(f'Normalizing factor: {normalizing_factor}')
        dataset = dataset.copy({'rewards': dataset['rewards'] / normalizing_factor})
        return dataset