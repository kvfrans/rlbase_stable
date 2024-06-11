###############################
#
#  Tools for evaluating policies in environments.
#
###############################


from typing import Dict
import gym
import numpy as np
from collections import defaultdict
import time
import wandb


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(policy_fn, env: gym.Env, num_episodes: int, record_video : bool = False, 
             return_trajectories=False):
    """
    Evaluates a policy in an environment by running it for some number of episodes,
    and returns average statistics for metrics.
    """
    stats = defaultdict(list)
    frames = []
    trajectories = []
    for i in range(num_episodes):
        now = time.time()
        trajectory = defaultdict(list)
        ob_list = []
        ac_list = []
        observation, done = env.reset(), False
        ob_list.append(observation)
        while not done:
            if type(observation) == dict:
                obgoal = np.concatenate([observation['observation'], observation['goal']])
                action = policy_fn(obgoal)
            else:
                action = policy_fn(observation)
            action = np.array(action)
            next_observation, r, done, info = env.step(action)
            mask = float(not done or 'TimeLimit.truncated' in info)
            add_to(stats, flatten(info))

            if type(observation) is dict:
                obs_pure = observation['observation']
                next_obs_pure = next_observation['observation']
            else:
                obs_pure = observation
                next_obs_pure = next_observation
            transition = dict(
                observation=obs_pure,
                next_observation=next_obs_pure,
                action=action,
                reward=r,
                done=done,
                mask=mask,
                info=info,
            )
            observation = next_observation
            ob_list.append(observation)
            ac_list.append(action)
            add_to(trajectory, transition)

            if i == 0 and record_video:
                frames.append(env.render(mode="rgb_array"))
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        print("Finished Episode", i, "in", time.time() - now, "seconds")

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if 'episode.return' in stats:
        print("Episode Return Mean is ", stats['episode.return'])

    if record_video:
        stacked = np.stack(frames)
        stacked = stacked.transpose(0, 3, 1, 2)
        while stacked.shape[2] > 160:
            stacked = stacked[:, :, ::2, ::2]
        stats['video'] = wandb.Video(stacked, fps=60)

    if return_trajectories:
        return stats, trajectories
    else:
        return stats