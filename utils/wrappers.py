###############################
#
#  Wrappers on top of gym environments
#
###############################

from typing import Dict
import gym
import numpy as np
import time
from collections import deque
from typing import Optional

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        self._reset_stats()
        return self.env.reset(**kwargs)

class RewardOverride(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reward_fn = None

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)
        reward = self.reward_fn(observation)
        return observation, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        return self.env.reset(**kwargs)
    
class NormalizeActionWrapper(gym.Wrapper):
    """A wrapper that maps actions from [-1,1] to [low, hgih]."""
    def __init__(self, env):
        super().__init__(env)
        self.active = type(env.action_space) == gym.spaces.Box
        if self.active:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_scale = (self.action_high - self.action_low) * 0.5
            self.action_mid = (self.action_high + self.action_low) * 0.5
            if not np.isclose(self.action_low[0], 1) or not np.isclose(self.action_high[0],1):
                print("Normalizing Action Space from [{}, {}] to [-1, 1]".format(self.action_low[0], self.action_high[0]))
    def step(self, action):
        if self.active:
            action = np.clip(action, -1, 1)
            action = action * self.action_scale
            action = action + self.action_mid
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def stack_obs(obs):
    return np.stack(obs, axis=0)

def space_stack(space: gym.Space, repeat: int):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError()
    
class ChunkingWrapper(gym.Wrapper):
    """
    Enables observation histories and receding horizon control.

    Accumulates observations into obs_horizon size chunks. Starts by sammpling random actions.

    Executes act_exec_horizon actions in the environment.
    """

    def __init__(self, env: gym.Env, obs_horizon: int, act_exec_horizon: Optional[int] = None):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        self.current_obs = deque(maxlen=self.obs_horizon)

        self.observation_space = space_stack(
            self.env.observation_space, self.obs_horizon
        )
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(
                self.env.action_space, self.act_exec_horizon
            )

    def step(self, action, *args):
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]
            act_exec_horizon = 1

        assert len(action) >= act_exec_horizon

        for i in range(act_exec_horizon):
            obs, reward, done, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)
        return (stack_obs(self.current_obs), reward, done, info)

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.current_obs.clear()
        self.current_obs.append(obs)
        while len(self.current_obs) < self.obs_horizon:
            self.step(self.env.action_space.sample())
        return stack_obs(self.current_obs)