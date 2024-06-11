from flax.core.frozen_dict import FrozenDict
from flax.core import freeze
import dataclasses
import numpy as np
import jax
import ml_collections
import copy

from utils.dataset import Dataset

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    discount: float
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    mask_terminal: int = 1

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 1,
            'discount': 0.99,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'mask_terminal': 1,
        })

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        self.terminal_locs = np.concatenate([self.terminal_locs, [self.dataset.size-1]])
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        
        # Goals from the same trajectory
        closest_goal = np.searchsorted(self.terminal_locs, indx)
        closest_goal = np.clip(closest_goal, 0, len(self.terminal_locs)-1)
        final_state_indx = self.terminal_locs[closest_goal]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)

        return goal_indx

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        # Different policy goals than value goals. Uniformly sample a point in the trajectory.
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        distance = np.random.rand(batch_size)
        policy_traj_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
        batch['policy_goals'] = jax.tree_map(lambda arr: arr[policy_traj_goal_indx], self.dataset['observations'])

        if self.mask_terminal:
            batch['masks'] = 1.0 - success.astype(float)
        else:
            batch['masks'] = np.ones(batch_size)

        return batch

    def train_valid_split(self, ratio):
        # split into trajectories (assuming trajectories are even).
        # then randomize the order, and take only the first ratio of the trajectories.
        # then flatten the trajectories.

        diff = np.diff(self.terminal_locs)[:-1]
        traj_len = diff.mean().astype(int)
        assert len(self.terminal_locs) > 2
        assert np.equal(diff, np.ones_like(diff) * traj_len).all() # make sure the trajectories are of the same length.
        
        reshaped_dataset = jax.tree_map(lambda arr: np.reshape(arr, (-1, traj_len, *arr.shape[1:])), self.dataset._dict)
        shuffled_indicies = np.random.permutation(reshaped_dataset['observations'].shape[0])
        split = int(len(shuffled_indicies) * ratio)
        train_indx = shuffled_indicies[:split]
        valid_indx = shuffled_indicies[split:]

        dataset_train = jax.tree_map(lambda arr: arr[train_indx], reshaped_dataset)
        dataset_valid = jax.tree_map(lambda arr: arr[valid_indx], reshaped_dataset)

        dataset_train = jax.tree_map(lambda arr: np.reshape(arr, (-1, *arr.shape[2:])), dataset_train)
        dataset_valid = jax.tree_map(lambda arr: np.reshape(arr, (-1, *arr.shape[2:])), dataset_valid)
        
        dataset_train = Dataset.create(**dataset_train)
        dataset_valid = Dataset.create(**dataset_valid)
        dataset_config = dataclasses.asdict(self)
        del dataset_config['dataset']
        return GCDataset(dataset_train, **dataset_config), GCDataset(dataset_valid, **dataset_config)

def flatten_obgoal(obgoal):
    return np.concatenate([obgoal['observation'], obgoal['goal']], axis=-1)