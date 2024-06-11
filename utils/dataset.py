###############################
#  Dataset Pytrees for offline data, replay buffers, etc.
#  See: https://github.com/dibyaghosh/jaxrl_m/blob/main/jaxrl_m/dataset.py
###############################

import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import tree_util


def get_size(data) -> int:
    sizes = tree_util.tree_map(lambda arr: len(arr), data)
    return max(tree_util.tree_leaves(sizes))


# A class for storing (and retrieving batches of) data in nested dictionary format.
class Dataset(FrozenDict):
    @classmethod
    def create(cls, observations, actions, rewards, masks, next_observations, freeze=True, **extra_fields):
        data = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "next_observations": next_observations,
            **extra_fields,
        }
        # Force freeze
        if freeze:
            tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

    def get_size(self):
        return self.size

    def sample(self, batch_size: int, indx=None):
        """
        Sample a batch of data from the dataset. Use `indx` to specify a specific
        set of indices to retrieve. Otherwise, a random sample will be drawn.

        Returns a dictionary with the same structure as the original dataset.
        """
        if indx is None:
            indx = np.random.randint(self.size, size=batch_size)
        return self.get_subset(indx)

    def get_subset(self, indx):
        return tree_util.tree_map(lambda arr: arr[indx], self._dict)

    def train_valid_split(self, ratio):
        # return two new Dataset objects, split by ratio.
        shuffled_indicies = np.random.permutation(self.size)
        split = int(self.size * ratio)
        train_indx = shuffled_indicies[:split]
        valid_indx = shuffled_indicies[split:]
        return Dataset.create(**self.get_subset(train_indx)), Dataset.create(**self.get_subset(valid_indx))

# Dataset where data is added to the buffer.
class ReplayBuffer(Dataset):

    @classmethod
    def create(cls, transition, size: int):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset: dict, size: int):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        self.size = self.pointer = 0
