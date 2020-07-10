from collections import deque
from typing import Union, Tuple, List

import numpy as np
from numpy.random import RandomState

from misc.utils import optional_random, array_random_choice


class ReplayBuffer(object):
    def __init__(self,
            capacity: int,
            random: Union[int, RandomState]):
        self.buffer: deque = deque(maxlen=capacity)
        self.capacity: int =  capacity
        self.random: RandomState = optional_random(random)

    def push(self, state, context, action, reward,
             next_state, next_context, done):
        if not isinstance(state, tuple):
            state = [state]
            next_state = [next_state]
        state = list(map(lambda s: np.expand_dims(s, 0), state))
        next_state = list(map(lambda s: np.expand_dims(s, 0), next_state))
        # state = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, context, action, reward,
                            next_state, next_context, done))

    def _concatenate_states_(self,
            states: List[List[np.ndarray]]) -> List[np.ndarray]:
        n_parts: int = len(states[0])
        assert len(np.unique(list(map(len, states)))) == 1
        # assert all have same number of parts
        return [np.concatenate(list(map(lambda state: state[i], states)), axis=0)
                for i in range(n_parts)]

    def sample(self, batch_size):
        states, contexts, actions, rewards, next_states, next_contexts, dones =\
            zip(*array_random_choice(self.buffer,
                count=batch_size, random=self.random))
        n_parts: int = len(states[0])
        return (self._concatenate_states_(states), contexts, actions, rewards,
                self._concatenate_states_(next_states), next_contexts, dones)

    def __len__(self):
        return len(self.buffer)
