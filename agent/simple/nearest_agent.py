from typing import Union

from gym.spaces import Discrete
import numpy as np
from numpy.random import RandomState
import torch

from agent import Agent
from env import State
from env.internals.continuous_simulation_engine import get_distances
from misc.utils import optional_random


class NearestAgent(Agent):
    def __init__(self,
            action_space: Discrete,
            device: torch.device = None,
            random: Union[int, RandomState] = None,
            **kwargs):
        """
        :param action_space: gym.spaces.Discrete
            action space of environment to act in
        :param (optional) device: torch.device
            device to put returned values on
        :param (optional) random: Union[int, RandomState]
            numpy random state. defaults to global random
        :param kwargs: for compatibility
        """
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device: torch.device = device
        self.n_actions: int = action_space.n
        self.random: RandomState = optional_random(random)

    def score(self,
            observation: State,
            context: State,
            network='q') -> torch.Tensor:
        """
        gives closer stations higher scores
        :param observation: raw from unwrapped environment
        :param context: same as observations
        :param network: ignored, present for compatibility
        :return: torch[f32, device] : [1, n_stations]
        """
        scores: np.ndarray = -get_distances(observation.query_loc,
                stations=observation.station_locations)
        scores: torch.Tensor = torch.from_numpy(scores)
        # [n_stations, ]
        return scores.type(torch.float32).to(self.device).unsqueeze((0))

    def remember(self,
            state,
            context,
            action,
            reward,
            next_state,
            next_context,
            done: bool) -> None:
        pass

    def act(self,
            observation: State,
            context: State,
            mode='test',
            network='q') -> torch.Tensor:
        """
        gives closer stations higher scores
        :param observation: raw from unwrapped environment
        :param context: same as observations
        :param mode: ignored, present for compatibility
        :param network: ignored, present for compatibility
        :return: torch[int64, device] : max_index
        """
        if not np.all(observation.query_loc == 0):
            # actual query, calculate action
            scores: torch.Tensor = self.score(observation, observation)
            # Tensor[f32, dev] : [1, n_stations]
            return scores.max(1)[1].data[0]
        else:
            # give back random action - no actual action to be taken
            action: int = self.random.randint(low=0, high=self.n_actions-1, size=(1,))
            return torch.from_numpy(action).type(torch.int64).to(self.device)

    def optimize(self) -> None:
        pass

    def step(self, global_timestep: int) -> None:
        pass

    def log(self, global_timestep: int) -> None:
        pass
