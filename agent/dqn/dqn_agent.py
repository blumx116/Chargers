from typing import Union, Iterator, Tuple, Any

import numpy as np
from numpy.random import RandomState
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from agent import Agent
from agent.dqn import ReplayBuffer
from env import ContinuousSimulation, State
from misc.config import Config
from misc.utils import optional_random
from minimal_dqn.main import get_epsilon, hard_update


class DQNAgent(Agent):
    def __init__(self,
            source_network: nn.Module,
            target_network: nn.Module,
            env: ContinuousSimulation,
            config: Config,
            device: torch.device = None,
            random: Union[int, RandomState] = None):
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.q_network: nn.Module = source_network.to(self.device)
        self.target_q_network: nn.Module = target_network.to(self.device)
        self.env: ContinuousSimulation = env
        self.config: Config = config
        self.n_actions: int = env.action_space.n
        self.random: RandomState = optional_random(random)
        self.dtype = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
        self.dtypelong = torch.cuda.LongTensor if self.device.type == 'cuda' else torch.LongTensor
        self.loss = nn.MSELoss()
        self.gamma = config.gamma
        self.replay_buffer = ReplayBuffer(config.replay_size, random)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

    def act(self,
            observation: Union[State, np.ndarray, Tuple],
            context: Any,
            mode: str = 'test',
            network: str = 'q'):
        if isinstance(observation, tuple):
            state = tuple(map(self._add_batch_dim, observation))
        else:
            state = self._add_batch_dim(observation)
        q_value = self.score(state, context, network)
        return q_value.max(1)[1].data[0]

    def score(self,
            observation: Union[State, np.ndarray, Tuple],
            context: Any,
            network: str = 'q') -> torch.Tensor:
        assert network in ['q', 'target']
        network = self.q_network if network == 'q' else self.target_q_network
        if isinstance(observation, list) or isinstance(observation, tuple):
            state = list(map(self._convert_state, observation))
            return network.forward(*state)
        else:
            state = self._convert_state(observation)
            return network.forward(state)

    def optimize(self) -> torch.Tensor:
        self.optimizer.zero_grad()
        loss = self.compute_td_loss(
            *self.replay_buffer.sample(self.config.batch_size))
        loss.backward()
        self.optimizer.step()
        return loss

    def compute_td_loss(self, state, context, action, reward,
                        next_state, next_context, done):
        action = torch.tensor(action).type(self.dtypelong)
        reward = torch.tensor(reward).type(self.dtype)
        done = torch.tensor(done).type(self.dtype)

        # Normal DDQN update
        q_values = self.score(state, self.q_network)  # (n_stations=actions, batch_size)
        # q_values = self.q_network(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # double q-learning
        online_next_q_values = self.score(next_state, next_context, 'q')
        # online_next_q_values = self.q_network(next_state)
        _, max_indices = torch.max(online_next_q_values, dim=1)
        target_q_values = self.score(next_state, next_context, 'target')
        # (n_stations, batch_size)
        # target_q_values = self.target_q_network(next_state)
        next_q_value = torch.gather(target_q_values, 1, max_indices.unsqueeze(1))

        expected_q_value = reward + self.gamma * next_q_value.squeeze() * (1 - done)
        return self.loss(q_value, expected_q_value)

    def remember(self,
            state,
            context,
            action,
            reward,
            next_state,
            next_context,
            done: bool):
        self.replay_buffer.push(
            state, context, action, reward,
            next_state, next_context, done)

    def step(self, global_timestep: int) -> None:
        pass

    def log(self, global_timestep: int) -> None:
        ...

    def _add_batch_dim(self, state: np.ndarray) -> np.ndarray:
        return state[np.newaxis, ...]

    def _convert_state(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.float32(state)).type(self.dtype)
