from typing import Union, Iterator, Tuple, Any

import numpy as np
from numpy.random import RandomState
import wandb

from agent import Agent
from agent.dqn import ReplayBuffer
from env import ContinuousSimulation, State
from misc.utils import optional_random
from minimal_dqn.main import get_epsilon, hard_update


class DQNAgent(Agent):
    def __init__(self,
            source_network: nn.Module,
            target_network: nn.Module,
            **kwargs):
        """
        :param source_network: nn.Module[f32, dev] : State* => [batch, n_stations]
            Network used for scoring
            State should be expected state  format with the first dimension
            reserved for batches
        :param target_network: Should basically be a copy of source_network
        :param action_space: gym.spaces.Discrete
        :param batch_size: int
            batch_size for training
        :param (optional) device: torch.device
            defaults to first gpu if available
        :param (optional) gamma: float in (0, 1)
            defaults to 0.99
        :param (optional) learning_rate: float
            defaults to 3e-4
        :param (optional) random: Union[int, RandomState] initial seed
            defaults to using global random
        :param kwargs: used for ReplayBuffer
        """
        device: torch.device = kwargs.get('device', None)
        if device is None:
            device= torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device: torch.device = device
        self.n_actions: int = kwargs['action_space'].n
        self.random: RandomState = optional_random(kwargs.get('random', None))
        self.gamma = kwargs.get('gamma', 0.99)
        learning_rate: float = kwargs.get('learning_rate', 3e-4)
        self.batch_size: int = kwargs['batch_size']

        self.q_network: nn.Module = source_network.to(self.device)
        self.target_q_network: nn.Module = target_network.to(self.device)
        self.loss = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=kwargs.get('replay_size'), **kwargs)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

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
            *self.replay_buffer.sample(self.batch_size))
        loss.backward()
        self.optimizer.step()
        return loss

    def compute_td_loss(self, state, context, action, reward,
                        next_state, next_context, done):
        action = torch.tensor(action).to(self.device).type(torch.int64)
        reward = torch.tensor(reward).to(self.device).type(torch.float32)
        done = torch.tensor(done).to(self.device).type(torch.float32)

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
        return torch.tensor(np.float32(state)).type(torch.float32).to(self.device)
