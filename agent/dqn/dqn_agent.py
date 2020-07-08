from typing import Union, Iterator

import numpy as np
from numpy.random import RandomState
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from wandb.util import PreInitObject as Config

from agent import Agent
from agent.dqn import ReplayBuffer
from env import ContinuousSimulation, State
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

    def _convert_state(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.float32(state)).type(self.dtype)

    def q_value(self,
            state: Union[State, np.ndarray],
            network: nn.Module) -> torch.Tensor:
        if isinstance(state, list):
            state = list(map(self._convert_state, state))
            return network.forward(*state)
        elif isinstance(state, np.ndarray) and state.dtype == object:
            state = list()
        else:
            state = self._convert_state(state)
            return network.forward(state)

    def _add_batch_dim(self, state: np.ndarray) -> np.ndarray:
        return state[np.newaxis, ...]

    def act(self, state, epsilon):
        """DQN action - max q-value w/ epsilon greedy exploration."""
        if self.random.random() > epsilon:
            if isinstance(state, tuple):
                state = tuple(map(self._add_batch_dim, state))
            else:
                # should be the single item case
                state = self._add_batch_dim(state)
            q_value = self.q_value(state, self.q_network)
            return q_value.max(1)[1].data[0]
        return torch.tensor(self.random.randint(self.env.action_space.n))

    def seed(self, random: Union[int, RandomState]) -> None:
        self.random = optional_random(random)

    def train(self, config=None, env_seeds: Iterator[int] = None):
        config = config if config is not None else self.config

        optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        replay_buffer = ReplayBuffer(config.replay_size, self.random)

        losses, all_rewards = [], []
        episode_reward = 0
        if env_seeds is not None:
            self.env.seed(next(env_seeds))
        state = self.env.reset()
        n_eps_completed: int = 0
        for ts in range(1, config.max_ts + 1):
            epsilon = get_epsilon(
                config.epsilon_start, config.epsilon_end, config.epsilon_decay, ts)

            action = self.act(state, epsilon)

            next_state, reward, done, _ = self.env.step(int(action.cpu()))
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                n_eps_completed += 1
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(replay_buffer) > config.start_train_ts:
                # Update the q-network & the target network
                loss = self.compute_td_loss(
                    config.batch_size, replay_buffer, config.gamma, optimizer)
                losses.append(loss.data)

                if ts % config.target_network_update_f == 0:
                    hard_update(self.q_network, self.target_q_network)

            if ts % config.log_every == 0:
                if len(losses) > 0 and len(all_rewards) > 0:
                    wandb.log({'Timestep': ts,
                               'Reward': all_rewards[-1],
                               'Loss': losses[-1],
                               'epsilon': epsilon,
                               'n_updates': int(ts / config.target_network_update_f),
                               'n_eps': n_eps_completed})

    def compute_td_loss(self, batch_size, replay_buffer, gamma, optimizer=None):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        # state = torch.tensor(np.float32(state)).type(self.dtype)
        # next_state = torch.tensor(np.float32(next_state)).type(self.dtype)
        action = torch.tensor(action).type(self.dtypelong)
        reward = torch.tensor(reward).type(self.dtype)
        done = torch.tensor(done).type(self.dtype)

        # Normal DDQN update
        q_values = self.q_value(state, self.q_network)
        # q_values = self.q_network(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # double q-learning
        online_next_q_values = self.q_value(self.q_network, next_state)
        # online_next_q_values = self.q_network(next_state)
        _, max_indicies = torch.max(online_next_q_values, dim=1)
        target_q_values = self.q_value(next_state, self.target_q_network)
        # target_q_values = self.target_q_network(next_state)
        next_q_value = torch.gather(target_q_values, 1, max_indicies.unsqueeze(1))

        expected_q_value = reward + gamma * next_q_value.squeeze() * (1 - done)
        loss = (q_value - expected_q_value.data).pow(2).mean()
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss
