from typing import Union, Iterator, Tuple, Any

import gym
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers

from agent import Agent
from agent.dqn import ReplayBuffer
from env import ContinuousSimulation, State
from misc.utils import optional_random, optional_device, kwargify


class DQNAgent(Agent):
    def __init__(self,
            source_network: keras.Model,
            target_network: keras.Model,
            action_space: gym.spaces.Discrete,
            batch_size: int,
            replay_size: int,
            device: tf.device = None,
            gamma: float = 0.99,
            learning_rate: float = 3e-4,
            random: Union[int, RandomState] = None,
            target_update_freq: int = 1000,
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
        :param replay_size: int
            number of instances to save in memory
        :param (optional) device: tf.device
            defaults to first gpu if available
        :param (optional) gamma: float in (0, 1)
            defaults to 0.99
        :param (optional) learning_rate: float
            defaults to 3e-4
        :param (optional) random: Union[int, RandomState] initial seed
            defaults to using global random
        :param kwargs: used for ReplayBuffer
        """
        kwargs = kwargify(locals())
        self.device: tf.device = optional_device(device)
        self.n_actions: int = action_space.n
        self.random: RandomState = optional_random(random)
        self.gamma = gamma
        learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.target_update_freq: int = target_update_freq
        assert self.batch_size > 0

        self.q_network: keras.Model = source_network
        self.target_q_network: keras.Model = target_network

        self.replay_buffer = ReplayBuffer(capacity=replay_size, **kwargs)
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

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
        return tf.math.argmax(q_value, axis=1).numpy()

    def score(self,
            observation: Union[State, np.ndarray, Tuple],
            context: Any,
            network: str = 'q') -> tf.Tensor:
        assert network in ['q', 'target']
        network = self.q_network if network == 'q' else self.target_q_network
        return network(observation)

    def update_target_network(self) -> None:
        self.target_q_network.set_weights(self.q_network.get_weights())

    def optimize(self) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self.compute_td_loss(
                *self.replay_buffer.sample(self.batch_size))
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        return loss

    def compute_td_loss(self, states, contexts, actions, rewards,
                        next_states, next_contexts, dones):
        """
        action = tf.Tensor(action).to(self.device)
        reward = tf.Tensor(reward).to(self.device)
        done = tf.Tensor(done).to(self.device)
        """
        actions: tf.Tensor = tf.squeeze(tf.constant(actions), 1)
        # actions : tf.Tensor[i32, dev] : (batch,)
        rewards = tf.constant(rewards, dtype=tf.float32)
        # rewards: tf.Tensor[f32, dev] : (batch, )
        dones = tf.constant(dones, dtype=tf.float32)
        # dones: tf.Tensor[f32, dev] : (batch, )

        # Normal DDQN update
        q_values = self.score(states, contexts, 'q')  # (n_stations=actions, batch_size)
        # q_values = self.q_network(state)

        xs = tf.constant(list(range(len(actions))))
        idxs: tf.Tensor = tf.stack([xs, actions],
                axis=0)
        # idxs: tf.Tensor[i32, dev] : (2, batch)
        idxs = tf.transpose(idxs)
        # idxs: tf.Tensor[i32, dev] : (batch, 2)
        chosen_q_values: tf.Tensor = tf.gather_nd(q_values, idxs)
        # chosen_q_values: tf.Tensor[f32, dev] : (batch,)

        # double q-learning
        online_next_q_values = self.score(next_states, next_contexts, 'q')
        # online_next_q_values : tf.Tensor[f32, dev] : (batch, n_stations)
        max_indices = tf.argmax(online_next_q_values, axis=1)
        # max_indices : tf.Tensor[i64, dev] : (batch, )
        max_indices = tf.cast(max_indices, dtype=tf.int32)
        # tf.Tensor[i64, dev] : (batch, )
        # _, max_indices = tf.max(online_next_q_values, dim=1)
        target_q_values = self.score(next_states, next_contexts, 'target')
        # (n_stations, batch_size)
        # target_q_values = self.target_q_network(next_state)
        # next_q_value = tf.gather(target_q_values, 1, max_indices.unsqueeze(1))
        next_idxs = tf.transpose(tf.stack([xs, max_indices]))
        # next_idxs : tf.Tensor[i32, dev] : (batch, 2)
        next_q_values = tf.gather_nd(target_q_values, next_idxs)
        # next_q_values : tf.Tensor[f32, dev] : (batch, )

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        # expected_q_values: tf.Tensor[f32, dev] : (batch, )
        return keras.losses.MSE(chosen_q_values, expected_q_values)

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
        if global_timestep % self.target_update_freq == 0:
            self.update_target_network()

    def log(self,
            global_timestep: int,
            writer: tf.summary.SummaryWriter) -> None:
        ...

    def _add_batch_dim(self, state: np.ndarray) -> np.ndarray:
        return state[np.newaxis, ...]
