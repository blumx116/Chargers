from env import State, Action
import tensorflow as tf

class Agent:
    def score(self, observation, context, network='q') -> tf.Tensor:
        pass

    def act(self, observation, context, mode='test', network='q') -> tf.Tensor:
        pass

    def optimize(self) -> None:
        pass

    def remember(self,
            state,
            context,
            action,
            reward,
            next_state,
            next_context,
            done: bool) -> None:
        pass

    def step(self, global_timestep: int) -> None:
        pass

    def log(self, global_timestep: int) -> None:
        pass
