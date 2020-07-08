from copy import deepcopy
import warnings

import numpy.random as rand
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)

from env import load_continuous_simulation
from env.wrappers import SummedRewardWrapper, StaticFlatWrapper, NormalizedPositionWrapper
from agent.dqn import make_model
import wandb

warnings.simplefilter('once')

wandb.login()
wandb.init(project='chargers')

wandb.config.update({
    'algorithm': 'basic dqn',
    'model': 'feedforward',
    'n_layers': 2,
    'n_nodes' : 1000,
    'normalize': True,
    'max_cars': 100,
    'car_speed': 0.1,
    'sample_distance': 0.3,
    'sample_amount': 0.4,
    'date': '06-27-2019',
    'region': 'haidian',
    'learning_rate': 1e-5,
    'replay_size': 100000,
    'target_network_update_f': 10000,
    'log_every': 10000,
    'gamma': 0.99,
    'start_train_ts': 1000,
    'batch_size': 32,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 300000,
    'max_ts': 140000
})

config = wandb.config

sim = load_continuous_simulation(wandb.config)
sim = NormalizedPositionWrapper(sim)
sim = StaticFlatWrapper(sim)

sim = SummedRewardWrapper(sim)
sim.seed(1)
sim.reset()

i = 0
done = False

class DQNModel(nn.Module):
    def __init__(self, in_dims: int, n_actions: int):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Linear(in_dims, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)

from minimal_dqn.main import Agent, get_epsilon, ReplayBuffer, compute_td_loss, hard_update
n_stations: int = sim.unwrapped.engine.n_stations

q_network = make_model(config, sim.observation_space, sim.action_space).cuda()
q_target = deepcopy(q_network).cuda()

env = sim

agent = Agent(sim, q_network, q_target)

optimizer = optim.Adam(q_network.parameters(), lr=config.learning_rate)
replay_buffer = ReplayBuffer(config.replay_size)

losses, all_rewards = [], []
episode_reward = 0
state = env.reset()
n_eps_completed: int = 0

for ts in range(1, config.max_ts + 1):
    epsilon = get_epsilon(
        config.epsilon_start, config.epsilon_end, config.epsilon_decay, ts
    )
    action = agent.act(state, epsilon)

    next_state, reward, done, _ = env.step(int(action.cpu()))
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        n_eps_completed += 1
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(replay_buffer) > config.start_train_ts:
        # Update the q-network & the target network
        loss = compute_td_loss(
            agent, config.batch_size, replay_buffer, optimizer, config.gamma
        )
        losses.append(loss.data)

        if ts % config.target_network_update_f == 0:
            hard_update(agent.q_network, agent.target_q_network)

    if ts % config.log_every == 0:
        if len(losses) > 0 and len(all_rewards) > 0:
            wandb.log({'Timestep' : ts,
                   'Reward': all_rewards[-1],
                   'Loss': losses[-1],
                   'epsilon': epsilon,
                   'n_updates': int(ts / config.target_network_update_f),
                   'n_eps': n_eps_completed})
        out_str = "Timestep {}".format(ts)
        if len(all_rewards) > 0:
            out_str += ", Reward: {}".format(all_rewards[-1])
        if len(losses) > 0:
            out_str += ", TD Loss: {}".format(losses[-1])
        print(out_str)


print()