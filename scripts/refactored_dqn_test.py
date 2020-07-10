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
    'n_heads': 4,
    'n_nodes' : 8,
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
    'log_every': 10,
    'diagnostic_every': 1000,
    'gamma': 0.99,
    'start_train_ts': 1000,
    'batch_size': 32,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 300000,
    'max_ts': 140000
})

config = wandb.config

from agent.dqn import DQNAgent, make_model
from agent.diagnostic import train
from env import make_and_wrap_env

sim = make_and_wrap_env(config)

q_model = make_model(config, sim.observation_space, sim.action_space).to(torch.device("cuda:0"))
q_target = deepcopy(q_model)

agent = DQNAgent(q_model, q_model, sim, config, device=torch.device('cuda:0'), random=0)

train(agent, sim, config)