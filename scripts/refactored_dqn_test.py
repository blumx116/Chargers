from copy import deepcopy
import warnings

import numpy.random as rand
import numpy as np
import torch

from agent.diagnostic import train, diagnostic
from agent import make_agent
from env import make_and_wrap_env
from misc.wandb_utils import init_config, use_wandb


warnings.simplefilter('once')

settings = {
    'algorithm': 'dqn',
    'model': 'transformer',
    'n_layers': 2,
    'n_heads': 4,
    'n_nodes' : 8,
    'normalize': True,
    'max_cars': 200,
    'car_speed': 0.1,
    'sample_distance': 0.3,
    'sample_amount': 5,
    'date': '06-27-2019',
    'region': 'haidian',
    'learning_rate': 1e-5,
    'replay_size': 100000,
    'target_network_update_freq': 10000,
    'log_every': 10,
    'test_every': 100,
    'gamma': 0.99,
    'start_train_ts': 1000,
    'batch_size': 32,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 300000,
    'max_ts': 140000,
    'seed': 0}

torch.manual_seed(settings['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(settings['seed'])

sim = make_and_wrap_env(**settings)

settings.update({
    'observation_space': sim.observation_space,
    'action_space': sim.action_space,
    'device': torch.device("cuda:0"),
})

init_config(settings, project='chargers', force_wandb=True)

agent = make_agent(**settings)
test = lambda agent: diagnostic(
    agent=agent, env=sim,
    use_wandb=settings['use_wandb'], seeds=iter(range(1, 10)))
train(agent, sim, test=test, **settings)