from typing import List
from gym.spaces import Space, Discrete, Box, Dict

import torch
import torch.nn as nn
import numpy as np

from misc.config import Config


def FeedforwardModel(
        in_dims: int,
        n_layers: int,
        n_nodes: int,
        n_actions: int,
        normalize: True) -> nn.Module:
    layers: List[nn.Module] = []
    hidden_layers: List[int] = [n_nodes] * n_layers
    ins: List[int] = [in_dims] + hidden_layers
    outs: List[int] = hidden_layers + [n_actions]
    for idx, (in_d, out_d) in enumerate(zip(ins, outs)):
        layers.append(nn.Linear(in_d, out_d))
        layers.append(nn.ReLU())
        if normalize and idx != (len(outs) - 1):
            # don't normalize last output
            layers.append(nn.LayerNorm(out_d))

    return nn.Sequential(*layers)


class TransformerModel(nn.Module):
    def __init__(self,
            car_dims: int,
            station_dims: int,
            n_actions: int,
            n_layers: int,
            n_nodes: int,
            n_heads: int,
            normalize: bool = True):
        super().__init__()
        assert n_layers > 0
        self.car_encoder: nn.Module = nn.Linear(car_dims, n_nodes)
        self.station_encoder: nn.Module = nn.Linear(station_dims, n_nodes)
        self.transformer: nn.Module = nn.Transformer(
            d_model=n_nodes,
            dim_feedforward=int(n_nodes * n_heads / 2),
            num_decoder_layers=n_layers,
            num_encoder_layers=n_layers,
            nhead=n_heads)
        self.linear_activation: nn.Module = nn.Linear(n_nodes, 1)

    def forward(self, stations: torch.Tensor, cars: torch.Tensor) -> torch.Tensor:
        """

        :param cars: torch.Tensor[f32]: (batch_dim, n_cars, car_dims)
        :param stations: torch.Tensor[f32]: (batch_dim, n_cars, car_dims)
        :return: stations: torch.Tensor[f32]: (batch_dim, n_stations, 1)
        """
        cars: torch.Tensor = self.car_encoder(cars)  # (batch_dim, n_cars, embed_dim)
        stations: torch.Tensor = self.station_encoder(stations)  # (batch_dim, n_stations, embed_dim)
        cars = cars.permute(1, 0, 2)  # (n_cars, batch_dim, embed_dim)
        stations = stations.permute(1, 0, 2)  # (n_stations, batch_dim, embed_dim)
        transformed = self.transformer(cars, stations)  # (n_stations, batch_dim, embed_dim)
        values = self.linear_activation(transformed)  # (n_stations, batch_dim, 1)
        return values.squeeze(2).permute(1,0)  # (batch_dim, n_stations)


def make_model(config: Config, observation_space: Space, action_space: Space) -> nn.Module:
    assert isinstance(action_space, Discrete)
    n_actions: int = action_space.n
    model: str = config.model.lower()
    if model == 'feedforward':
        assert isinstance(observation_space, Box)
        in_dims = observation_space.shape[0]
        return FeedforwardModel(in_dims, config.n_layers, config.n_nodes, n_actions, config.normalize)
    if model == 'transformer':
        assert isinstance(observation_space, Dict)
        assert 'cars' in observation_space.spaces and 'stations' in observation_space.spaces
        car_dims: int = observation_space['cars'].shape[1]
        station_dims: int = observation_space['stations'].shape[1]
        return TransformerModel(car_dims, station_dims, n_actions, config.n_layers, config.n_nodes, config.n_heads, config.normalize)
