import sys
from typing import Dict, Any, Union

import wandb
from wandb.util import PreInitObject as wandbConfig


class GenericConfig:
    def __init__(self, values: Dict[str, Any]):
        self._values: Dict[str, Any] = values

    def __getattr__(self, item: str):
        return self._values[item]

    def __setattr__(self, key: str, value: Any):
        if key[0] == '_':
            super().__setattr__(key, value)
        else:
            self._values[key] = value

    def __getitem__(self, item: str) -> Any:
        return self._values[item]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = value


Config = Union[wandbConfig, GenericConfig]


def log(
        config: Config,
        values: Dict[str, Any],
        **kwargs) -> None:
    if config['wandb']:
        wandb.log(values, **kwargs)
    else:
        for key in values.keys():
            print(key, end=' : ')
            print(values[key])


def init_config(
        values: Dict[str, Any],
        project: str,
        log_debug: bool = False) -> Config:
    values['wandb'] = log_debug or (sys.gettrace() is None)
    if values['wandb']:
        print("RUNNING WITH WANDB")
        wandb.login()
        wandb.init(project=project)
        wandb.config.update(values)
        return wandb.config
    else:
        print("RUNNING WITHOUT WANDB")
        return GenericConfig(values)
