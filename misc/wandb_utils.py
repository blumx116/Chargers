import sys
from typing import Dict, Any, Union

import wandb


def log(
        wandb: bool,
        values: Dict[str, Any],
        **kwargs) -> None:
    if wandb:
        wandb.log(values, **kwargs)
    else:
        for key in values.keys():
            print(key, end=' : ')
            print(values[key])


def use_wandb(
        force_wandb: bool = False) -> bool:
    return force_wandb or (sys.gettrace() is None)


def init_config(
        values: Dict[str, Any],
        project: str,
        force_wandb: bool) -> None:
    if 'wandb' not in values:
        values['wandb'] = use_wandb(force_wandb)
    if values['wandb'] or force_wandb:
        print("RUNNING WITH WANDB")
        wandb.login()
        wandb.init(project=project)
        wandb.config.update(values)
    else:
        print("RUNNING WITHOUT WANDB")
