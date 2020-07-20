import sys
from typing import Dict, Any, Union

import wandb


def log(
        use_wandb: bool,
        values: Dict[str, Any],
        **kwargs) -> None:
    if use_wandb:
        wandb.log(values, **kwargs)
    else:
        for key in values.keys():
            print(key, end=' : ')
            print(values[key])

def log_histogram(
        use_wandb: bool,
        attribute: str,
        values: Dict[str, Any],
        **kwargs) -> None:
    if not use_wandb:
        log(use_wandb, values)
    else:
        for key in values.keys():
            wandb.log({key: wandb.Histogram(values[key])})

def use_wandb(
        force_wandb: bool = False) -> bool:
    return force_wandb or (sys.gettrace() is None)


def init_config(
        values: Dict[str, Any],
        project: str,
        force_wandb: bool) -> None:
    if 'use_wandb' not in values:
        values['use_wandb'] = use_wandb(force_wandb)
    if values['use_wandb'] or force_wandb:
        print("RUNNING WITH WANDB")
        wandb.login()
        wandb.init(project=project)
        wandb.config.update(values)
    else:
        print("RUNNING WITHOUT WANDB")
