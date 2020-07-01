from dataclasses import dataclass
from typing import TypeVar, List, Generic, Optional


State = TypeVar("State")
Reward = TypeVar("Reward")
Action = TypeVar("Action")
OptionData = TypeVar("OptionData")

@dataclass 
class Option(Generic[OptionData]):
    value: OptionData
    depth: int 

@dataclass
class Transition(Generic[State, Action, Reward]):
    state: State 
    action: Action
    reward: Reward 

class Trajectory(
    Generic[State, Action, Reward], 
    List[Transition[State, Action, Reward]]):
    pass 

@dataclass
class TrainSample(Generic[State, Action, Reward, OptionData]):
    prev_option: Optional[Option[OptionData]]
    initial_state: State
    suboption_trajectory: Trajectory[State, Action, Reward]
    suboption: Option[OptionData]
    midpoint_state: State 
    option_trajectory: Trajectory[State, Action, Reward]
    option: Option[OptionData]
    terminal_state: State


lambda sample: self._get_v_target_(
    sample.initial_state,
    sample.prev_option,
    sample.suboption,
    sample.option,
    sample.suboption_trajectory,
    sample.option_trajectory),
