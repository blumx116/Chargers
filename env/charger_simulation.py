########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
File: simulation.py
Author: bil(bil@baidu.com)
Date: 2019/07/24 11:50:30
"""
import random
import numpy as np
from numpy.random import RandomState
from numpy.linalg import norm
import pandas as pd
from scipy.stats import norm as Gaussian
from env import ArrivalEvent, BacklogEvent, CallbackEvent, DispatchEvent, QueryEvent, BoundsMapper
from misc.utils import array_random_choice, optional_random
from typing import NamedTuple, Union
import copy

State = NamedTuple("State",
                   [('chargers', np.ndarray), ('incoming', np.ndarray), ('task', ArrivalEvent),
                    ('t', int),
                    ('locations', pd.DataFrame), ('date', str)])


class ChargerSimulation:
    def __init__(self,
                 initial_state,  # np.ndarray[int]: status of each charger
                 changes,  # np.ndarray[str]: list of lists of naturally occurring events
                 arrival_events,
                 # List[List[arrival_event]]: list of lists of naturally occurring queries at each timestep
                 region,  # str
                 helper,  # SimulationHelper:
                 n_samples,  #: float or int
                 replace_ratio,
                 max_distance,
                 start_date,
                 free_rewards=0):  #: int
        self.initial_state = initial_state  # list of statuses (1, 2, 3) for each charger
        self.og_changes = changes  # for each timestep, lists '' 'finish' or 'use' for each t -> t+1 timestep
        self.helper = helper  # helper with useful functions
        self.n_actions = len(helper.locations)  # number of stations
        self.n_chargers = len(self.initial_state)  # number of chargers
        self.free_rewards = free_rewards  # free_rewards to be received whenever you successfully recommend a car
        self.start_date = start_date  # start date is useful for mappers to be able to distinguish between multiple simulations, etc
        self.random: RandomState = optional_random()

        self.arrival_events = arrival_events  # List[List[ArrivalEvent]], ArrivalEvent has chargeridx=int and duration=int at each timestep, redundant with og_changes
        self.bounds = BoundsMapper(region, coordsys='gjc02')
        self.n_samples = n_samples  # number of samples to replace => either float % or int strict number

        assert replace_ratio > 0
        self.replace_ratio = replace_ratio  # for each sample to replace, how many simulated events to make

        self.max_distance = max_distance  # max_distance to recommend cars

        self.num_t_steps = len(arrival_events)  # number of timesteps that the simulation lasts
        # np.ndarray[float] : [(2*self.max_distance)+1, (2*self.max_distance)+1]
        self.reset()

        print("n_stations: %d" % self.n_actions)
        print("n_tasks: %d" % np.sum(self.resample_counts))

    def reset(self, random_seed=None, view=lambda x: None):
        self.current_state = self.initial_state.copy()  # same form as self.initial_state, except updated for each timestep
        self.t = 0  # self. current_timestep
        self.incoming_cars = np.zeros(
            (self.n_actions, self.num_t_steps))  # number of of arrivals on the way at each timestep
        # of note, this does double-count cars that come in 2x+ whenever they get redirected
        self.changes = self.og_changes.copy()  # not sure if this serves any purpose, this doesn't seem to get updated
        self.successful = []  # successful dispatches this timestep, contains list of ArrivalEvents
        self.denied = []  # dispatches that got bounced this timestep, contains list of DispatchEvents?
        self.backlogs = [[] for _ in range(len(
            self.arrival_events))]  # self.backlogs list of tasks from the last timestep that need to be redispatched List[DispatchEvent]?
        self.dispatches = [[] for _ in range(
            len(self.arrival_events))]  # for each timestep, list of DispatchEvents that were issued at that point
        self._allocate_samples(self.n_samples)  # allocate when our samples are from
        self._make_query_arrival_split()
        self.unaddressed = []  # list of allocations that will never be fulfilled within the episode, so we won't get to see the effects
        self.worst_reward = 0  # the single worst reward that we have received this simulation
        self.queued_actions = []  # actions for the coming timestep, becaues the agent only puts out a single action at a time

        while len(self.get_tasks()) == 0:
            obs, reward, done, info = self.step(None)
            assert reward == 0
            assert done == False

        print("n_stations: %d" % self.n_actions)
        print("n_tasks: %d" % np.sum(self.resample_counts))

        return self.state()

    def seed(self, random: Union[int, RandomState] = None) -> None:
        self.random = optional_random(random)

    def _allocate_samples(self, n_samples):
        counts = list(map(len, self.arrival_events))
        probabilities = np.asarray(counts, dtype=float) / sum(counts)

        if n_samples < 1:
            n_samples = int(np.ceil(n_samples * sum(counts)))

        choices = self.random.choice(len(probabilities), int(n_samples), p=probabilities)
        print(choices)
        self.resample_counts = [0 for _ in range(len(probabilities))]

        for choice in choices:
            self.resample_counts[choice] += 1

        print(np.nonzero(self.resample_counts))

    def _get_probabilities(self):
        def dist_from_center(val: int) -> int:
            return abs(val - (self.max_distance + 1))

        shape: int = (self.max_distance*2) + 1
        probabilities: np.ndarray = np.zeros_like((shape, shape), dtype=float)
        for x in range(shape):
            for y in range(shape):
                dx = dist_from_center(x)
                dy = dist_from_center(y)
                z = norm([dx, dy])
                probabilities[x, y] = Gaussian.pdf(z, loc=0.0, scale=self.max_distance/2)
        probabilities /= probabilities.sum()
        return probabilities

    def _make_query_arrival_split(self):
        to_be_processed = copy.deepcopy(self.arrival_events[self.t])
        num_replace = min(self.resample_counts[self.t], len(to_be_processed))
        self.random.shuffle(to_be_processed)
        to_be_queried = to_be_processed[:num_replace]
        normal_arrivals = to_be_processed[num_replace:]

        self.tasks = self._make_tasks(to_be_queried)
        self.current_arrivals = normal_arrivals

    def _make_tasks(self, queries):
        result = [None for _ in range(len(queries) * self.replace_ratio)]
        for i, query in enumerate(queries):
            for j in range(self.replace_ratio):
                station_name = self.helper.get_station_name(query.idx)
                x, y = self.helper.locations.loc[station_name, ['x', 'y']]

                xidx, yidx = self.bounds.get_index(x, y)

                dx, dy = self.random.normal(0, self.max_distance, (2,))
                #probabilities = map(lambda tup: self._get_current_traffic_data()[tup[1], tup[0]] + 1,
                #                   adjacent_tiles)
                # add 1 to normalize a little bit - make sure nothing is impossible
                """
                try:
                    new_x, new_y = array_random_choice(adjacent_tiles, probas=probabilities, random=self.random)
                    choice = self.random.choice(len(adjacent_tiles), 1, probabilities)[0]
                except Exception as e:
                    print(query.idx)
                    print(xidx, yidx)
                    print(x, y)
                    print(adjacent_tiles)
                    raise e
                new_x, new_y = adjacent_tiles[choice]
                """
                new_x, new_y = xidx+dx, yidx+dy

                if new_x > 3e3 or new_y > 3e3:
                    print(new_x, new_y)

                result[(i * self.replace_ratio) + j] = QueryEvent(new_x, new_y, query.duration)

        return result

    def _get_adjacent_tiles(self, xidx, yidx):
        ymax: int = self.bounds.y_shape
        xmax: int = self.bounds.x_shape
        results = []
        for x in range(xidx - self.max_distance, xidx + self.max_distance + 1):
            if x >= 0 and x < xmax:
                for y in range(yidx - self.max_distance, yidx + self.max_distance + 1):
                    if y >= 0 and y < ymax:
                        results.append((x, y))

        return results

    def is_done(self):
        return self.t == self.num_t_steps

    def get_tasks(self):
        # return List[Tuple[float, float]]
        return self.tasks + self.backlogs[self.t - 1]

    def _make_dispatch(self, query_event, idx):
        station_name = self.helper.get_station_name_from_station_idx(idx)
        truex, truey = self.helper.locations.loc[station_name, ['x', 'y']]
        txidx, tyidx = self.bounds.get_index(truex, truey)

        distance = 1 + abs(query_event.x - txidx) + abs(query_event.y - tyidx)
        # print("distance: %d" % distance)
        try:
            assert isinstance(distance, int)
        except Exception as e:
            print(distance)
            print(query_event)
            print(idx)
            print(truex, truey)
            print(txidx, tyidx)

        # add 1 because it will no matter take them some time to get to the next place
        # also ensures that we can add it to the next timestep
        already_traveled = query_event.drive_dist if isinstance(query_event, DispatchEvent) else 0
        already_waited = query_event.wait_time if isinstance(query_event, DispatchEvent) else 0

        if isinstance(query_event, DispatchEvent):
            # print("re-dispatching: already_traveled %d, already_waited %d, at %idx for time %d" %
            #       (already_traveled, already_waited, idx, self.t))
            pass

        if query_event.x > 500 or query_event.y > 500:
            print(query_event)

        new_dispatch = DispatchEvent(
            query_event.x,
            query_event.y,
            wait_time=already_waited,
            duration=query_event.duration,
            drive_dist=already_traveled + distance,
            station_name=station_name)

        try:
            if distance + self.t < self.num_t_steps:
                self.dispatches[distance + self.t].append(new_dispatch)
                self.incoming_cars[idx, distance + self.t] += 1
            else:
                self.unaddressed.append(new_dispatch)
        except:
            print(new_dispatch)
            print(query_event.x)
            print(distance)
            print(self.t)

    def state(self):
        # return np.ndarray
        return State(self.current_state.copy(),
                     self.incoming_cars[:, self.t:].copy(), self.current_task(),
                     self.t, self.helper.locations, self.start_date)

    def _current_dispatches(self):
        return self.dispatches[self.t]

    def current_task(self):
        tasks = self.get_tasks()
        if len(tasks) > 0:
            return self.get_tasks()[len(self.queued_actions)]
        else:
            return None

    def _add_to_queue(self, action):
        self.queued_actions.append(action)

    def step(self, action, view=lambda x: None):
        """
            action : id of the station to assign the most recent query to
            multi_step : skip timesteps with no queries

            action can be null iff there are no possible actions (only relevant if multi_step is False)
            if there are still more actions remaining this timestep, returns the same state, 0 reward, and the next task
        """
        if action is not None:
            self._add_to_queue(action)
        else:
            assert len(self.get_tasks()) == 0
        reward = 0
        while len(self.queued_actions) == len(self.get_tasks()) and not self.is_done():
            self.successful = []
            self.denied = []

            assert len(self.queued_actions) == len(self.get_tasks())
            self._implement_changes()

            self._implement_arrivals(self.current_arrivals)

            for query_event, dispatch in zip(self.get_tasks(), self.queued_actions):
                self._make_dispatch(query_event, dispatch)

            self._implement_arrivals(self._current_dispatches())
            self._handle_denied()
            self.t += 1

            reward += self.reward()
            self.queued_actions = []
            if not self.is_done():
                self._make_query_arrival_split()

                for success in self.successful:
                    if isinstance(success[1], DispatchEvent):
                        i_reward = self._get_individual_reward(success[1])
                        if i_reward < self.worst_reward:
                            self.worst_reward = i_reward

            view(self.state())

        return self.state(), reward, self.is_done(), action

    def _handle_denied(self):
        for denial in self.denied:
            if isinstance(denial, DispatchEvent):
                denial.wait_time += 1
            if isinstance(denial, ArrivalEvent):
                station = self.helper.get_station_name(denial.idx)
                x, y = self.helper.locations.loc[station, ['x', 'y']]
                xidx, yidx = self.bounds.get_index(x, y)
                denial = QueryEvent(xidx, yidx, denial.duration)
            if self.t < self.num_t_steps - 1:
                self.backlogs[self.t].append(denial)
            else:
                self.unaddressed.append(denial)

    def _implement_changes(self):
        new_state = self.current_state.copy()
        for changes in [self.changes[:, self.t * 2], self.changes[:, (self.t * 2) + 1]]:
            # cars should only come in via events [but can currently come in at the end of a timestep]
            new_state[(self.current_state == 2) & (changes == 'use')] = 3
            new_state[(self.current_state == 3) & (changes == 'finish')] = 2
            new_state[changes == 'break'] = 1
            new_state[(self.current_state == 1) & (changes == 'fix')] = 2

        self.current_state = new_state

    def _implement_arrivals(self, events):
        def try_append(event, index):
            if self.current_state[index] == 2:
                self.current_state[index] = 3
                t = (self.t + event.duration)
                if t < self.num_t_steps:
                    if self.changes[index, (t * 2)] == '':
                        self.changes[index, (t * 2)] = 'finish'
                    else:
                        self.changes[index, (t * 2) + 1] == \
                        self.changes[index, (t * 2)]
                        self.changes[index, t * 2] = 'finish'
                if isinstance(event, DispatchEvent):
                    self.successful.append((index, event))
                return True
            else:
                return False

        for event in events:
            if hasattr(event, "idx"):
                idx = event.idx
            else:
                idx = self.helper._station_to_chargers[event.station_name][0]

            if isinstance(event, DispatchEvent):
                station_idx = self.helper.locations.index.get_loc(event.station_name)

            if not try_append(event, idx):
                siblings = self.helper.get_siblings(idx)
                for idx in siblings:
                    if try_append(event, idx):
                        break
                else:
                    self.denied.append(event)

    def _get_individual_reward(self, event):
        return self.free_rewards - ((event.wait_time + event.drive_dist) ** 1)

    def reward(self):
        # return float
        num_in_transit = np.sum(self.incoming_cars[:, self.t:])
        num_successful = len(self.successful)

        return (num_successful * self.free_rewards) - num_in_transit

        """
        if not self.is_done():
            return sum(map(
                self._get_individual_reward,
                filter(
                    lambda event: isinstance(event, DispatchEvent),
                        map(lambda tup: tup[1], self.successful))))
        else:
            return sum(
                map(
                lambda e: self._get_individual_reward(e) + self.worst_reward,
                filter(
                    lambda event: isinstance(event, DispatchEvent),
                        self.unaddressed)))
        """


"""
class ChargerSimulation(gym.Env):
    def __init__(self,
            bounds: Bounds):
        self.reward_range: Tuple[float, float] = ...
        self.action_space: Space = ...
        self.observation_space: Space = ...
        self.random: RandomState = ...
        pass

    def step(self, action: Action) -> Tuple[State, Reward, bool, Any]:
        pass

    def reset(self) -> None:
        pass

    def render(self,
            mode: str = "human") -> np.ndarray:
        pass

    def close(self):
        pass

    def seed(self, seed: Union[int, RandomState]=None) -> None:
        pass

    def __str__(self) -> str:
        pass
"""