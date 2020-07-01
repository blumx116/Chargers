# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
File: simulation_helper.py
Author: bil(bil@baidu.com)
Date: 2019/07/29 20:22:17
"""
from typing import List, Tuple
import pandas as pd
import numpy as np
import copy
from misc.utils import listify, root_dir, optional_random
from typing import NamedTuple
from env.load_charger_data import load_charger_data
from env import ChargerSimulation, BoundsMapper
from env.simulation_events import Arrivals, ArrivalEvent
import os
import pickle
from wandb.util import PreInitObject as Config
from env import ContinuousSimulation


class SimulationHelper:
    def __init__(self, locations, names):
        """
            locations : pd.DataFrame, index=station name, has x and y in bd09mc // 1000
            names : listlike, names of stations ### LIKE 90% SURE I SHOULD JUST DELETE THIS ARGUMENT ###
        """

        self.locations = locations  # dataframe station_name -> x, y
        self.name_lookup = list(names)  # idx -> charger_name
        self.idx_lookup = dict(zip(names, range(len(names))))  # charger_name -> idx
        self.names = names  # not sure why I keep this

        self._station_to_chargers = dict([(station, listify(self.get_idx(chargers)))
                                          for (station, chargers) in
                                          pd.Series(self.names).groupby(lambda s: self.names[s].split(":")[0])])
        # station name to list of ids of chargers

        self._charger_idx_to_station = list(map(lambda s: s.split(":")[0], self.name_lookup))
        # charger idx to station name

    def get_nearest_station(self, lat, lng):
        return (((self.locations['lat'] - lat) ** 2) + ((self.locations['lng'] - lng) ** 2)).idxmin()

    def get_station_name(self, charger_idx):
        return self._charger_idx_to_station[charger_idx]

    def get_station_name_from_station_idx(self, station_idx):
        return self.locations.index[station_idx]

    def get_siblings(self, charger_idx):
        station = self._charger_idx_to_station[charger_idx]
        chargers = copy.copy(self._station_to_chargers[station])
        chargers.remove(charger_idx)
        return chargers

    def get_name(self, *idxs):
        if len(idxs) == 1:
            if hasattr(idxs[0], '__iter__'):
                idxs = list(idxs[0])

        result = [self.name_lookup[idx] for idx in idxs]
        if len(result) == 1:
            result = result[0]
        return result

    def get_idx(self, *names):
        if len(names) == 1:
            if hasattr(names[0], '__iter__'):
                names = list(names[0])

        result = [self.idx_lookup[name] for name in names]
        if len(result) == 1:
            result = result[0]
        return result


def _get_changes(charger_timeseries, handle_breaking='full'):
    """
        charger_timeseries: pd.DataFrame, [charger name => status [int]]
        handle_breaking : what to do with broken chargers => make them as 'full'
            or as 'empty'
        returns : np.ndarray[str], with ['', 'use', 'finish', 'break', 'fix'] denoting the change
            at each timestep. Note that each timestep can have up to two changes each fix => use
            so the data is of the form [n_chargers, (n_timesteps-1) * 2]
    """
    og_data = charger_timeseries.values

    if handle_breaking == 'full':
        og_data[og_data == 1] = 3
    elif handle_breaking == 'empty':
        og_data[og_data == 1] = 2

    NUM_T_STEPS = og_data.shape[1]
    changes = np.full((og_data.shape[0], (NUM_T_STEPS - 1) * 2), fill_value='', dtype='S18')

    for t in range(0, (NUM_T_STEPS - 1) * 2, 2):
        old = og_data[:, int(t / 2)]
        new = og_data[:, int(t / 2) + 1]
        # 2 represents free, #3 represents in use
        changes[(old == 2) & (new == 3), t] = 'use'.encode('utf-8')
        changes[(old == 3) & (new == 2), t] = 'finish'.encode('utf-8')
        # print(changes[6,:])

        # if it switches from 3 -> 1, that means it actually did 3 -> 2 -> 1
        changes[(old == 3) & (new == 1), t] = 'finish'.encode('utf-8')
        changes[(old == 3) & (new == 1), t + 1] = 'break'.encode('utf-8')
        changes[(old == 2) & (new == 1), t] = 'break'.encode('utf-8')
        # print(changes[6, :])

        # similarly, if it switches from 1 -> 3, that mean sit actually did 1 -> 2 -> 3
        changes[(old == 1) & (new != 1), t] = 'fix'.encode('utf-8')
        changes[(old == 1) & (new == 3), t + 1] = 'use'.encode('utf-8')

    return changes


def _remove_use_finish(changes, events):
    changes = changes.copy()
    for timestep, contemporaneous_events in enumerate(events):
        for event in contemporaneous_events:
            end_t = (event.duration + timestep) * 2
            for t in (timestep * 2, (timestep * 2) + 1, end_t, end_t + 1):

                if (changes[event.idx, t] == b'use') | (changes[event.idx, t] == b'finish'):
                    changes[event.idx, t] = b''

    return changes


def _get_arrival_events(changes):
    NUM_T_STEPS = int(changes.shape[1] / 2)
    events = [[] for i in range(NUM_T_STEPS)]
    for i in range(len(changes)):
        starts = np.where(changes[i, :] == b'use')[0]
        ends = np.where(changes[i, :] == b'finish')[0]

        assert abs(len(starts) - len(ends)) < 2
        if len(starts) == 0:
            continue

        if len(ends) == 0:
            continue

        if ends[0] < starts[0]:  # starts the day occupied
            ends = ends[1:]
            if len(ends) == 0:
                continue
        if starts[-1] > ends[-1]:
            starts = starts[:-1]
            if len(starts) == 0:
                continue
        assert len(starts) == len(ends)

        for start, end in zip(starts, ends):
            start = start // 2
            end = end // 2
            events[start].append(ArrivalEvent(idx=i, duration=end - start))

    return events

def _get_departure_events(changes: np.ndarray) -> List[List[int]]:
    """
    NOTE: right now, this has a LOT of code overlap with _get_arrival_events
    Returns all of the times where a car leaves a station but we never see it arrive
    :param changes: np.ndarray[bstr] : [n_chargers, n_tsteps * 2]
        valid values are '', b'finish', b'use', b'fix' and b'break'
        we will ignore all fixes and breaks
    :return: List of lists for each timestep. At each timestep, list the
        chargers with a car leaving.
    """
    NUM_T_STEPS: int = int(changes.shape[1] / 2)
    events: List[List[int]] = [[] for _ in range(NUM_T_STEPS)]

    for i in range(len(changes)):
        starts: np.ndarray = np.where(changes[i, :] == b'use')[0]
        # np.ndarray[int64] : [num_starts, ]
        ends: np.ndarray = np.where(changes[i, :] == b'finish')[0]
        # np.ndarray[int64] : [num_ends, ]

        assert abs(len(starts) - len(ends)) < 2
        # sanity check -> can't have too many more starts than ends
        if len(starts) == 0 or len(ends) == 0:
            continue

        if ends[0] < starts[0]:
            # start the day occupied
            tstep: int = int(ends[0] / 2)
            events[tstep].append(i)
            continue

    return events

def _load_changes(daterange, region, handle_missing='replace', handle_breaking='full',
                  limiting_chargers=None, force_reload=False, group_stations=True):
    """
        Loads changes (when cars enter and leave) for a given time range and location
        also returns slightly modified charger_timeseries, charger_locations, used in
        make_simulation, but  haven't looked at the changes in too much detail
        daterange: pd.date_range, dates to base data off of
        region : str, in chargers.misc.region_bounding
        handle_breaking : protocol for handling broken chargers in data,
        handle_missing : how to handle missing timesteps, 'replace' simply
            uses data from nearby timesteps
        limiting_chargers : None | pd.DataFrame with chargers to limit to
            only loads charges with the given name
        force_reload : bool, whether or not to ignore cache
        group_stations : bool, whether or not to group all stations in the same grid tile
        returns :
    """

    charger_timeseries, charger_locations = load_charger_data(daterange, region,
                                                              handle_missing=handle_missing,
                                                              limiting_chargers=limiting_chargers,
                                                              force_reload=force_reload, group_stations=group_stations)

    # subset the data so we only take the part within bounds
    # this part is essentially rewriting code, should probably be refactored in the future
    # heck, this should probably be in load_charger_data to save runtime
    locs = charger_locations[['x', 'y']].values
    bounds = BoundsMapper(region, 'gjc02')
    inrange = np.asarray([bounds.is_within_bounds(*locs[i, :]) for i in range(locs.shape[0])])
    charger_station_mapping = charger_timeseries.index.str.split(":") \
        .map(lambda l: l[0]).map(lambda s: charger_locations.index.get_loc(s))

    charger_timeseries = charger_timeseries.iloc[inrange[charger_station_mapping], :]
    charger_locations = charger_locations.iloc[inrange, :]

    return _get_changes(charger_timeseries=charger_timeseries, handle_breaking=handle_breaking), \
           charger_timeseries, charger_locations


def make_simulation(daterange, region, n_samples, replace_ratio, sample_range,
                    handle_missing='replace', handle_breaking='full', limiting_chargers=None, force_reload=False,
                    group_stations=True):
    changes, charger_timeseries, charger_locations = _load_changes(daterange, region, handle_missing, handle_breaking,
                                                                   limiting_chargers, force_reload=False,
                                                                   group_stations=True)
    # don't force reload because we just called load_charger_data, load_station_data

    arrival_events = _get_arrival_events(changes)
    updated_changes = _remove_use_finish(changes, arrival_events)
    initial_state = charger_timeseries.values[:, 0]
    helper = SimulationHelper(charger_locations, names=charger_timeseries.index)
    return ChargerSimulation(initial_state, updated_changes, arrival_events, 'haidian', helper, n_samples,
                             replace_ratio, sample_range, start_date=str(daterange[0]))


def load_day_simulation(date, region, n_samples, replace_ratio, sample_range, limiting_chargers=None,
                        force_reload=False, group_stations=True):
    """
        loads a catched version of the simulation for the day
        includes all data from the specified day and region
        limiting_chargers is only used to subset the chargers in the model iff there is no catched simulation found
        if no cached simulation is found, a new one is loaded and cached in "simulation_saves/YYYY-MM-DD.pickle"
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    if isinstance(date, pd.Timestamp):
        date = pd.Timestamp(year=date.year, month=date.month, day=date.day)

    directory = os.path.join(root_dir, 'data', 'simulation_saves')
    fname = _datetime_to_sim_name(date) + region

    full_fname = os.path.join(directory, fname + ".pickle")

    date_range = pd.date_range(date, date + pd.Timedelta(days=1), freq='h')
    print(full_fname)
    if not os.path.isfile(full_fname) or force_reload:
        print("reloading")

        sim = make_simulation(date_range, region, n_samples, replace_ratio, sample_range,
                              limiting_chargers=limiting_chargers, force_reload=force_reload,
                              group_stations=group_stations)

        with open(full_fname, "wb") as f:
            rand = sim.random
            sim.random = None
            # can't pickle random when random is np.random
            pickle.dump(sim, f)
            sim.random = optional_random()

    with open(full_fname, "rb") as f:
        sim = pickle.load(f)
        sim.replace_ratio = replace_ratio
        sim.n_samples = n_samples
        sim.sample_range = sample_range
        sim.start_date = str(date_range[0])
        sim.random = optional_random()
        sim.reset()

        return sim


def _datetime_to_sim_name(date):
    return "%04d-%02d-%02d" % (date.year, date.month, date.day)

def load_continuous_simulation(
        config: Config,
        handle_missing: str = 'replace',
        handle_breaking: str = 'full',
        force_reload: bool = False) -> ContinuousSimulation:
    for attr in ['car_speed', 'max_cars', 'sample_amount', 'sample_distance']:
        assert hasattr(config, attr)
    for attr in ['date', 'region']:
        assert hasattr(config, attr)
    start_date: pd.datetime = pd.to_datetime(config.date)

    daterange: pd.DatetimeIndex = pd.date_range(
        start=start_date,
        end=start_date + pd.Timedelta(days=1),
        freq='h')

    changes, charger_timeseries, charger_locations = \
        _load_changes(daterange, config.region,
                      handle_missing, handle_breaking,
                      limiting_chargers=None, force_reload=force_reload,
                      group_stations=False)
    # don't force reload because we just called load_charger_data, load_station_data


    mapping: List[int] = charger_station_mapping(charger_timeseries, charger_locations)
    # len = n_chargers
    arrivals: Arrivals = _get_arrival_events(changes)
    arrivals = arrivals_charger2station(arrivals, mapping)
    departures: List[List[int]] = _get_departure_events(changes)
    departures = departures_charger2station(departures, mapping)
    # both: len = n_tsteps
    initial_state: np.ndarray = occupancy_chargers2station(
        charger_timeseries.values[:, 0], mapping)
    max_occ: np.ndarray = max_occupancy(mapping)
    max_occ = max_occ[:, np.newaxis]
    # np.ndarray[int32] : [n_stations, 1]
    assert len(max_occ) == len(initial_state)
    locations: np.ndarray = charger_locations[['lng', 'lat']].values
    # np.ndarray[float64] : [n_stations, 2] => (x, y)
    station_info: np.ndarray = np.concatenate((locations, max_occ), axis=1)
    station_info: np.ndarray = station_info.astype(np.float32)
    # np.ndarray[float32] : [n_stations, 3] => (x, y, max_occ)

    return ContinuousSimulation(
        arrivals,
        departures,
        initial_state,
        station_info,
        config)

def charger_station_mapping(
        charger_timeseries: pd.DataFrame,
        charger_locations: pd.DataFrame) -> List[int]:
    return [charger_locations.index.get_loc(charger.split(":")[0]) for
            charger in charger_timeseries.index]

def arrivals_charger2station(
        arrivals: Arrivals,
        mapping: List[int]) -> Arrivals:
    return [
        [ArrivalEvent(mapping[event.idx], event.duration) for event in tstep_events]
        for tstep_events in arrivals]

def departures_charger2station(
        departures: List[List[int]],
        mapping: List[int]) -> List[List[int]]:
    return [
        [mapping[idx] for idx in tstep_departures]
        for tstep_departures in departures]

def occupancy_chargers2station(
        initial_occupancy: np.ndarray,
        mapping: List[int]) -> np.ndarray:
    """

    :param initial_occupancy: np.ndarray[float64] : [n_chargers, ]
        3 indicates full, 2 indicates empty, no other values accepted
    :param mapping: List[int]
        the value of the ith index is the index of the station that the
        ith charger belongs to
    :return: np.ndarray[int32] : [n_stations,]
        number of chargers in use at the station
    """
    n_stations: int = max(mapping) + 1
    result: np.ndarray = np.zeros((n_stations, ), dtype=np.int32)
    # np.ndarray[int32] : [n_stations,]
    in_use: np.ndarray = (initial_occupancy == 3.)
    # np.ndarray[bool] : [n_chargers, ]
    for i in range(len(in_use)):
        if in_use[i]:
            result[mapping[i]] += 1
    return result

def max_occupancy(mapping: List[int]) -> np.ndarray:
    """

    :param mapping: List[int]
        the value of the ith index is the index of the station that the
        ith charger belongs to
    :return: np.ndarray[int32] : [n_stations,]
        number of chargers in use at the station
    """
    return np.unique(mapping, return_counts=True)[1].astype(np.int32)
