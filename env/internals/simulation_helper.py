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
from typing import List
import pandas as pd
import numpy as np
import copy
from misc.utils import listify, root_dir, optional_random
from env import BoundsMapper
from env.simulation_events import Arrivals, ArrivalEvent
from env.internals.charger_data import lcd
import pickle


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

    charger_timeseries, charger_locations = lcd(daterange, region,
                                                handle_missing=handle_missing,
                                                limiting_chargers=limiting_chargers,
                                                force_reload=force_reload, group_stations=group_stations)

    # subset the data so we only take the part within bounds
    # this part is essentially rewriting code, should probably be refactored in the future
    locs = charger_locations[['x', 'y']].values
    bounds = BoundsMapper(region, 'gjc02')
    inrange = np.asarray([bounds.is_within_bounds(*locs[i, :]) for i in range(locs.shape[0])])
    charger_station_mapping = charger_timeseries.index.str.split(":") \
        .map(lambda l: l[0]).map(lambda s: charger_locations.index.get_loc(s))

    charger_timeseries = charger_timeseries.iloc[inrange[charger_station_mapping], :]
    charger_locations = charger_locations.iloc[inrange, :]

    return _get_changes(charger_timeseries=charger_timeseries, handle_breaking=handle_breaking), \
           charger_timeseries, charger_locations

def _datetime_to_sim_name(date):
    return "%04d-%02d-%02d" % (date.year, date.month, date.day)

