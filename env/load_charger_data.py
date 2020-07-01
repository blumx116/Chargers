# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: load_occupancy_data.py
Author: bil(bil@baidu.com)
Date: 2019/07/23 13:03:51
"""

# NOTE: All location data in this project will be using GCJ coordinate system

from collections import namedtuple
import json
import os

import numpy as np
import pandas as pd

from misc.utils import df_flatmap, root_dir
from env.region_bounds import city_of, BoundsMapper

import dill
import pickle


CACHE_DIR = os.path.join(root_dir, 'data', 'detailed', 'cache')
DATA_DIR = os.path.join(root_dir, 'data', 'detailed')

LABEL_FILE_TYPE = ".lbl"
CACHE_FILE_TYPE = ".pkl"

ChargerData = namedtuple("ChargerData", ("charger_timeseries", "locations"))

def _charger_file_name_to_time(fname):
    fname = os.path.split(fname)[-1]
    fname = fname.replace(".json", "")
    parsed = list(map(int, fname.split("-")))
    return pd.Timestamp(year=parsed[0], month=parsed[1], day=parsed[2],
            hour=parsed[3], minute=parsed[4])

def _time_to_charger_file_name(time):
    return _hourly_to_charger_file_format(time) + ("-%02d.json" % (time.minute))

def _hourly_to_charger_file_format(time):
    return "%04d-%02d-%02d-%02d" % (time.year, time.month, time.day, time.hour)

def time_to_time_range(time):
    """
        time should either be a string timestamp, a timestamp or a pd.DatetimeIndex
    """
    if isinstance(time, str):
        time = pd.Timestamp(time)
    if isinstance(time, pd.Timestamp):
        start_time = time
        end_time = start_time + pd.Timedelta(minutes=59)
    elif isinstance(time, pd.DatetimeIndex):
        start_time = time[0]
        end_time = time[-1]
    else:
        raise Exception

    return pd.date_range(start_time, end_time, freq='H')


def get_charger_file_names(time, region):
    """
        time: pd.DateTime or pd.time_range
        region: region
    """
    
    datapath = os.path.join(DATA_DIR,city_of[region])
    
    time_range = time_to_time_range(time)

    f_names = []
    
    for time_slot in time_range[:-1]:
        format = _hourly_to_charger_file_format(time_slot)
        f_names += list(filter(lambda s: format in s, os.listdir(datapath)))
    f_names = sorted(np.unique(f_names), key=_charger_file_name_to_time)
    return list(map(lambda s: os.path.join(datapath, s), f_names))

def raw_charger_data(time, region, handle_missing='erase'):
    file_names = get_charger_file_names(time, region)
    return raw_charger_data_from_files(file_names,region, handle_missing)

def raw_charger_data_from_files(file_names,region,  handle_missing='erase'):
    data = list(map(lambda fname: list(map(json.loads, open(fname, 'r'))),
                file_names))

    named_data = list(zip(file_names, data))

    if city_of[region] != region:
        mapper = BoundsMapper('haidian', coordsys='gjc02')
        names, values = zip(*named_data)
        values = map(lambda tstep: list(filter(lambda entry: mapper.is_within_bounds(entry['lng'], entry['lat']), tstep)), values)
        named_data = list(zip(names, values))

    if handle_missing == 'erase':
        new_data = list(filter(lambda tup: tup[1] != [], named_data))
    elif handle_missing == 'replace':
        named_data = list(named_data)

        first_non_null = np.where(list(map(lambda tup: tup[1] != [], named_data)))[0][0]
        for i in range(first_non_null):
            named_data[i] = (named_data[i][0], named_data[first_non_null][1])

        #for all other missing values, we assume that they are equal to the timestep before
        new_data = []
        for i, (fname, data) in enumerate(named_data):
            if i > 0:
                data_path = "/".join(os.path.split(fname)[:-1])
                
                last_time_slot = _charger_file_name_to_time(new_data[-1][0])
                current_time_slot = _charger_file_name_to_time(fname)

                while (current_time_slot - last_time_slot) > pd.Timedelta(minutes=15):
                    print("warning : creating time slot after ", last_time_slot)
                    new_data.append((
                                os.path.join(data_path, _time_to_charger_file_name(last_time_slot + pd.Timedelta(minutes=15))),
                                new_data[-1][1]))
                    last_time_slot = _charger_file_name_to_time(new_data[-1][0])
            if data == []:
                new_data.append((fname, new_data[-1][1]))
            else:
                new_data.append((fname, data))
    else:
        assert handle_missing is None
        new_data = named_data


    return new_data

def _get_cache_name(time, region, handle_missing):
    time = time_to_time_range(time)
    
    def time2str(time):
        return "%04d%02d%02d%02d" % (time.year, time.month, time.day, time.hour)

    fname = time2str(time[0]) + "-" +  time2str(time[-1]) + region + handle_missing
    return fname

def _get_matching_cache_names(time, region, handle_missing):
    files = os.listdir(CACHE_DIR)
    files = filter(lambda fname: fname.endswith(LABEL_FILE_TYPE), files) #subset down to labels
    files = map(lambda fname: fname[:-len(LABEL_FILE_TYPE)], files) #remove file type
    pattern = _get_cache_name(time, region, handle_missing) 

    return list(filter(lambda fname: pattern in fname, files)) #select ones that match pattern

def _limiting_chargers_to_list(limiting_chargers):
    if limiting_chargers is None:
        return None
    if isinstance(limiting_chargers, pd.DataFrame):
        limiting_chargers = limiting_chargers.index
    if isinstance(limiting_chargers, pd.Series) or isinstance(limiting_chargers, pd.Index):
        limiting_chargers = limiting_chargers.tolist()
    else:
        limiting_chargers = list(limiting_chargers)

    return limiting_chargers

def _find_in_cache(time, region, handle_missing, limiting_chargers):
    limiting_chargers = _limiting_chargers_to_list(limiting_chargers)

    cache_names = _get_matching_cache_names(time, region, handle_missing)
    for name in cache_names:
        cached_chargers = json.load(open(os.path.join(CACHE_DIR, name+LABEL_FILE_TYPE), 'r'))
        if cached_chargers == limiting_chargers:
           return name
    return None

def _add_to_cache(data, time, region, handle_missing, limiting_chargers):
    limiting_chargers = _limiting_chargers_to_list(limiting_chargers)
    cache_match = _find_in_cache(time, region, handle_missing, limiting_chargers)
    if cache_match is not None:
        target_fname = cache_match
    else:
        target_fname = _get_cache_name(time, region, handle_missing)
        cache_names = _get_matching_cache_names(time, region, handle_missing)
        appendix = "(%d)" % len(cache_names) if len(cache_names) > 0 else ""

        target_fname = target_fname + appendix
    with open(os.path.join(CACHE_DIR, target_fname + LABEL_FILE_TYPE), 'w') as fp:
        json.dump(limiting_chargers, fp)

    with open(os.path.join(CACHE_DIR, target_fname + CACHE_FILE_TYPE), 'wb') as fp:
        pickle.dump(data, fp)

def _load_from_cache(time, region, handle_missing, limiting_chargers):
    matched_file = _find_in_cache(time, region, handle_missing, limiting_chargers)
    if matched_file is None:
        return None
    else:
        return pickle.load(open(os.path.join(CACHE_DIR, matched_file + CACHE_FILE_TYPE), 'rb'))

def _map_location_to_stations(locations, region):
    """
        Returns a dictionary with keys being grid tiles and a list of station names
        at that grid location
        locations : pd.DataFrame [station_id] => ['lat', 'lng', 'x', 'y']
        region : str in chargers.misc.region_bounds, the region the data is from, in
        return Dict[(x:int, y:int)] => List[station_id: string]
    """
    result = {}
    mapper = BoundsMapper(region)
    for i in range(len(locations)):
        x, y = locations.iloc[i][['x', 'y']]
        x, y = mapper.get_index(x, y) 
        if (x, y) not in result:
            result[(x, y)] = []
        result[(x, y)].append(locations.index[i])
    return result 

def _group_stations(charger_data, region):
    """
        Groups all stations in the same grid cell together
        more specifically, returns a new ChargerData with the indices of the dataframes modified
        such that all indices list themselves as the same station_id (note that station ids 
        will no longer be accurate). Does not modify in place 
        charger_data : ChargerData, to be modified
        region : str in chargers.misc.region_bounds, the region the data is from, in
        returns ChargerData, with changes applied to the indices of the dataframes
    """
    location_station_map = _map_location_to_stations(charger_data.locations, region)
    mapper = BoundsMapper(region)

    def _map_charger_names(charger_names, locations, 
            location_station_map, mapper):
        
        charger_stations = list(map(lambda name: name.split(":")[0], charger_names))
        charger_ids = list(map(lambda name: name.split(":")[1:], charger_names))
        # charger_stations : List[station_name : str]
        # charger_ids : List[[str, str]]
        # all charger names are of the form station_name:id_stuff:id_stuff

        charger_locations = locations.loc[charger_stations]
        # pd.DataFrame, like locations, with a bunch of duplicated rows
        # gives location of each charger
        charger_locations = [tuple(x) for x in charger_locations[['x', 'y']].values]
        # [Tuple[x:int, y:int]], coordinates in bd09mc
        charger_locations = list(map(lambda tup: mapper.get_index(*tup), charger_locations))
        # Generator?[Tuple[x:int, y:int]], coordinates in grid cells

        station_names = list(map(lambda tup: location_station_map[tup][0], charger_locations))
        # Generator?[station_name: str]
        # name of first station recorded of the grid cell that 
        # each station is in

        new_names = list(map(lambda tup: ':'.join([tup[0]] + tup[1]),
            zip(station_names, charger_ids)))
        # new_names : new_station_name:id_stuff:id_stuff

        return new_names 

    def _map_station_names(locations, location_station_map, mapper):
        station_locations = [tuple(x) for x in locations[['x', 'y']].values]
        # [Tuple[x:int, y:int]], coordinates in bd09mc

        station_locations = map(lambda tup: mapper.get_index(*tup), station_locations)
        # Generator?[Tuple[x:int, y:int]], coordinates in grid cells]

        station_names = map(lambda tup: location_station_map[tup][0], station_locations)

        return list(station_names)

    charger_names = _map_charger_names(charger_data.charger_timeseries.index, 
        charger_data.locations, location_station_map, mapper)
    station_names = _map_station_names(charger_data.locations,
        location_station_map, mapper)

    ts = charger_data.charger_timeseries.copy()
    locs = charger_data.locations.copy()
    ts.index = charger_names
    locs.index = station_names
    locs = locs[~locs.index.duplicated(keep='first')] # drop extra rows

    return ChargerData(ts, locs)


def load_charger_data(time, region, handle_missing='erase', limiting_chargers=None, 
        force_reload=False, group_stations=False):
    if not force_reload: 
        in_cache = _find_in_cache(time, region, handle_missing, limiting_chargers) is not None
        if in_cache:
            result = _load_from_cache(time, region, handle_missing, limiting_chargers)
            if group_stations:
                result = _group_stations(result, region)
            return result

    data = raw_charger_data(time, region, handle_missing)
    times, data = zip(*data)
    
    def simplify_states(df):
        df[(df['status'] != 2) & (df['status'] != 3)] = 1
        return df

    data = list(map(pd.DataFrame, data))
    data = list(map(lambda df: df.set_index('id').sort_index(), data))
    locations = data[0]

    NUM_T_STEPS = len(data)

    """
    nd = []
    for df in data:
        add = []
        for c in df.chargers:
            add.append({'id': c['id'], 'status': c['status']})
    """
    data = list(map(lambda df: df_flatmap(df, lambda r: [{'id': c['id'], 'status': c['status']} for c in r.chargers]),
        data))

    data = list(map(pd.DataFrame, data))
    data = list(map(lambda df: df.set_index('id').sort_index(), data))
    if limiting_chargers is not None:
        limiting_chargers = _limiting_chargers_to_list(limiting_chargers)
        data = list(map(lambda df: df.loc[limiting_chargers].sort_index(), data))
    data = list(map(simplify_states, data))

    og_data = pd.DataFrame(index=sorted(list(set.union(*map(lambda df: set(df.index), data)))))

    for i, timestep in enumerate(data):
        og_data[str(i)] = np.NaN
        og_data.loc[timestep.index, str(i)] = timestep.status

    
    og_data=og_data[~og_data.isna().any(axis=1)]
    
    locations = locations.loc[list(set(map(lambda s: s.split(':')[0], og_data.index.values)))][['lat', 'lng']]


    locations['x'], locations['y'] = np.NaN, np.NaN

    
    for i in range(len(locations)):
        x = locations.iloc[i]['lng']
        y = locations.iloc[i]['lat']
        locations.loc[locations.index[i], 'x'], locations.loc[locations.index[i], 'y'] = x, y
        
    locations = locations.sort_index()
    og_data = og_data.sort_index()

    result = ChargerData(og_data, locations)
    _add_to_cache(result, time, region, handle_missing, limiting_chargers)

    if group_stations:
        result = _group_stations(result, region)
    return result