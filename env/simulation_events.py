# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
File: simulation_events.py
Author: bil(bil@baidu.com)
Date: 2019/07/31 13:23:02
"""

from typing import NamedTuple, List
QueryEvent = NamedTuple("QueryEvent", [("x", float), ("y", float), ("duration", int), ('og_distance', float)])
ArrivalEvent = NamedTuple("ArrivalEvent", [("idx", int), ("duration", int)])
BacklogEvent = NamedTuple('BacklogEvent', [("idx", int), ('duration', int)])
CallbackEvent = NamedTuple('CallbackEvent', [("idx", int), ('duration', int)])

Arrivals = List[List[ArrivalEvent]]
Queries  = List[List[QueryEvent]]


class DispatchEvent:
    def __init__(self, x, y, wait_time, duration, drive_dist, station_name):
        self.x = x
        self.y = y
        self.wait_time = wait_time
        self.duration = duration
        self.drive_dist = drive_dist
        self.station_name = station_name

    def __repr__(self):
        return "x=%d, y=%d, wait_time=%d, duration=%d, drive_dist=%d, station_name=%s" % \
               (self.x, self.y, self.wait_time, self.duration, self.drive_dist, self.station_name)
