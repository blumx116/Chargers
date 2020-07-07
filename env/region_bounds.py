# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: region_bounds.py
Author: bil(bil@baidu.com)
Date: 2019/07/23 18:59:30
"""

bd09mc_region_bounds = {'beijing': {'bottom': 4785949.940386938,
  'left': 12888857.088323496,
  'right': 13031563.012696654,
  'top': 4877982.061948756},
 'chongqing': {'bottom': 3337819.6580818193,
  'left': 11767407.021479651,
  'right': 12021871.009801775,
  'top': 3485275.7732492383},
 'guangzhou': {'bottom': 2586685.724181055,
  'left': 12539054.98396966,
  'right': 12619439.044986438,
  'top': 2715197.854024718},
 'haidian': {'bottom': 4824471.871013043,
  'left': 12932799.152296724,
  'right': 12955327.04204662,
  'top': 4849559.68012955},
 'hangzhou': {'bottom': 3459013.3399929916,
  'left': 13317514.922142906,
  'right': 13455242.967853846,
  'top': 3563613.804840059},
 'shanghai': {'bottom': 3572619.7091274527,
  'left': 13469255.00634756,
  'right': 13591111.033316549,
  'top': 3707275.9238245427},
 'shenzhen': {'bottom': 2556605.9231463433,
  'left': 12663790.972770771,
  'right': 12740591.10915029,
  'top': 2611614.0114863394},
 'tianjin': {'bottom': 4645853.897967383,
  'left': 13000587.086881202,
  'right': 13142630.97790707,
  'top': 4788189.980328515}}

 
wgs84_region_bounds = {'beijing': {'bottom': 39.63024402152495,
  'left': 115.7685970643055,
  'right': 117.0501716639315,
  'top': 40.26673960796385},
 'chongqing': {'bottom': 28.86351926,
  'left': 105.69724031,
  'right': 107.98236502,
  'top': 30.02262503},
 'guangzhou': {'bottom': 22.759428864617988,
  'left': 112.62749002963504,
  'right': 113.34904024836278,
  'top': 23.826011968043222},
 'haidian': {'bottom': 39.89753878336386,
  'left': 116.16344689230421,
  'right': 116.36570683485039,
  'top': 40.07071022242619},
 'hangzhou': {'bottom': 29.816618541532097,
  'left': 119.62073325371418,
  'right': 120.85862369531885,
  'top': 30.632786121612465},
 'shanghai': {'bottom': 30.702589377882536,
  'left': 120.98423594270382,
  'right': 122.07867389037662,
  'top': 31.741661141267553},
 'shenzhen': {'bottom': 22.508947593073565,
  'left': 113.74792181876991,
  'right': 114.43798060721728,
  'top': 22.966639154279378},
 'tianjin': {'bottom': 38.65112303938804,
  'left': 116.77254135917364,
  'right': 118.04817902023306,
  'top': 39.645855368272855}}

gjc02_region_bounds = {'beijing': {'bottom': 39.63162326388889,
  'left': 115.774833984375,
  'right': 117.05661241319444,
  'top': 40.26799207899305},
 'chongqing': {'bottom': 28.86016791449653,
  'left': 105.7008238389757,
  'right': 107.98667887369791,
  'top': 30.019999186197918},
 'guangzhou': {'bottom': 22.75663275824653,
  'left': 112.63248616536458,
  'right': 113.35456814236112,
  'top': 23.823507215711807},
 'haidian': {'bottom': 39.898920084635414,
  'left': 116.1696568467882,
  'right': 116.37192762586805,
  'top': 40.072039930555555},
 'hangzhou': {'bottom': 29.814188639322918,
  'left': 119.62552951388889,
  'right': 120.86274142795139,
  'top': 30.630508897569445},
 'shanghai': {'bottom': 30.700456271701388,
  'left': 120.9885630967882,
  'right': 122.0831271701389,
  'top': 31.7398046875},
 'shenzhen': {'bottom': 22.506253797743057,
  'left': 113.75310980902778,
  'right': 114.44288764105903,
  'top': 22.96408447265625},
 'tianjin': {'bottom': 38.65194878472222,
  'left': 116.77854112413195,
  'right': 118.05451958550347,
  'top': 39.64723849826389}}

assert set(gjc02_region_bounds.keys()) == set(wgs84_region_bounds.keys()) == set(bd09mc_region_bounds.keys())

city_of = {key:key for key in gjc02_region_bounds.keys()}
city_of['haidian'] = 'beijing'



class BoundsMapper:
    def __init__(self, region, coordsys='gjc02'):
        bounds_dict = {
            'bd09mc' : bd09mc_region_bounds,
            'wgs84'  : wgs84_region_bounds,
            'gjc02'  : gjc02_region_bounds
        }
        self.bounds = bounds_dict[coordsys][region]
        self.coordsys = coordsys

        self.x_shape = self.get_x_index(self.bounds['right']) + 1
        self.y_shape = self.get_y_index(self.bounds['top']) + 1
        
    @staticmethod    
    def _convert_to_block(val, convert='auto'):
        if convert == 'auto':
            convert = val > 1e5
        if convert:
            return int(val // 1000)
        else:
            return int(val)
        
    @staticmethod
    def _normalize(val, min, convert='auto'):
        return BoundsMapper._convert_to_block(val) - min
    
    def get_x_index(self, val, convert='auto'):
        # assert self.coordsys == 'bd09mc' #should implement the others later
        return BoundsMapper._normalize(val, 
                               BoundsMapper._convert_to_block(self.bounds['left'], True), convert)
     
    def get_y_index(self, val, convert='auto'):
        # assert self.coordsys == 'bd09mc' #should implement the others later
        return BoundsMapper._normalize(val, 
                               BoundsMapper._convert_to_block(self.bounds['bottom'], True), convert)
    
    def get_index(self, x, y):
        return self.get_x_index(x), self.get_y_index(y)
   

    def is_within_bounds(self, x, y):
        if self.coordsys == 'bd09mc':
            x, y = self.get_index(x,y)
            return x >=0 and x < self.x_shape and y >= 0 and y < self.y_shape
        else:
            return x >= self.bounds['left'] and x < self.bounds['right'] and y >= self.bounds['bottom'] and y < self.bounds['top']

    
