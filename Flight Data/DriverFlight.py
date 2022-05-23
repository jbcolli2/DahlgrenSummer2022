#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:46:32 2022

@author: jebcollins
"""

from os import listdir
import csv

import scipy.io as sio

import pandas as pd

import matplotlib.pyplot as plt

import gzip


flights = pd.read_csv('WeeklyFlights.csv/WeeklyFlight.csv.gz', compression='gzip')

#%%

start = flights.loc[flights.time == flights.time[0]]
start = start[['lat','lon']]


#%%

plt.plot(start['lat'], start['lon'],'o')



# data_files = listdir("./FlightData")
# data_files = ["./FlightData/" + s for s in data_files]

# flight = sio.loadmat(data_files[5])

# flight = pd.DataFrame(data=flight['data_to_output'])

# def decomment(csvfile):
#     for row in csvfile:
#         raw = row.split('%')[0].strip()
#         if raw: yield raw
        

# colnames = []        
# with open('./FlightData/header.txt') as headerfile:
#     reader = csv.reader(decomment(headerfile))
#     for name in reader:
#         colnames.append(name)
        
# colnames = [name[0] for name in colnames]

# flight.columns = colnames




# from tslearn.utils import to_time_series
# my_ts = flight.iloc[:,1]
# formatted_ts = to_time_series(my_ts)

