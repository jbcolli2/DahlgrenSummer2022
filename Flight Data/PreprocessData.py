from os import listdir
import csv

import pandas as pd

import DataUtil

dataDir = './data/Blake Files/baserollout_and_10MCs/'

data =DataUtil.MCTSData(dataDir)


for mctsdata in data.mcts_data[1:]:
    mctsdata.paths.to_csv(dataDir+'Path'+mctsdata.filename.rsplit('/')[-1], index=True)

data.rollout_data.paths.to_csv(dataDir+'Path'+data.rollout_data.filename.rsplit('/')[-1], index=True)