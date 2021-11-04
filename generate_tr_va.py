#!/usr/bin/python3
import glob
import pandas as pd
import os
import numpy as np
import sys
#from sklearn.model_selection import KFold

path = sys.argv[1]

if not os.path.isdir(path):
    print("Path {} is not valid".format(path))
    exit(1)

all_datapoint_paths = glob.glob(path+'/**/*.png', recursive=True)
all_datapoint_paths = [os.path.abspath(p) for p in all_datapoint_paths]
all_datapoint_paths = np.random.permutation(all_datapoint_paths)

cut_idx = int(len(all_datapoint_paths)*.90)
tr_datapoint_list = all_datapoint_paths[:cut_idx]
va_datapoint_list = all_datapoint_paths[cut_idx:]

for filePath in tr_datapoint_list:
    print("{}".format(filePath))

for filePath in va_datapoint_list:
    print("{}".format(filePath))

tr_datapoint_df = pd.DataFrame({'path': tr_datapoint_list})
tr_datapoint_df.to_csv('/home/ram/rl/tr.lst',index=False)
va_datapoint_df = pd.DataFrame({'path': va_datapoint_list})
va_datapoint_df.to_csv('/home/ram/rl/va.lst',index=False)
