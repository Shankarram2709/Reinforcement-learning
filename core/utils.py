import cv2
import numpy as np
import os
import pandas as pd
import pathlib
#from tf.keras.models import Model
def get_datapoint_list(inp):
    """
    generates a list of paths
    :param inp: csv file where column path is used OR already a list of paths
    :return: list of paths
    """
    datapoint_list = []
    if isinstance(inp, str) and \
            pathlib.Path(inp).suffix in ['.lst', '.csv']:
        datapoint_list += pd.read_csv(inp)['path'].values.tolist()
    elif isinstance(inp, list):
        for i in inp:
            if isinstance(i, str):
                if pathlib.Path(i).suffix in ['.lst', '.csv']:
                    datapoint_list += pd.read_csv(i)['path'].values.tolist()
                else:
                    datapoint_list += i
            else:
                raise ValueError('bad file inserted')
    else:
        raise ValueError('corrupt input to get_datapoint_list data inserted')
    return datapoint_list

def ensure_dir(file_path):
    if file_path[-1] != '/':
        file_path += '/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)