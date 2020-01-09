# identify outliers with standard deviation

import pandas as pd
import numpy as np
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std

def remove_outliers(data):
    
    # calculate summary statistics

    data_mean = []
    data_std = []
    for i in range(len(data.columns)):
        data_mean.append(mean(data[data.columns[i]]))
        data_std.append(std(data[data.columns[i]]))
    
    stats = pd.DataFrame({'data_mean': data_mean,
                      'data_std': data_std,
                      'two_sigma': np.array(data_std)*2,
                      'three_sigma': np.array(data_std)*3}, index=data.columns)
    
    # identify outliers

    cut_off = np.array(stats.two_sigma)
    stats['lower'], stats['upper'] = np.array(data_mean) - cut_off, np.array(data_mean) + cut_off

    # identify outliers

    outliers_set = set()
    for feat in stats.index:
        low = stats.loc[feat, 'lower']
        up = stats.loc[feat, 'upper']
        outliers_set = outliers_set | set(data[(data[feat] < low) | (data[feat] > up)].index)

    outliers = data[data.index.isin(outliers_set)]
    number_outs = outliers.shape[0]
    
    # remove outliers

    outliers_removed = data[~data.index.isin(outliers_set)]
    number_outs_rem = outliers_removed.shape[0]

    
    return outliers_removed