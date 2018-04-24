from __future__ import print_function
from random import randint
import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
from salty import merge_duplicates


def expand_convex_hull(data, expansion_target=[1, 1.005], target_number=10):
    """
    A method of identifying property target values based on historical
    experimental data. The function returns targets outside the convex hull
    formed by the experimental data.

    Parameters
    ----------
    data: pandas dataframe
        the data frame to be interrogated
    expansion_target: float
        the relative area/volume expansion desired (compared to the initial
        area/volume described by the historical data)
    target_number: int
        the number of targets desired

    Returns
    ----------
    target_list: list of floats
        the list of property targets
    """
    vals, stds, size, salts = merge_duplicates(data)
    dataDf = pd.DataFrame([np.reshape(vals[1].values, (len(vals[1]))),
                           np.reshape(vals[0].values, (len(vals[0])))]).T
    hull = ConvexHull(dataDf)
    target_list = []
    while True:
        candidate = [randint(int(np.min(vals[1])), int(np.max(vals[1]))),
                     randint(int(np.min(vals[0])), int(np.max(vals[0])))]
        df = pd.DataFrame(candidate, index=np.arange(2).reshape(2)).T
        df = pd.concat((dataDf, df))
        newhull = ConvexHull(df)
        rel_size = (newhull.area / hull.area)
        if expansion_target[0] < rel_size < expansion_target[1]:
            target_list.append(candidate)
        if len(target_list) == target_number:
            break
    target_list = np.array(target_list)
    return target_list
