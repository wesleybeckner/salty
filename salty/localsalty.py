from __future__ import print_function
import statistics
import time
from os.path import dirname, join
import pandas as pd
import os
import sys
import matplotlib.pylab as plt
import numpy as np
import itertools as it
from scipy.stats import uniform as sp_rand
from scipy.stats import mode
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from numpy.random import randint
import numpy.linalg as LINA
from sklearn.preprocessing import StandardScaler
__all__ = ["load_data", "suppress_stdout_stderr", "Benchmark", "check_name"]


"""
Salty is a toolkit for interacting with ionic liquid data from ILThermo
"""


def check_name(user_query, index=False):
    """
    checkName uses a database to return either SMILES or IUPAC 
    names of cations/anions.

    Default behavior is to return the SMILES encoding of an ion given
    the ion name as input.
    
    Parameters
    ------------------
    user_query : str
        string that will be used to query the database.
        
    Returns
    ------------------
    output: str
        either the name of the salt, cation, or anion; or SMILES of the
        salt, cation, or anion (SMILES for the salt are written as the 
        cation and ion SMILES strings separated by a comma)
    """
    ###Check to see that the database is present
    df_cation = load_data("cationInfo.csv")
    df_anion = load_data("anionInfo.csv")
    
    def _look_up_info_file(df):
        target_lookup = df.loc[(df == user_query).any(axis=1),:]
        input_type = df.loc[:,(df == user_query).any(axis=0)].columns.values
        column_index = df.columns.get_loc(input_type[0])
        row_index = df.loc[(df == user_query).any(axis=1),:].index.tolist()[0]
        return target_lookup, input_type, column_index, row_index
    
    try:
        target_lookup, input_type, column_index, row_index = _look_up_info_file(df_cation)
    except:
        try:
            target_lookup, input_type, column_index, row_index = _look_up_info_file(df_anion)
        except:
            print("query %s not found" % target_lookup)
            return 0
    if column_index == 1:
        target = target_lookup.iloc[0][column_index-1]
    else:
        target = target_lookup.iloc[0][column_index+1]
    if index:
        return target, row_index
    else:
        return target



def load_data(data_file_name, pickleFile=False, simpleList=False):
    """Loads data from module_path/data/data_file_name.
    Parameters
    ----------
    data_file_name : String. Name of csv or pickle file to be loaded from
    module_path/data/data_file_name. For example 'salt_info.csv'.
    Returns
    -------
    data : Pandas DataFrame
        A data frame. For example with each row representing one
        salt and each column representing the features of a given
        salt.
    """
    module_path = dirname(__file__)
    if pickleFile:
        with open(join(module_path, 'data', data_file_name), 'rb') as \
                pickle_file:
            data = pickle.load(pickle_file, encoding='latin1')
    elif simpleList:
        with open(join(module_path, 'data', data_file_name)) as csv_file:
            data = csv_file.read().splitlines()
    else:
        with open(join(module_path, 'data', data_file_name), 'rb') as csv_file:
            data = pd.read_csv(csv_file, encoding='latin1')
    return data


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class Benchmark:
    @staticmethod
    def run(function):
        timings = []
        stdout = sys.stdout
        for i in range(5):
            sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            sys.stdout = stdout
            timings.append(seconds)
            mean = statistics.mean(timings)
            print("{} {:3.2f} {:3.2f}".format(
                1 + i, mean,
                statistics.stdev(timings, mean) if i > 1 else 0))


