from __future__ import print_function
import statistics
import time
from os.path import dirname, join
import pandas as pd
import os
import sys
import pickle
import dill
from math import inf
from math import log
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import StandardScaler
__all__ = ["load_data", "suppress_stdout_stderr", "Benchmark",
           "check_name", "dev_model", "load_model", "aggregate_data",
           "devmodel_to_array"]


"""
Salty is a toolkit for interacting with ionic liquid data from ILThermo
"""


class qspr_model():
    def __init__(self, model, summary, descriptors):
        self.Model = model
        self.Summary = summary
        self.Descriptors = descriptors


class dev_model():
    def __init__(self, coef_data, data_summary, data):
        self.Coef_data = coef_data
        self.Data_summary = data_summary
        self.Data = data


def devmodel_to_array(model_name, train_fraction=1):
    if model_name is str:
        model_outputs = len(model_name.split("_"))
        pickle_in = open("../salty/data/MODELS/%s_devmodel.pkl" % model_name,
                         "rb")
        devmodel = dill.load(pickle_in)
    else:
        model_outputs = -6 + model_name.Data_summary.shape[0]
        devmodel = model_name
    rawdf = devmodel.Data
    rawdf = rawdf.sample(frac=1)
    datadf = rawdf.select_dtypes(include=[np.number])

    data = np.array(datadf)

    n = data.shape[0]
    d = data.shape[1]
    d -= model_outputs
    n_train = int(n * train_fraction)  # set fraction for training
    n_test = n - n_train

    X_train = np.zeros((n_train, d))  # prepare train/test arrays
    X_test = np.zeros((n_test, d))
    Y_train = np.zeros((n_train, model_outputs))
    Y_test = np.zeros((n_test, model_outputs))
    X_train[:] = data[:n_train, :-model_outputs]
    Y_train[:] = (data[:n_train, -model_outputs:].astype(float))
    X_test[:] = data[n_train:, :-model_outputs]
    Y_test[:] = (data[n_train:, -model_outputs:].astype(float))
    return X_train, Y_train, X_test, Y_test


def aggregate_data(data, T=[0, inf], P=[0, inf], data_ranges=None,
                   merge="overlap", feature_type=None, impute=False):
    """
    Aggregates molecular data for model training

    Parameters
    ----------
    data: list
        density, cpt, and/or viscosity
    T: array
        desired min and max of temperature distribution
    P: array
        desired min and max of pressure distribution
    data_ranges: array
        desired min and max of property distribution(s)
    merge: str
        overlap or union, defaults to overlap. Merge type of property sets
    feature_type: str
        desired feature set, defaults to RDKit's 2D descriptor set

    Returns
    -----------
    devmodel: dev_model obj
        returns dev_model object containing scale/center information,
        data summary, and the data frame
    """
    data_files = []
    for i, string in enumerate(data):
        data_files.append(load_data("MODELS/%s_premodel.csv" % string))
        if i == 0:
            merged = data_files[0]
        if i == 1:
            merged = pd.merge(data_files[0], data_files[1], sort=False,
                              how='outer')
        elif i > 1:
            merged = pd.merge(merged, data_files[-1], sort=False, how='outer')
    if merge == "overlap":
        merged.dropna(inplace=True)
    # select state variable and data ranges
    merged = merged.loc[merged["Temperature, K"] < T[1]]
    merged = merged.loc[merged["Temperature, K"] > T[0]]
    merged = merged.loc[merged["Pressure, kPa"] < P[1]]
    merged = merged.loc[merged["Pressure, kPa"] > P[0]]
    for i in range(1, len(data) + 1):
        merged = merged[merged.iloc[:, -i] != 0]  # avoid log(0) error
        if data_ranges:
            merged = merged[merged.iloc[:, -i] < data_ranges[::-1][i - 1][1]]
            merged = merged[merged.iloc[:, -i] > data_ranges[::-1][i - 1][0]]
    merged.reset_index(drop=True, inplace=True)
    # Create summary of dataset
    unique_salts = merged["smiles-cation"] + merged["smiles-anion"]
    unique_cations = repr(merged["smiles-cation"].unique())
    unique_anions = repr(merged["smiles-anion"].unique())
    actual_data_ranges = []
    for i in range(1, len(data) + 3):
        actual_data_ranges.append("{} - {}".format(
            str(merged.iloc[:, -i].min()), str(merged.iloc[:, -i].max())))
    a = np.array([len(unique_salts.unique()), unique_cations, unique_anions,
                 len(unique_salts)])
    a = np.concatenate((a, actual_data_ranges))
    cols1 = ["Unique salts", "Cations", "Anions", "Total datapoints"]
    cols2 = ["Temperature range (K)", "Pressure range (kPa)"]
    cols = cols1 + data[::-1] + cols2
    data_summary = pd.DataFrame(a, cols)
    # scale and center
    metaDf = merged.select_dtypes(include=["object"])
    dataDf = merged.select_dtypes(include=[np.number])
    cols = dataDf.columns.tolist()
    if impute:
        imp = Imputer(missing_values='NaN', strategy="median", axis=0)
        X = imp.fit_transform(dataDf)
        dataDf = pd.DataFrame(X, columns=cols)
    for i in range(1, len(data) + 1):
        dataDf.is_copy = False
        dataDf.iloc[:, -i] = dataDf.iloc[:, -i].apply(lambda x: log(float(x)))
    instance = StandardScaler()
    scaled_data = pd.DataFrame(instance.fit_transform(
        dataDf.iloc[:, :-len(data)]), columns=cols[:-len(data)])
    df = pd.concat([scaled_data, dataDf.iloc[:, -len(data):], metaDf], axis=1)
    mean_std_of_coeffs = pd.DataFrame([instance.mean_, instance.scale_],
                                      columns=cols[:-len(data)])
    devmodel = dev_model(mean_std_of_coeffs, data_summary, df)
    return devmodel


def load_model(data_file_name):
    """Loads data from module_path/data/MODELS/data_file_name.
    Parameters
    ----------
    data_file_name : String. Name of dill file to be loaded from
    module_path/data/data_file_name. For example 'density_devmodel.pkl'.
    Returns
    -------
    data : dev_model object
    """
    module_path = dirname(__file__)
    with open(join(module_path, 'data/MODELS', data_file_name), 'rb') as \
            pickle_file:
        data = dill.load(pickle_file)
    return data


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
    df_cation = load_data("cationInfo.csv")
    df_anion = load_data("anionInfo.csv")

    def _look_up_info_file(df):
        target_lookup = df.loc[(df == user_query).any(axis=1), :]
        input_type = df.loc[:, (df == user_query).any(axis=0)].columns.values
        column_index = df.columns.get_loc(input_type[0])
        row_index = df.loc[(df == user_query).any(axis=1), :].index.tolist()[0]
        return target_lookup, input_type, column_index, row_index

    try:
        target_lookup, input_type, column_index, row_index =\
            _look_up_info_file(df_cation)
    except BaseException:
        try:
            target_lookup, input_type, column_index, row_index = \
                _look_up_info_file(df_anion)
        except BaseException:
            print("query not found")
            return 0
    if column_index == 1:
        target = target_lookup.iloc[0][column_index - 1]
    else:
        target = target_lookup.iloc[0][column_index + 1]
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
