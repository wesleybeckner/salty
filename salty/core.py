from __future__ import print_function
import statistics
import time
from os.path import dirname, join
import pandas as pd
import sys
import dill
from math import inf
from math import log
from math import exp
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import StandardScaler
__all__ = ["load_data", "Benchmark",
           "check_name", "dev_model", "aggregate_data",
           "devmodel_to_array", "merge_duplicates"]


"""
Salty is a toolkit for interacting with ionic liquid data from ILThermo
"""


class dev_model():
    """
    the dev_model is the properly formated object to be passed to machine
    learning engine. The input features are all scaled and centered, the data
    summary describes the distribution of the data (in terms of state variables
    and output values).
    """
    def __init__(self, coef_data, data_summary, data):
        self.Coef_data = coef_data
        self.Data_summary = data_summary
        self.Data = data


def merge_duplicates(model_name):
    """
    Identifies repeated experimental values and returns mean values for those
    data along with their standard deviation. Only aggregates experimental
    values that have been aquired at the same temperature and pressure.

    Parameters
    ----------
    model_name: dev_model
        the dev_model object to be interrogated

    Returns
    -----------
    output_val: array
        array of the means of experimental measurements
    output_xtd: array
        array of the standard deviations of repeated experimental measurements
    running_size: int
        number of unique experiments
    salts: list
        names of salts included in the dataset
    """
    model_outputs = -6 + model_name.Data_summary.shape[0]
    devmodel = model_name
    cols = devmodel.Data.columns
    if (devmodel.Data.iloc[:, -(4 + model_outputs):-4].max() < 700).all():
        for output_index in range(model_outputs):
            devmodel.Data.iloc[:, -(5 + output_index)] = \
                devmodel.Data.iloc[:, -(5 + output_index)].apply(
                    lambda x: exp(float(x)))
    output_val = []
    output_xtd = []
    running_size = []
    for output_index in range(model_outputs):
        output_val.append(
            devmodel.Data.groupby(['smiles-cation', 'smiles-anion']
                                  )[cols[-(5 + output_index)]].mean())
        output_xtd.append(
            devmodel.Data.groupby(['smiles-cation', 'smiles-anion']
                                  )[cols[-(5 + output_index)]].std())
        running_size.append(
            devmodel.Data.groupby(['smiles-cation', 'smiles-anion']
                                  )[cols[-(5 + output_index)]].count())
    salts = (devmodel.Data["smiles-cation"] + "." +
             devmodel.Data["smiles-anion"]).unique()  # grab unique salts
    return output_val, output_xtd, running_size, salts


def devmodel_to_array(model_name, train_fraction=1):
    """
    a standardized method of turning a dev_model object into training and
    testing arrays

    Parameters
    ----------
    model_name: dev_model
        the dev_model object to be interrogated
    train_fraction: int
        the fraction to be reserved for training

    Returns
    ----------
    X_train: array
        the input training array
    X_test: array
        the input testing array
    Y_train: array
        the output training array
    Y_test: array
        the output testing array
    """
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
        data_files.append(load_data("%s_premodel.csv" % string))
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


def load_data(data_file_name, dillFile=False):
    """Loads data from module_path/data/data_file_name.
    Parameters
    ----------
    data_file_name : String. Name of csv or dill file to be loaded from
    module_path/data/data_file_name. For example 'salt_info.csv'.
    Returns
    -------
    data : Pandas DataFrame
        A data frame. For example with each row representing one
        salt and each column representing the features of a given
        salt.
    """
    module_path = dirname(__file__)
    if dillFile:
        with open(join(module_path, 'data', data_file_name), 'rb') as \
                dill_file:
            data = dill.load(dill_file)
    else:
        with open(join(module_path, 'data', data_file_name), 'rb') as csv_file:
            data = pd.read_csv(csv_file, encoding='latin1')
    return data


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
