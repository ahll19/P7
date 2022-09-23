# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mn
import pandas as pd
import pandas_datareader as pdr
import statsmodels as sm
from scipy.interpolate import interp1d
from statsmodels.tsa.api import VAR


def MI(idx1, idx2, cov):
    """Calculates the mutual information between the marginals of a multivariate gaussian distribution: I(X1; X2)

    Args:
        idx1 (int): Index of X1 in the vector it came from
        idx2 (int): Index of X2 in the vector it came from
        cov (2D array float): Covariance matrix of the vector X, which X1 and X2 came from

    Returns:
        float: Returns the mutual information of X1 and X2
    """
    top = cov[idx1, idx1] * cov[idx2, idx2]
    bot = top - 2*cov[idx1, idx2]

    return .5 * np.log2(top / bot)


def simulate_AR(coefficients, noise_cov, num_steps, x0=None):
    """Simulate the AR(p) process, with iid zero-mean gaussian additive noise.

    Args:
        coefficients (list of arrays): List of the coefficient matrices to use in the realization. Entries in list are 2D float arrays
        noise_cov (2D array float): Covariance matrix of the gaussian additive noise
        num_steps (int): Number of iterations for which the simulation should run
        x0 (1D float array, optional): The starting value of the process. If specified the derivations of the MI don't fit as of the end of september 2022. Defaults to None.
    
    Returns:
        array float: Returns the resulting array of the realization of the process
    """
    # Setup
    p = len(coefficients)
    dim = noise_cov.shape[0]
    results = np.zeros((num_steps + 1, dim))

    # initialize the initial value
    if x0 is None:
        x0 = np.zeros(p)
    results[0] = x0

    # simulate the process
    for i in range(1, num_steps + 1):
        iter_sum = np.zeros(dim)

        # Try except makes sure we don't try to index results too far back. Could be solved with a bool check changed when i >= p
        for j in range(p):
            try:
                iter_sum += coefficients[j] @ results[i - (p + 1)]
            except IndexError:
                break
        
        # Save iteration to results
        iter_sum += np.random.multivariate_normal(mean=np.zeros(dim), cov=noise_cov)
        results[i] = iter_sum
    
    return results
    

def get_fred_data():
    """Pulls data from FRED

    Returns:
        List of arrays: A list of time-series data pulled from the FRED website.
    """
    # Info to get data from fred
    data_names = [
            "PCOPPUSDM",
            "PWHEAMTUSDM",
            "PRAWMINDEXM",
            "APU000072610",
            "APU0000703112",
            "DHHNGSP",
            "PSUNOUSDM",
            "WTISPLC",
            "DCOILWTICO",
            "DCOILBRENTEU",
            "PNGASEUUSDM",
            "APU0000708111",
        ]
    dates = ["1990-01-01", "2020-01-01"]
    data_list = []

    # load the data from the website
    for i in range(len(data_names)):
        name = data_names[i]
        _dat = pdr.get_data_fred(name, dates[0], dates[1])
        _dat = _dat.interpolate(method='nearest').ffill().bfill()  # interpolate to remove NaN values
        data_list.append(_dat)
    
    # Convert to numpy
    arr_dat = []
    for dat in data_list:
        arr_dat.append(dat.to_numpy())
    
    return arr_dat


def interpolate_data(data_list, recalled=False):
    """Interpolates the data loaded by the function get_fred_data(). Should be able to be generalized if need be.

    Args:
        data_list (List of arrays): List returned by get_fred_data()
        recalled (bool, optional): DO NOT CHANGE MANUALLY. This boolean was used when developing the function. If set to true no second run will be done when trying to interpolate. Defaults to False.

    Raises:
        Exception: Raises an exception if NaN is present in the interpolated data. If that is the case Anders will jump in front of a bus.

    Returns:
        List of arrays: List of interpolated time-series data from get_fred_data()
    """
    max_data_len = 0
    min_data_len = np.Inf

    # get longest length of data
    for dat in data_list:
        max_data_len = max(max_data_len, len(dat))
        min_data_len = min(min_data_len, len(dat))
    
    # we interpolate from 0 to 10e4 such that we have an axis, and dont need much decimal precision
    interp_axis = np.linspace(1, 10e4-1, max_data_len)
    interpolated_data = []

    for dat in data_list:
        n = len(dat)
        if n != max_data_len:
            _axis = np.linspace(0, 10e4, n)
            f = interp1d(_axis, dat.reshape(len(dat)), kind='cubic', fill_value=np.mean(dat), bounds_error=False)
            inter_data = f(interp_axis)
        
        else:
            inter_data = dat
        
        interpolated_data.append(inter_data.reshape(len(inter_data)))

    # Scipy is kinda weird, so we check if there are NaN values in the interpolated data
    has_nan = []
    
    # check each array if it contains nan
    for inter in interpolated_data:
        has_nan.append(np.isnan(np.sum(inter)))
    
    if True not in has_nan:  # everything went according to plan, we return the data
        return interpolated_data
    
    if not recalled:  # something went wrong in the first case, we try again
        interpolate_data(data_list, recalled=True)
    
    # We tried twice, and didn't get the data to interpolate
    raise Exception("Data would not interpolate in two tries in function interpolate_data()")

    return interpolated_data


def create_coefficients(data_list, series_list, model_order):
    """Creates the coefficients needed to simulate the data, using statsmodels package. 

    Args:
        data_list (List of arrays): List of time-series data returned by the interpolation function.
        series_list (List of int): Indexes (0, ..., 11) which decides what data to use in the model. Len(series_list) = k -> model of dimension k
        model_order (Int): Order of the model. Must be int >= 1

    Returns:
        Tuple (array, array): Results from VAR.fit(model_order)
            array: coefficient matrices. Size is [p, k, k] where p is the model order, and k is the dimension of the model
            array: covariance matrix. Size is [k, k]
    """
    new_data = []

    for idx in series_list:
        new_data.append(data_list[idx])
    
    exog_var = np.array(new_data)

    model = VAR(exog_var.T)
    result = model.fit(model_order)

    return result.coefs, result.sigma_u


def save_load_coefs(name, coefficients=None, covariance=None, save=False):
    """Save or Load coeffiecients from the AR(p) model

    Args:
        name (str): name under which the coefficients should be save. using name='abc' saves or loads files 'abc_coefs.npy' and 'abc_cov.npy'
        coefficients (array): array of the coefficients returned from create_coefficients()
        covariance (array): array of the covariance returned from create_coefficients()
        save (bool, optional): If set to True the function saves the data specified in coefficients and covariance. If False the functions loads files specified by name. Defaults to False.

    Returns:
        Tuple (array, array | None): If save is false the function returns None
            array: Coefficient matrix loaded by name
            array: Covariance matrix loaded by name
    """
    coef_str = name + "_coefs.npy"
    cov_str = name + "_cov.npy"

    if save:
        np.save(coefficients, coef_str)
        np.save(covariance, cov_str)

        return None
    
    coefs = np.load(coef_str)
    cov = np.load(cov_str)

    return coefs, cov
    