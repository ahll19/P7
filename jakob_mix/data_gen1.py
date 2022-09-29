import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mn


def gen_list(n, var_x, var_y, dim=2):
    """
    Generates a set of random points according to <statistical inference of information in networks>
    on page 51. The third variable, Z, is ignored as this is only used to create training data
    for MI, not CMI.
    The covariance matrices are both a scaled identity matrix.
    :param n: Number of points realized
    :param var_x: Variance of X
    :param var_y: Variance of Y
    :param dim: Dimension of the variables X and Y
    :return:
    :returns values: a realization of the process described in the book. Dimensions are [n, X/Y, dim].
                     e.g. [20, 0, 2] gives the second element the 20th realization of X.
    :returns mi: The mutual information: I(X; Y)
    """
    values = np.zeros((n, 2, dim))  # init
    values[:, 0, :] = mn.rvs(mean=np.zeros(dim), cov=var_x * np.identity(dim), size=(n, dim)).reshape((n, dim))

    for i, x in enumerate(values[:, 0, :]):
        values[i, 1, :] = mn.rvs(mean=x, cov=var_y * np.identity(dim), size=1)

    mi = dim * np.log2(1 + var_x / var_y) / 2

    return values, mi


def create_data(realisations, xy_len=1000, min_var0=0.1,
                 max_var0=10, min_var1=0.1, max_var1=10, dim=1):
    x0_var = np.random.uniform(min_var0, max_var0, realisations)
    x1_var = np.random.uniform(min_var1, max_var1, realisations)
    data = np.zeros((realisations, xy_len, 2))
    label = np.zeros([realisations])

    for i in range(realisations):
        _test, label[i] = gen_list(
            xy_len, x0_var[i], x1_var[i], dim)
        x, y = _test[:, 0, :], _test[:, 1, :]

        data[i] = np.hstack((x, y))

    return data, label


def save_data(file_path, data, labels):
    """
    Takes the path and name in one string and saves the file
    at that location as a .npy.

    :param file_path: The path with the path to the folder where the
    data is saved.
    :param data: The samples consisting of x and y
    :param labels: Labels to the data
    :return: No returns

    Example using savedata/loaddata:
    path = "test"
    save_data(path, data, labels) # saves data as .npy file
    data, labels = load_data(path) # loads .npy file
    """
    if file_path[-4:] != '.npy':
        file_path += '.npy'

    with open(file_path, 'wb') as f:
        np.save(f, data)
        np.save(f, labels)


def load_data(file_path):
    """
    Takes the path to where the .npy file is located.

    :param file_path: The path with the path to the folder where the
    data is saved.
    :return: data, labels

    Example using savedata/loaddata:
    path = "test"
    save_data(path, data, labels) # saves data as .npy file
    data, labels = load_data(path) # loads .npy file
    """
    if file_path[-4:] != '.npy':
        file_path += '.npy'

    with open(file_path, 'rb') as f:
        data = np.load(f)
        label = np.load(f)

    return data, label


if __name__ == "__main__":
    realisations, xy_len = 500, 5000
    data, labels = create_data(realisations, xy_len)

    path = f"data/realisations={realisations}_xy_len={xy_len}"
    save_data(path, data, labels)
    #data_loaded, labels_loaded = load_data(path)