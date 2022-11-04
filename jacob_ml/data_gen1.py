#%%
from xml.etree.ElementTree import tostring
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mn
import pandas as pd

def gen_list1(n, var_x, var_y, dim=2):
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

    return values[:, 0, :], values[:, 1, :], mi

def create_data_for_csv(realisations, data_len=1000, min_var0=0.1, max_var0=10, min_var1=0.1, max_var1=10, dim=1):
    x0_var = np.random.uniform(min_var0, max_var0, realisations)
    x1_var = np.random.uniform(min_var1, max_var1, realisations)
    data = {'x': [], 'y': [], 'realisation': [], 'dim': []} #np.zeros((realisations, data_len, 2))
    label = np.zeros([realisations])

    for i in range(realisations):
        x, y, label[i] = gen_list( data_len, x0_var[i], x1_var[i], dim)
        for key in range(dim):
            data[f'x'] += list(x[:, key])
            data[f'y'] += list(y[:, key])
            data['realisation'] += [i]*data_len
            data['dim'] += [key]*data_len
    return data, label

def get_data_from_csv(path, realisations, data_len, dim):
    data_import = pd.read_csv(path, index_col=[0])
    data_result = np.zeros([realisations, data_len, 2])
    for i in range(realisations):
        x = np.array(data_import.query(f'realisation == {i} and dim == 0')['x'])
        y = np.array(data_import.query(f'realisation == {i} and dim == 0')['y'])
        data_result[i, :] = np.vstack((x, y))
    return data_result

if __name__ == "__main__":
    realisations = 2
    data_len = 2
    dim = 1
    min_var0=0.1, 
    max_var0=10, 
    min_var1=0.1, 
    max_var1=10
    data, labels = create_data_for_csv(realisations, data_len, min_var0, max_var0, min_var1, max_var1, dim)
    df_data = pd.DataFrame(data)
    print(df_data)
    df_labels = pd.DataFrame(labels)
    save_string_labels = f"jacob_ml/csv_files/labels_realisations={realisations}_data_len={data_len}_dim={dim}_min_var0={min_var0}_max_var0={max_var0}_min_var1={min_var1}_max_var1={max_var1}.csv"
    save_string_data = f"jacob_ml/csv_files/data_realisations={realisations}_data_len={data_len}_dim={dim}_min_var0={min_var0}_max_var0={max_var0}_min_var1={min_var1}_max_var1={max_var1}.csv"
    df_labels.to_csv(save_string_labels)
    df_data.to_csv(save_string_data)
    data_test = get_data_from_csv(save_string_data, realisations, data_len, dim)
    labels_test = np.array(pd.read_csv(save_string_labels, index_col=[0]))
    # print(data_test)
    # print(f'\n--------------------\n')
    # print(labels_test)

    # print(df_labels)
    # print(data)
    # print(labels)
    # _test, mutual_information = gen_list(3000, 10, 10, 1)
    # x, y = _test[:, 0, :], _test[:, 1, :]

    # plt.scatter(x, y, alpha=.1)
    # plt.show()
    # print(f"Mutual information is approx {mutual_information:.2f} \n")