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


if __name__ == "__main__":
    _test, mutual_information = gen_list(2, 10, 10, 1)
    x, y = _test[:, 0, :], _test[:, 1, :]
    print(x,y)
    plt.scatter(x, y, alpha=.1)
    plt.show()
    print(f"Mutual information is approx {mutual_information:.2f} \n")