import enum
import numpy as np
from scipy.stats import multivariate_normal as mn


def gen_MI(n, var_x, var_y, dim=2):
    """ Generates a set of random points according to <statistical inference
    of information in networks> on page 51. The third variable, Z, is 
    ignored as this is only used to create training data for MI, not CMI.

    Args:
        n (int): number of samples to draw
        var_x (float): Variance to multiply the identity with
        var_y (float): Variance to multiply the identity with
        dim (int, optional): Dimension of the samples to draw. Defaults to 2.

    Returns:
        _type_: _description_
    """
    I = np.identity(dim)
    zero = np.zeros(dim)
    
    x_drawn = mn.rvs(mean=zero, cov=var_x*I, size=n)
    y_drawn = np.zeros(x_drawn.shape)

    for i, x in enumerate(x_drawn):
        draw = mn.rvs(mean=x, cov=var_y*I)
        y_drawn[i] = draw
    
    mi = dim * np.log2(1 + var_x / var_y) / 2

    return x_drawn, y_drawn, mi
