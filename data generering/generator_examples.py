#%%
# Imports and documentation
from generator import varp, gauss
import matplotlib.pyplot as plt
import numpy as np


"""
This document is shows how one might use the generator package by 
showing the use of some functions from the different modules


Gauss:              So far only used for a single function which creates 
                    labeled data for mutual information
                    Might be expanded later to also contain a function to
                    create labeled conditional mutual information.

varp:               Module for creating and labelling VAR(p) processes
"""


#%%
# Using the Gauss module

x_data, y_data, label = gauss.gen_MI(100, 20, 1, dim=1)


# %%
# Using the varp module to get data and coefficients

data = varp.get_data_list()  # takes some time because it has to download

_coefs1, _cov1 = varp.create_coefficients(data, [0, 1, 2], 1)
coefs2, cov2 = varp.create_coefficients(data, [1, 5, 9, 10], 4)
coefs3, cov3 = varp.create_coefficients(data, [i for i in range(12)], 10)

varp.save_load_coefs("test", _coefs1, _cov1, save=True)
coefs1, cov1 = varp.save_load_coefs("test")

#%%
# Using the varp module to create VAR(p) processes
process, covariance = varp.simulate_VAR(coefs1, cov1, 1000)
plt.plot(process[:, 0])
plt.title("x1")
plt.show()

MI_12 = varp.calculate_MI(0, 1, covariance)
MI_13 = varp.calculate_MI(0, 2, covariance)
MI_23 = varp.calculate_MI(1, 2, covariance)

print(
    f"I(x1; x2) = {MI_12}\n",
    f"I(x1; x3) = {MI_13}\n",
    f"I(x2; x3) = {MI_23}\n"
)
