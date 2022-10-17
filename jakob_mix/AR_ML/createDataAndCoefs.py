# Temp file that should be worked more upon
# Make generator available
import sys

sys.path.append("~/Git/P7")

import numpy as np
from generator import varp
from itertools import combinations
import os


# Specify the parameters for the datageneration (DNC means do not change)
num_var = 11  # DNC
series = list(combinations([i for i in range(12)], num_var))  # DNC
model_realizations = 5000  # DNC
model_orders = [1, 2, 5, 10]

# Generate the coefficients for all the models
data_list = varp.get_data_list()
coefs = dict()
covs = dict()

for i, s in enumerate(series):
    for j, p in enumerate(model_orders):
        missing_series = None
        for key_check in range(12):
            if key_check not in s:
                missing_series = key_check
                break

        # Keys for the dictionary: (index of time-series not used, model-order)
        key = (missing_series, p)
        coef, cov = varp.create_coefficients(data_list, s, p)

        coefs[key] = coef
        covs[key] = cov

# Save the coefficients, so that they can be used again later
for k1 in coefs:
    save_name = os.getcwd() + "/AR_ML/coefs/coefs_key_" + str(k1)
    varp.save_load_coefs(save_name, coefs[k1], covs[k1], True)

# Generate the data
all_keys = []
for key in coefs:
    data, final_cov = varp.simulate_VAR(coefs[key], covs[key], model_realizations)
    data = data[1:]

    # Calculate the MI for all the entries
    entries = combinations([i for i in range(num_var)], 2)
    mi_matrix = np.zeros((num_var, num_var))
    for e in entries:
        mutual_information = varp.calculate_MI(e[0], e[1], final_cov)
        mi_matrix[e[0], e[1]] = mutual_information
        mi_matrix[e[1], e[0]] = mutual_information

    # The matrix (A) stores the MI: A_ij = A_ji = I(x_j; x_I)
    np.save(os.getcwd() + f"/AR_ML/MI_matrices/MI_matrix_{key}", mi_matrix)
    np.save(os.getcwd() + f"/AR_ML/data/data_{key}", data.reshape(model_realizations, 1, num_var))
    np.save(os.getcwd() + f"/AR_ML/covariances/cov_{key}", final_cov)
    all_keys.append(key)

np.save(os.getcwd() + f"/AR_ML/keys/all_keys", all_keys)
