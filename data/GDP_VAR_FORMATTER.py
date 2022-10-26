# %%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
with open("data/order_1 samples_7 choose3.pkl", "rb") as f:
    data_order1 = pickle.load(f)

with open("data/order_2 samples_5 choose3.pkl", "rb") as f:
    data_order2 = pickle.load(f)

# %%
# data shape: [realizations, length, dimension]
# label shape: [realization, dimension, dimension]
data = []
labels = []

for key, value in data_order1.items():
    data.append(value[0][1:])
    labels.append(value[1])

for key, value in data_order2.items():
    data.append(value[0][1:])
    labels.append(value[1])

data = np.asarray(data)
labels = np.asarray(labels)

np.save("data/jakob_formatted_data.npy", data)
np.save("data/jakob_formatted_labels.npy", labels)


