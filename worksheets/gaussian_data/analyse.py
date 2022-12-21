#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

path_model = 'trained_models/21-11-2022_12-09-29'

with open(path_model + '/model_out-ksg-label_tst.npy', 'rb') as f:
    lab_model = np.load(f).reshape(1000)
    lab_ksg = np.load(f)
    lab_truth = np.load(f).reshape(1000)

res_model = lab_truth - lab_model
res_ksg = lab_truth - lab_ksg

p_model = float(stats.ttest_1samp(res_model, 0)[1])
p_ksg = float(stats.ttest_1samp(res_ksg, 0)[1])

print(f"Mean of RNN residuals: {np.mean(res_model):.4f}")
print(f"P-value for RNN unbiased: {p_model:.2f}")
print()
print(f"Mean of KSG residuals: {np.mean(res_ksg):.4f}")
print(f"P-value for KSG unbiased: {p_ksg:.2f}")

model_frame = pd.DataFrame(data=[res_model, lab_truth]).T
model_frame.columns = ["Residuals", "True label"]

model_frame.sort_values(by=["True label"])

ksg_frame = pd.DataFrame(data=[res_ksg, lab_truth]).T
ksg_frame.columns = ["Residuals", "True label"]

ksg_frame.sort_values(by=["True label"])

stats.levene(
    model_frame["Residuals"].values,
    ksg_frame["Residuals"].values,
    center="mean"
)

