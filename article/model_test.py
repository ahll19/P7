#%%
import numpy as np
import matplotlib.pyplot as plt
from train_test_v2 import test, data_prep
from data_gen1 import load_data
import ksg
import torch
import pandas as pd


if __name__ == "__main__":
    path_model = 'trained_models/21-11-2022_12-09-29'
    tst_data = "data/17-11-2022_14-05-48/unif.npy"
    ksg_unbiased = False

    # Loading models:
    checkpoint = torch.load(path_model+'/model.pt')

    model_list = []
    for models in checkpoint.keys():
        model_list.append(checkpoint[models])

    specs = pd.read_csv(path_model+'/specs.csv', index_col=False)
    batch_size = int(specs['batch_size'][0])
    ksg_avg_err_trn = specs['avg error ksg'][0]
    network = [specs['network'][0][2:-2]][0]
    xy_len = int(specs['xy_len'][0])
    realisations = int(specs['realisations'][0])

    with open(path_model+'/loss_list.npy', 'rb') as f:
        loss_trn = np.load(f)
        loss_val = np.load(f)

    # Load data:
    data, label = load_data(tst_data)
    data, label = data[:realisations, :xy_len, :], label[:realisations]
    trn_loader, tst_loader = data_prep(data, label, batch_size=batch_size)

    test_dic = {}

    test_dic[network] = test(model_list[-1], tst_loader, batch_size, network=network)

    # KSG
    test_X = test_dic[network]['tst_data']

    ksg_list = np.zeros(len(test_X))
    for i in range(len(test_X)):
        ksg_list[i] = ksg.predict(test_X[i,:,0], test_X[i,:,1], 3)
    
    ksg_err = ksg_list - test_dic[network]['label_list'].T[0]
    ksg_avg_err = np.mean(np.abs(ksg_err))

    if ksg_unbiased:
        ksg_err += ksg_avg_err
        ksg_avg_err = np.mean(np.abs(ksg_err))

    ###################################
    # Plots
    ###################################

    print(f"Avg error model = {test_dic[network]['acc']:.5f}")
    print(f"Avg error KSG   = {ksg_avg_err:.5f}")

    ############### Residuals Scatter ###############
    figsize = (8, 6)
    fig, axes = plt.subplots(1, 1, sharey=True , figsize=figsize)
    fig.suptitle('Simple plot of results', fontsize=18)
    axes.plot(test_dic[network]['out_list'], '*', label="Model output   ")
    axes.plot(test_dic[network]['label_list'], "*", label="Label")
    axes.plot(ksg_list, '*', label="KSG")
    axes.grid()
    axes.set_title(network, fontsize=12)
    axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    plt.tight_layout()
    plt.show()

    ############### Residuals Histogram ###############
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    #fig.suptitle('Histogram of error (NN_geuss - label)', fontsize=18)
    hist_model = np.array(test_dic[network]['out_list']) - test_dic[network]['label_list']
    concat = np.concatenate([hist_model.reshape(-1), ksg_err])
    bins = np.linspace(np.min(concat)-0.1, np.max(concat)+0.1, 50)
    axes.hist(ksg_err, bins=bins, alpha=0.5, label="$r_{KSG}$")
    axes.hist(hist_model, bins=bins, alpha=0.5, label="$r_{network}$")
    axes.axvline(ksg_err.mean(), label='$r_{KSG}$ mean',color='#175987', linestyle='dashed', linewidth=2)
    axes.axvline(hist_model.mean(), label='$r_{network}$ mean', color='#db6d0b', linestyle='dashed', linewidth=2)
    min_ylim, max_ylim = plt.ylim()
    axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    plt.tight_layout()
    #plt.savefig(f'results/hist.pdf')
    plt.show()

    ############### Validation ###############
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('Validation vs Training', fontsize=18)
    ksgx = np.zeros(len(loss_trn))+ksg_avg_err_trn
    axes.plot(ksgx, label="KSG")
    axes.plot(loss_trn, label = 'Training')
    axes.plot(loss_val, label="Validation")
    axes.set_title('RNN', fontsize=12)
    axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    plt.tight_layout()
    # plt.savefig()
    plt.show()

    ############### Plotting lowest and highest MI ################
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    #fig.suptitle('Validation vs Training', fontsize=18)
    idx_min = np.argmin(label)
    axes.plot(data[idx_min,:,0], data[idx_min,:,1], '*')
    axes.plot(data[idx_min,0,0], data[idx_min,0,0], label=f'MI = {label[idx_min]:.3f}', ls='None')
    axes.set_xlabel('$X_i$')
    axes.set_ylabel('$Y_i$')
    axes.legend(bbox_to_anchor=(0.5, -0.08), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12, handlelength=0)
    plt.tight_layout()
    plt.savefig(f'results/data_lowest_mi.pdf')
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    #fig.suptitle('Validation vs Training', fontsize=18)
    idx_max = np.argmax(label)
    axes.plot(data[idx_max,:,0], data[idx_max,:,1], '*')
    axes.plot(data[idx_max,0,0], data[idx_max,0,0], label=f'MI = {label[idx_max]:.3f}', ls='None')
    axes.set_xlabel('$X_i$')
    axes.set_ylabel('$Y_i$')
    axes.legend(bbox_to_anchor=(0.5, -0.08), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12, handlelength=0)
    plt.tight_layout()
    plt.savefig(f'results/data_highest_mi.pdf')
    plt.show()


