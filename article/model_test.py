#%%
import numpy as np
import matplotlib.pyplot as plt
from train_test_v2 import test, data_prep
from data_gen1 import load_data
import ksg
import torch
import pandas as pd

def getTestResults(path_model_list):
    test_dic = {}
    data = {}
    label = {}
    loss_trn = {}
    loss_val = {}
    for path_model in path_model_list:
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
            loss_trn[network] = np.load(f)
            loss_val[network] = np.load(f)

        # Load data:
        data[network], label[network] = load_data(tst_data)
        data[network], label[network] = data[network][:realisations, :xy_len, :], label[network][:realisations]
        trn_loader, tst_loader = data_prep(data[network], label[network], batch_size=batch_size)
        test_dic[network] = test(model_list[-1], tst_loader, batch_size, network=network)

        # KSG
        test_X = test_dic[network]['tst_data']
        ksg_list = np.zeros(len(test_X))
        for i in range(len(test_X)):
            if network == 'FNN':
                ksg_list[i] = ksg.predict(test_X[i,:50], test_X[i,50:], 3)
            else:
                ksg_list[i] = ksg.predict(test_X[i,:,0], test_X[i,:,1], 3)
        ksg_err = ksg_list - test_dic[network]['label_list'].T[0]
        ksg_avg_err = np.mean(np.abs(ksg_err))

        if ksg_unbiased:
            ksg_err += ksg_avg_err
            ksg_avg_err = np.mean(np.abs(ksg_err))

    return {'test_dic': test_dic, 'ksg_avg_err': ksg_avg_err, 
            'ksg_list': ksg_list, 'loss_trn': loss_trn, 'loss_val': loss_val,
            'ksg_avg_err_trn': ksg_avg_err_trn, 'data': data, 
            'label': label, 'ksg_err': ksg_err}

def createPlots(test_results, networks):
    # for network in networks:
    #     print(f"Avg error {network} = {test_results['test_dic'][network]['acc']:.5f}")
    # print(f"Avg error KSG   = {test_results['ksg_avg_err']:.5f}")

    figsize = (10, 5)
    fig, axes = plt.subplots(1, 3, sharey=True , figsize=figsize)
    for i, network in enumerate(networks):
        fig.suptitle('Simple plot of results', fontsize=18)
        axes[i].plot(test_results['test_dic'][network]['out_list'], '*', label="Model output")
        axes[i].plot(test_results['test_dic'][network]['label_list'], "*", label="Label")
        axes[i].plot(test_results['ksg_list'], '*', label="KSG")
        axes[i].grid()
        axes[i].set_title(network, fontsize=12)
    axes[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for i, network in enumerate(networks):
        fig.suptitle('Histogram of error (NN_geuss - label)', fontsize=18)
        hist_model = np.array(test_results['test_dic'][network]['out_list']) - test_results['test_dic'][network]['label_list']
        concat = np.concatenate([hist_model.reshape(-1), test_results['ksg_err']])
        bins = np.linspace(np.min(concat)-0.1, np.max(concat)+0.1, 50)
        axes[i].hist(test_results['ksg_err'], bins=bins, alpha=0.5, label="KSG")
        axes[i].hist(hist_model, bins=bins, alpha=0.5, label="Model")
        axes[i].axvline(test_results['ksg_err'].mean(), label='KSG mean',color='#175987', linestyle='dashed', linewidth=2)
        axes[i].axvline(hist_model.mean(), label='Model mean', color='#db6d0b', linestyle='dashed', linewidth=2)
        axes[i].set_title(network, fontsize=12)
    axes[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    plt.tight_layout()
    #plt.savefig(f'results/hist.pdf')
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for i, network in enumerate(networks):
        fig.suptitle('Validation vs Training', fontsize=18)
        ksgx = np.zeros(len(test_results['loss_trn'][network]))+test_results['ksg_avg_err_trn']
        axes[i].plot(ksgx, label="KSG")
        axes[i].plot(test_results['loss_trn'][network], label = 'Training')
        axes[i].plot(test_results['loss_val'][network], label="Validation")
        axes[i].set_title(network, fontsize=12)
    axes[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    plt.tight_layout()
    # plt.savefig()
    plt.show()

    # Plotting lowest and highest MI
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('A trainingset with low mutial information', fontsize=18)
    idx_min = np.argmin(test_results['label'][network])

    axes.plot(test_results['data'][networks[0]][idx_min,:,0], test_results['data'][networks[0]][idx_min,:,1], '*')
    axes.plot(test_results['data'][networks[0]][idx_min,0,0], test_results['data'][networks[0]][idx_min,0,0], 
                label=f'MI = {test_results["label"][networks[0]][idx_min]:.3f}', ls='None')
    axes.set_xlabel('$X_i$')
    axes.set_ylabel('$Y_i$')


    axes.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12, handlelength=0)
    plt.tight_layout()
    # plt.savefig(f'results/data_lowest_mi.pdf')
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('A trainingset with high mutial information', fontsize=18)
    idx_max = np.argmax(test_results['label'][network])

    axes.plot(test_results['data'][networks[0]][idx_max,:,0], test_results['data'][networks[0]][idx_max,:,1], '*')
    axes.plot(test_results['data'][networks[0]][idx_max,0,0], test_results['data'][networks[0]][idx_max,0,0], \
                    label=f'MI = {test_results["label"][networks[0]][idx_max]:.3f}', ls='None')
    axes.set_xlabel('$X_i$')
    axes.set_ylabel('$Y_i$')

    axes.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12, handlelength=0)
    plt.tight_layout()
    # plt.savefig(f'results/data_highest_mi.pdf')
    plt.show()

#%%
if __name__ == "__main__":
    path_model_list = ['trained_models_FNN/06-12-2022_15-12-05', 'trained_models_CNN/06-12-2022_14-52-49', 
                'trained_models/21-11-2022_12-09-29']
    test_result = getTestResults(path_model_list)
    #%%
    networks = ['FNN', 'CNN', 'RNN']
    createPlots(test_result)
    
# %%
