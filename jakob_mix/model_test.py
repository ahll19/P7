#%%
import numpy as np
import matplotlib.pyplot as plt
from train_test_v2 import test, data_prep
from data_gen1 import load_data
import ksg
import torch
import pandas as pd


if __name__ == "__main__":
    path_model = 'trained_models/21-11-2022_10-33-41'
    tst_data = "data/realisations=5000_xy_len=1000.npy"
    ksg_unbiased = True

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

    #%% KSG
    test_X = test_dic[network]['tst_data']

    ksg_list = np.zeros(len(test_X))
    for i in range(len(test_X)):
        ksg_list[i] = ksg.predict(test_X[i,:,0], test_X[i,:,1], 3)
    
    ksg_err = ksg_list - test_dic[network]['label_list'].T[0]
    ksg_avg_err = np.mean(np.abs(ksg_err))

    if ksg_unbiased:
        ksg_err += ksg_avg_err
        ksg_avg_err = np.mean(np.abs(ksg_err))

    #%%
    ###################################
    # Plots
    ###################################

    print(f"Avg error model = {test_dic[network]['acc']:.5f}")
    print(f"Avg error KSG   = {ksg_avg_err:.5f}")

    figsize = (12, 5)
    fig, axes = plt.subplots(1, 1, sharey=True , figsize=figsize)
    fig.suptitle('Simple plot of results', fontsize=18)

    axes.plot(test_dic[network]['out_list'], '*', label="Model output")
    axes.plot(test_dic[network]['label_list'], "*", label="Label")
    axes.plot(ksg_list, '*', label="KSG")
    axes.grid()
    axes.set_title(network, fontsize=12)

    axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
  
    plt.tight_layout()
    plt.show()


    fig, axes = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('Histogram of error (NN_geuss - label)', fontsize=18)

    hist_model = np.array(test_dic[network]['out_list']) - test_dic[network]['label_list']
    concat = np.concatenate([hist_model.reshape(-1), ksg_err])

    bins = np.linspace(np.min(concat)-0.1, np.max(concat)+0.1, 50)
    
    axes.hist(ksg_err, bins=bins, alpha=0.5, label="KSG")
    axes.hist(hist_model, bins=bins, alpha=0.5, label="Model")


    axes.axvline(ksg_err.mean(), label='KSG mean',color='#175987', linestyle='dashed', linewidth=2)
    axes.axvline(hist_model.mean(), label='Model mean', color='#db6d0b', linestyle='dashed', linewidth=2)
    min_ylim, max_ylim = plt.ylim()
    #axes.text(hist_model.mean()+0.02, max_ylim*0.9, 'Model\nMean: {:.2f}'.format(hist_model.mean()))
    #axes.text(ksg_err.mean()+0.02, max_ylim*0.9, 'KSG\nMean: {:.2f}'.format(ksg_err.mean()))
    axes.set_title(network, fontsize=12)
        
    axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
   
    plt.tight_layout()
      
    #plt.savefig(f'../graphs/hist_{network}_lr_{learning_rate}_ep_{epochs}_bs_{batch_size}.pdf')
    plt.show()


    fig, axes = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('Validation vs Training', fontsize=18)

    ksgx = np.zeros(len(loss_trn))+ksg_avg_err_trn
    axes.plot(ksgx, label="KSG")
    axes.plot(loss_trn, label = 'Training')
    axes.plot(loss_val, label="Validation")
    axes.set_title('RNN', fontsize=12)

    axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    plt.tight_layout()
    # plt.savefig(f'../graphs/loss_{network}_lr_{learning_rate}_ep_{epochs}_bs_{batch_size}.pdf')
    plt.show()

# %%
