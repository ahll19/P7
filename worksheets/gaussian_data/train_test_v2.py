import sys, os
sys.path.append(os.getcwd())
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from scipy.ndimage import uniform_filter1d
import numpy as np
import data_gen1
import models
import ksg
import copy
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_test(train_loader, test_loader, learning_rate, num_epoch, 
                input_size, batch_size, network='FNN') -> dict:
    """
    Training function for the Neural Network. 
    The training data has in the case with two random variabl-
    es the shape [realisations, input_size, 2]. Realisations 
    is the same as len(train_loader)*batch_size. During trani-
    ng this shape becomes [batch_size, input_size, 2]. 

    Args:
        train_loader (torch.utils.data.dataloader.DataLoader): 
            Training data. Also containing labels.
        learning_rate (float): 
            Used for the optimizer. If set too high then there
            Nan may occour.
        num_epoch (int): 
            Amount of epochs. This defines the amount of times
            the neural network will work through the entire d-
            ataset
        input_size (int): 
            Length of the random variables. May be reffered to
            as xy_len
        batch_size (int): 
            The amount of realisations of the random variables 
            that we use in each training loop.
        network (str, optional): 
            Specifies the type of NN that is used. Options are
            CNN, RNN, and RNN. Defaults to 'FNN'.

    Returns:
        trn_dict = {
            'model': trained model.
            'loss_list': loss list from training.
            'loss_val_list': loss list from test data-set
            'model_list': a list of models corresponding to one model 
                for each epoch.
        }
    """
    if network == 'CNN':
        model = models.CNN(input_size, batch_size)
    elif network == 'RNN':
        model = models.RNN(input_size, hidden_dim=input_size, layer_dim=2,
                           output_dim=1, batch_size=batch_size)
    else:
        model = models.FNN(input_size)

    print(model)
    # loss and optimizer
    # criterion = nn.MSELoss(reduction='sum')
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    loss_list = []
    loss_temp = []
    loss_tst_list = []
    loss_temp_test = []
    model_list =[]
    label_list = []
    for epoch in tqdm(range(num_epoch)):
        for i, (dat_trn, labels) in enumerate(train_loader):
            if network == 'CNN':
                dat_trn = dat_trn.to(device)
            elif network == 'RNN':
                dat_trn = dat_trn.to(device)
            else:
                dat_trn = dat_trn.reshape(batch_size, -1).to(device)

            labels = labels.view(labels.shape[0], 1).to(device) # makes it a column vector

            # forward
            output = model(dat_trn)
            loss = criterion(output, labels)
            loss_temp.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model_list.append(copy.deepcopy(model))

        loss_list.append(np.mean(np.array(loss_temp)))
        loss_temp = []
        with torch.no_grad():
            for dat_tst, lab_tst in test_loader:
                if network == 'CNN':
                    dat_tst = dat_tst.to(device)
                elif network == 'RNN':
                    dat_tst = dat_tst.to(device)
                else:
                    dat_tst = dat_tst.reshape(batch_size, -1).to(device)

                lab_tst = lab_tst.view(labels.shape[0], 1).to(device)
                label_list += lab_tst.tolist()

                output_tst = model(dat_tst) # trained model

                loss_tst = criterion(output_tst, lab_tst)
                loss_temp_test.append(loss_tst.item())

            loss_tst_list.append(np.mean(np.array(loss_temp_test)))
            loss_temp_test = []

    hsh = '\n#################################\n'
    print(hsh+"# TRAINING DONE"+hsh)

    trn_dict = {
        'model': model,
        'loss_list': loss_list,
        'loss_val_list': loss_tst_list,
        'model_list': model_list
    }

    return trn_dict
  
def test(model, test_loader, batch_size, network='FNN') -> dict:
    """
    This function is testing the trained model. This model c-
    an be obtained using train().

    Args:
        model (nn.Module): 
            Trained model
        test_loader (torch.utils.data.dataloader.DataLoader): 
            Test data.
        batch_size (int):
            Used to reshape data if network='FNN'
        network (str, optional): 
            Specifies the type of NN that is used. Options are
            CNN, RNN, and RNN. Defaults to 'FNN'.

    Returns:
        test_dict = {
            'out_list': Estimated labels using the model.
            'label_list': Labels of the training data.
            'acc': The average error between out_list and label_list.
            'tst_data': The test dataset.
        }
    """
    with torch.no_grad():
        out_list = []
        label_list = []

        tst_dat = []

        n_samples = 0
        n_diff = 0
        for dat_tst, labels in test_loader:
            if network == 'CNN':
                dat_tst = dat_tst.to(device)
            elif network == 'RNN':
                dat_tst = dat_tst.to(device)
            else:
                dat_tst = dat_tst.reshape(batch_size, -1).to(device)

            labels = labels.view(labels.shape[0], 1).to(device)
            label_list += labels.tolist()

            outputs = model(dat_tst) # trained model
            out_list += outputs.tolist()
            n_diff += torch.mean(torch.abs(outputs-labels))
            tst_dat.append(dat_tst)
            n_samples += 1

        acc = n_diff/n_samples

    test_dat = torch.cat(tst_dat, dim=0).numpy()
    label_list = np.array(label_list)

    test_dict = {
        'out_list': out_list,
        'label_list': label_list,
        'acc': acc,
        'tst_data': test_dat
    }

    return test_dict

def data_prep(data, label, batch_size, test_size=0.1, shuffle_test=True,
              shuffle_train=True) -> DataLoader:
    """Converts data and labels to training and test
    data as Dataloaders

    Args:
        data (np.array): Timeseries
        label (np.array): labels
        batch_size (int): batch size
        test_size (float, optional): amount of data saved for test. 
                                     Defaults to 0.1.
        shuffle_test (boolian, optional): Defaults to True.
        shuffle_train (boolian, optional): Defaults to True.

    Returns:
        trn_loader : DataLoader 
        tst_loader : DataLoader
    """

    data_train, data_test, label_train, label_test = tts(data, label, test_size=test_size)

    train_data = models.NumbersDataset(data_train, label_train)
    test_data = models.NumbersDataset(data_test, label_test)

    trn_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_train)
    tst_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test)

    return trn_loader, tst_loader

if __name__ == "__main__":
    path = "data/17-11-2022_14-05-48"
    data, label = data_gen1.load_data(path+'/unif.npy')

    '''
    best results achieved: 
    realisations = 500
    xy_len = 50
    batch_size = 10
    learning_rate = 0.00001
    epochs = 1000
    '''
    realisations = len(label)
    xy_len = 50
    batch_size = 100
    learning_rate = 0.0001
    epochs = 20
    gif = False
    NN = 'RNN'
    save_model_path = ['trained_models_'+NN+'/'] # Saves models if list is non-empty

    networks = [NN]
    train_test_dic = {}
    test_dic = {}

    data, label = data[:realisations, :xy_len, :], label[:realisations]

    train_loader, test_loader = data_prep(data, label, batch_size=batch_size)

    for network in networks:
        # train and running test
        train_test_dic[network] = train_test(train_loader, test_loader, learning_rate, epochs,
                                    xy_len, batch_size, network=network)

        # test
        test_dic[network] = test(train_test_dic[network]['model'], test_loader, batch_size, network=network)
        print(f"Avg error model = {test_dic[network]['acc']:.5f}")

    # Calculating ksg:
    test_X = test_dic[network]['tst_data']

    ksg_list = np.zeros(len(test_X))
    # for i in range(len(test_X)):
    #     ksg_list[i] = ksg.predict(test_X[i,:,0], test_X[i,:,1], 3)
    
    ksg_err = ksg_list - test_dic[network]['label_list'].T[0]
    ksg_avg_err = np.mean(np.abs(ksg_err))

    print(f"Avg error KSG   = {ksg_avg_err:.5f}")

    if save_model_path:
        import pandas as pd
        from datetime import datetime

        now = datetime.now()
        time = now.strftime("%d-%m-%Y_%H-%M-%S")
        
        path_model = save_model_path[0]+time
        isExist = os.path.exists(path_model)
        if not isExist:
            os.makedirs(path_model)
            print("The new directory is created!")

        sample_dict = {
            'data file' : path,
            'realisations' : realisations,
            'xy_len' : xy_len,
            'batch_size' : batch_size,
            'learning_rate' : learning_rate,
            'epochs' : epochs,
            'network' : networks,
            'avg error model' : round(test_dic[network]['acc'].item(), 5),
            'avg error ksg' : round(ksg_avg_err, 5)
        }
        df = pd.DataFrame.from_dict([sample_dict]) 
        df.to_csv (path_model+f'/specs.csv', index = False, header=True)

        # to save:
        nam = 'model'
        dictmodels = {nam+f"{i}" : model for i, model in enumerate(train_test_dic[NN]['model_list'])}
        torch.save(dictmodels, path_model+f'/model.pt')

        loss_trn = train_test_dic[network]['loss_list']
        loss_val = train_test_dic[network]['loss_val_list']
        with open(path_model+'/loss_list.npy', 'wb') as f:
            np.save(f, loss_trn)
            np.save(f, loss_val)
        
        model_out = test_dic[network]['out_list']
        ksg = ksg_list
        label_tst = test_dic[network]['label_list']
        with open(path_model+'/model_out-ksg-label_tst.npy', 'wb') as f:
            np.save(f, model_out)
            np.save(f, ksg)
            np.save(f, label_tst)
        

            
    ###################################
    # Plots
    ###################################
    figsize = (12, 5)
    fig, axs = plt.subplots(1, len(networks), sharey=True , figsize=figsize)
    fig.suptitle('Simple plot of results', fontsize=18)
    for key, network in enumerate(networks):
        if len(networks) == 1:
            axes = axs
        else:
            axes = axs[key]

        axes.plot(test_dic[network]['out_list'], '*', label="Model output")
        axes.plot(test_dic[network]['label_list'], "*", label="Label")
        axes.plot(ksg_list, '*', label="KSG")
        axes.grid()
        axes.set_title(network, fontsize=12)

    if len(networks) == 1:
        axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    else:
        axs[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.show()


    fig, axs = plt.subplots(1, len(networks), figsize=figsize)
    fig.suptitle('Loss throughout the learning', fontsize=18)
    for key, network in enumerate(networks):
        if len(networks) == 1:
            axes = axs
        else:
            axes = axs[key]
        axes.plot(train_test_dic[network]['loss_list'], label="Loss")
        axes.set_title(network, fontsize=12)
    
    if len(networks) == 1:
        axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12)
    else:
        axs[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12)
    
    plt.show()
    

    fig, axs = plt.subplots(1, len(networks), figsize=figsize)
    fig.suptitle('Histogram of error (NN_geuss - label)', fontsize=18)
    for key, network in enumerate(networks):
        if len(networks) == 1:
            axes = axs
        else:
            axes = axs[key]

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
        

    if len(networks) == 1:
        axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    else:
        axs[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    
    plt.tight_layout()
      
    #plt.savefig(f'../graphs/hist_{network}_lr_{learning_rate}_ep_{epochs}_bs_{batch_size}.pdf')
    plt.show()

    fig, axs = plt.subplots(1, len(networks), figsize=figsize)
    fig.suptitle('Validation vs Training', fontsize=18)
    for key, network in enumerate(networks):
        if len(networks) == 1:
            axes = axs
        else:
            axes = axs[key]

        ksgx = np.zeros(len(train_test_dic[network]['loss_list']))+ksg_avg_err
        axes.plot(ksgx, label="KSG")
        axes.plot(train_test_dic[network]['loss_list'], label = 'Training')
        axes.plot(train_test_dic[network]['loss_val_list'], label="Validation")
        axes.set_title(network, fontsize=12)

    if len(networks) == 1:
        axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    else:
        axs[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=12)
    
    plt.tight_layout()
    # plt.savefig(f'../graphs/loss_{network}_lr_{learning_rate}_ep_{epochs}_bs_{batch_size}.pdf')
    plt.show()


    #gif
    if gif:
        import imageio.v2 as imageio
        import os
        out_arr = np.zeros((epochs, xy_len))
        labels_arr = np.zeros((epochs, xy_len))
        errors_arr = np.zeros(epochs)
        for i, model in enumerate(train_test_dic[network]['model_list']):
            dic = test(model, test_loader, batch_size, network=network)
            out_arr[i] = np.array(dic['out_list']).T[0]
            labels_arr[i] = dic['label_list'].T[0]
            errors_arr[i] = np.mean(np.abs(out_arr[i]-labels_arr[i]))
        
        filenames = []
        for i, (out, lab) in tqdm(enumerate(zip(out_arr, labels_arr))):
            figsize = (12, 5)
            fig, axs = plt.subplots(1, len(networks), figsize=figsize)
            fig.suptitle(f'Epoch {i+1}', fontsize=18)
            for key, network in enumerate(networks):
                if len(networks) == 1:
                    axes = axs
                else:
                    axes = axs[key]

                axes.plot(out[:], '*', label="Model output")
                axes.plot(lab[:], "*", label="Label")
                axes.plot(ksg_list[:], '*', label="KSG")
                axes.grid()
                axes.set_title(network, fontsize=12)
                lower = np.min(np.concatenate([out_arr, labels_arr]))-0.3
                upper = np.max(np.concatenate([out_arr, labels_arr]))+0.3
                axes.set_ylim([lower, upper])

                ax_bar = fig.add_axes([0.92,0.11,0.1,0.77])
                error_bar = ['Avg L1 error']
                errors = [errors_arr[i]]
                ax_bar.bar(error_bar, errors, label='Model error')
                ax_bar.set_ylim([0,np.max(errors_arr)])
                ax_bar.plot([-1,1],[ksg_avg_err]*2, color='g', label="KSG error")
                ax_bar.yaxis.tick_right()
                #ax_bar.set_yticklabels([])
                ax_bar.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12)

            if len(networks) == 1:
                axes.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12)
            else:
                axs[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12)

            #gifstuff
            filename = f"{i}.png"
            for j in range(3):
                filenames.append(filename)
            
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

        with imageio.get_writer("pred_MSE.gif", mode='I') as writer:
            for file in filenames:
                image = imageio.imread(file)
                writer.append_data(image)

        for file in set(filenames):
            os.remove(file)

# %%
