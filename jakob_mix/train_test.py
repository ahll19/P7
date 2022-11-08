#%%
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training loop
def train(train_loader, learning_rate, num_epoch, input_size, batch_size, network='FNN'):
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
        Model: Trained model. 
        Loss: The loss that is calculated during training.
    """
    if network == 'CNN':
        model = models.CNN(input_size, batch_size)
    elif network == 'RNN':
        model = models.RNN(input_size, hidden_dim=input_size, layer_dim=1,
                           output_dim=1, batch_size=batch_size)
    else:
        model = models.FNN(input_size)

    print(model)
    # loss and optimizer
    #criterion = nn.MSELoss(reduction='sum')
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    loss_list = []
    model_list =[]
    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            if network == 'CNN':
                sample = images.to(device)
            elif network == 'RNN':
                sample = images.to(device)
            else:
                sample = images.reshape(batch_size, -1).to(device)

            labels = labels.view(labels.shape[0], 1).to(device) # makes it a column vector

            # forward
            output = model(sample)
            loss = criterion(output, labels)
            loss_list.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            if (epoch) % 100 == 0 and i % 10 == 0:
                print(
                    f'epoch {epoch} / {num_epoch-1}, step {i}/{n_total_steps-1} loss = {loss.item():.8f}')
        model_list.append(copy.deepcopy(model))
        
    print(f"\n#################################\n# TRAINING DONE\n#################################\n")

    return model, loss_list, model_list

def train_test(train_loader, test_loader, learning_rate, num_epoch, input_size, batch_size, network='FNN'):
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
        Model: Trained model. 
        Loss: The loss that is calculated during training.
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
    #criterion = nn.MSELoss(reduction='sum')
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    loss_list = []
    loss2_list = []
    model_list =[]
    label_list = []
    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            if network == 'CNN':
                sample = images.to(device)
            elif network == 'RNN':
                sample = images.to(device)
            else:
                sample = images.reshape(batch_size, -1).to(device)

            labels = labels.view(labels.shape[0], 1).to(device) # makes it a column vector

            # forward
            output = model(sample)
            loss = criterion(output, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            if (epoch) % 100 == 0 and i % 10 == 0:
                print(
                    f'epoch {epoch} / {num_epoch-1}, step {i}/{n_total_steps-1} loss = {loss.item():.8f}')
        model_list.append(copy.deepcopy(model))

        loss_list.append(loss.item())
        with torch.no_grad():
            for images2, labels2 in test_loader:
                if network == 'CNN':
                    sample2 = images2.to(device)
                elif network == 'RNN':
                    sample2 = images2.to(device)
                else:
                    sample2 = images2.reshape(batch_size, -1).to(device)

                labels2 = labels2.view(labels.shape[0], 1).to(device)
                label_list += labels2.tolist()

                output2 = model(sample2) # trained model

                loss2 = criterion(output2, labels2)
            loss2_list.append(loss2.item())


    print(f"\n#################################\n# TRAINING DONE\n#################################\n")

    return { 'model': model, 'loss_list': loss_list, 'loss2_list': loss2_list, 'model_list': model_list}
  
def test(model, test_loader, batch_size, network='FNN'):
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
        out_list: Estimated labels using the model.
        label_list: Labels of the training data.
        acc: The average error between out_list and label_list.
    """
    with torch.no_grad():
        out_list = []
        label_list = []
        n_samples = 0
        n_diff = 0
        for images, labels in test_loader:
            if network == 'CNN':
                sample = images.to(device)
            elif network == 'RNN':
                sample = images.to(device)
            else:
                sample = images.reshape(batch_size, -1).to(device)

            labels = labels.view(labels.shape[0], 1).to(device)
            label_list += labels.tolist()

            outputs = model(sample) # trained model
            out_list += outputs.tolist()
            n_diff += torch.mean(torch.abs(outputs-labels))
            print(torch.abs(outputs-labels))
            n_samples += 1

        acc = n_diff/n_samples

    return { 'out_list': out_list, 'label_list': label_list, 'acc': acc }

def test_models(model_list, test_loader, batch_size, network='FNN'):
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
        out_list: Estimated labels using the model.
        label_list: Labels of the training data.
        acc: The average error between out_list and label_list.
    """
    acc_list = []
    for model in model_list:
        with torch.no_grad():
            out_list = []
            label_list = []
            n_samples = 0
            n_diff = 0
            for images, labels in test_loader:
                if network == 'CNN':
                    sample = images.to(device)
                elif network == 'RNN':
                    sample = images.to(device)
                else:
                    sample = images.reshape(batch_size, -1).to(device)

                labels = labels.view(labels.shape[0], 1).to(device)
                label_list += labels.tolist()

                outputs = model(sample) # trained model
                out_list += outputs.tolist()
                n_diff += torch.mean(torch.abs(outputs-labels))
                n_samples += 1

            acc = n_diff/n_samples
        acc_list.append(acc.item())

    return out_list, label_list, acc, acc_list


if __name__ == "__main__":
    path = "data/realisations=5000_xy_len=1000.npy"
    '''
    best results achieved: 
    realisations = 500
    xy_len = 50
    batch_size = 10
    learning_rate = 0.00001
    epochs = 1000
    '''
    realisations = 500
    xy_len = 50
    batch_size = 50
    learning_rate = 0.0001
    epochs = 250
    networks = ['FNN', 'CNN', 'RNN']
    train_test_dic = {}
    test_dic = {}
    # save_str = f'network_{network}_epochs_{epochs}_lr_{learning_rate}_bs_{batch_size}_xylen_{xy_len}_realisations_{realisations}'

    data, label = data_gen1.load_data(path)

    data, label = data[:realisations, :xy_len, :], label[:realisations]

    # creating test/train data
    data_train, data_test, label_train, label_test = tts(data, label, test_size=0.1)

    train_data = models.NumbersDataset(data_train, label_train)
    test_data = models.NumbersDataset(data_test, label_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # train
    # model_trained, loss, model_list = train(train_loader, learning_rate, epochs,
    #                             xy_len, batch_size, network=network)

    for network in networks:
        # train and running test
        train_test_dic[network] = train_test(train_loader, test_loader, learning_rate, epochs,
                                    xy_len, batch_size, network=network)
        # model_trained, loss_list, loss2_list, model_list = train_test(train_loader, test_loader, learning_rate, epochs,
        #                             xy_len, batch_size, network=network)
        # loss = loss_list

        # test
        test_dic[network] = test(train_test_dic[network]['model'], test_loader, batch_size, network=network)
        # output_test, label_test, acc = test(model_trained, test_loader, batch_size, network=network)
        # output_test, label_test, acc, acc_list = test_models(model_list, test_loader, batch_size, network=network)

    # Calculating ksg:
    ksg_list = np.zeros(len(data_test))
    for i in range(len(data_test)):
        ksg_list[i] = ksg.predict(data_test[i][:, 0], data_test[i][:, 1])

    avg_ksg = np.sum(ksg_list)/len(ksg_list)

    # print(f"Avg error model = {acc:.5f}")
    # print(f"Avg error KSG   = {avg_ksg:.5f}")

    #%%
    figsize = (12, 5)

    fig, axs = plt.subplots(1, len(networks), figsize=figsize)
    fig.suptitle('Simple plot of results', fontsize=18)
    for key, network in enumerate(networks):
        axs[key].plot(test_dic[network]['out_list'], '*', label="Model output")
        axs[key].plot(test_dic[network]['label_list'], "*", label="Label")
        axs[key].plot(np.reshape(ksg_list, (-1, 1)), '*', label="KSG")
        axs[key].grid()
        axs[key].set_title(network, fontsize=12)
    axs[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12)
    # plt.plot(output_test, "*", label="Model output")
    # plt.plot(label_test, "*", label="Label")
    # plt.plot(np.reshape(ksg_list, (-1, 1)), '*', label="KSG")
    # plt.title('Simple plot of results')
    # plt.legend()
    # plt.grid()
    plt.show()

    fig, axs = plt.subplots(1, len(networks), figsize=figsize)
    fig.suptitle('Loss throughout the learning', fontsize=18)
    for key, network in enumerate(networks):
        axs[key].plot(train_test_dic[network]['loss_list'], label="Loss")
        axs[key].plot(uniform_filter1d(train_test_dic[network]['loss_list'], size = 100), label="Loss moving avg", linewidth=2.5)
        # axs[key].legend()
        axs[key].set_title(network, fontsize=12)
        # axs[key].ylabel('Loss')
    axs[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12)
    # plt.plot(loss, label="Loss")
    # plt.plot(uniform_filter1d(loss, size = 100), label="Loss moving avg", linewidth=2.5)
    # plt.ylabel('Loss')
    # plt.legend()
    plt.show()



    # hist_model = np.array(output_test) - np.array(label_test)

    fig, axs = plt.subplots(1, len(networks), figsize=figsize)
    fig.suptitle('Histogram of error (NN_geuss - label)', fontsize=18)
    for key, network in enumerate(networks):
        hist_model = np.array(test_dic[network]['out_list']) - np.array(test_dic[network]['label_list'])
        hist_ksg = np.reshape(ksg_list, (-1, 1)) - np.array(test_dic[network]['label_list'])
        _, bins, _ = axs[key].hist(hist_model, bins=50, alpha=0.5, label="Model")
        _ = axs[key].hist(hist_ksg, bins=bins, alpha=0.5, label="KSG")
        axs[key].set_title(network, fontsize=12)
        # axs[key].legend()
    axs[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12)
    # _, bins, _ = plt.hist(hist_model, bins=50, alpha=0.5, label="Model")
    # _ = plt.hist(hist_ksg, bins=bins, alpha=0.5, label="KSG")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'../graphs/hist_{network}_lr_{learning_rate}_ep_{epochs}_bs_{batch_size}.pdf')
    plt.show()
    # print(f"Avg err: {acc:.5f}, Mean err: {np.mean(hist_model):.5f}, Var error: {np.var(hist_model):.5f}")

    # acc_min = min(acc_list)
    # plt.plot(acc_list)
    # plt.title("Model error after each epoch saved")
    # plt.ylabel('Loss')
    # plt.show()
    # print(f"Min acc: {acc_min}, final acc: {acc_list[-1]}")

    fig, axs = plt.subplots(1, len(networks), figsize=figsize)
    fig.suptitle('Validation vs Training', fontsize=18)
    for key, network in enumerate(networks):
        ksgx = np.zeros(len(train_test_dic[network]['loss_list']))+avg_ksg
        axs[key].plot(ksgx, label="KSG")
        axs[key].plot(uniform_filter1d(train_test_dic[network]['loss_list'], size = 10), label = 'Training')
        axs[key].plot(uniform_filter1d(train_test_dic[network]['loss2_list'], size = 10), label="Validation")
        # axs[key].ylabel('Loss')
        axs[key].set_title(network, fontsize=12)
        # axs[key].legend()
    axs[1].legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", fancybox=True, shadow=True, ncol=3, fontsize=12)
    # plt.plot(ksgx, label='KSG')
    # plt.plot(uniform_filter1d(loss_list, size = 10), label = 'Training')
    # plt.plot(uniform_filter1d(loss2_list, size = 10), label = 'Validation')
    # plt.ylabel('Loss')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'../graphs/loss_{network}_lr_{learning_rate}_ep_{epochs}_bs_{batch_size}.pdf')
    plt.show()

    # ksgx = np.zeros(len(loss_list))+avg_ksg
    # plt.plot(ksgx, label='KSG')
    # plt.plot(loss_list, label = 'Training')
    # plt.plot(loss2_list, label = 'Validation')
    # plt.ylabel('Loss')
    # plt.title('Validation vs Training')
    # plt.legend()
    # plt.show()



# %%
