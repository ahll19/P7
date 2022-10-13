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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training loop
def train(train_loader, learning_rate, num_epoch, input_size, batch_size, network='FNN'):
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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    loss_list = []
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
            loss = criterion(output, labels)#/labels.shape[0]
            loss_list.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            break
            if (epoch) % 100 == 0 and i % 10 == 0:
                print(
                    f'epoch {epoch} / {num_epoch-1}, step {i}/{n_total_steps-1} loss = {loss.item():.8f}')
        break

    print(f"\n#################################\n# TRAINING DONE\n#################################\n")

    return model, loss_list


def test(model, test_loader, batch_size, network='FNN'):
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

    return out_list, label_list, acc


if __name__ == "__main__":
    path = "data/realisations=5000_xy_len=1000.npy"
    realisations = 500
    xy_len = 50
    batch_size = 10
    learning_rate = 0.001
    epochs = 250
    network = 'RNN'

    data, label = data_gen1.load_data(path)

    data, label = data[:realisations, :xy_len, :], label[:realisations]

    # creating test/train data
    data_train, data_test, label_train, label_test = tts(data, label, test_size=0.1)

    train_data = models.NumbersDataset(data_train, label_train)
    test_data = models.NumbersDataset(data_test, label_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # train
    model_trained, loss = train(train_loader, learning_rate, epochs,
                                xy_len, batch_size, network=network)

    # test
    output_test, label_test, acc = test(model_trained, test_loader, batch_size, network=network)


    # Calculating ksg while still in numpy:
    ksg_list = np.zeros(len(data_test))
    for i in range(len(data_test)):
        ksg_list[i] = ksg.predict(data_test[i][:, 0], data_test[i][:, 1])

    avg_ksg = np.sum(ksg_list)/len(ksg_list)

    print(f"Avg error model = {acc:.5f}")
    print(f"Avg error KSG   = {avg_ksg:.5f}")

    plt.plot(output_test, "*", label="Model output")
    plt.plot(label_test, "*", label="Label")
    plt.plot(np.reshape(ksg_list, (-1, 1)), '*', label="KSG")
    plt.legend()
    plt.show()

    plt.plot(loss, label="Loss")
    plt.plot(uniform_filter1d(loss, size = 100), label="Loss moving avg", linewidth=2.5)
    plt.legend()
    plt.show()

    hist_model = np.array(output_test) - np.array(label_test)
    hist_ksg = np.reshape(ksg_list, (-1, 1)) - np.array(label_test)

    _, bins, _ = plt.hist(hist_model, bins=50, alpha=0.5, label="Model")
    _ = plt.hist(hist_ksg, bins=bins, alpha=0.5, label="KSG")
    plt.legend()
    plt.show()






