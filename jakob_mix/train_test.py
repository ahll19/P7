import sys, os
sys.path.append(os.getcwd())
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
import numpy as np
import data_gen1
import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training loop
def train(train_loader, learning_rate, num_epoch, input_size, batch_size):
    model = models.Article_nn(input_size)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            sample = images.reshape(batch_size, -1).to(device)
            labels = labels.view(labels.shape[0], 1).to(device) # makes it a column vector

            # forward
            output = model(sample)
            loss = criterion(output, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch) % 100 == 0 and i % 10 == 0:
                print(
                    f'epoch {epoch} / {num_epoch-1}, step {i}/{n_total_steps-1} loss = {loss.item():.4f}')

    print(f"\n#################################\n# TEST DONE\n#################################\n")

    return model


def test(model, test_loader, batch_size):
    with torch.no_grad():
        out_list = []
        label_list = []
        n_samples = 0
        n_diff = 0
        for images, labels in test_loader:
            sample = images.reshape(batch_size, -1).to(device)
            labels = labels.view(labels.shape[0], 1).to(device)
            label_list += labels.tolist()

            outputs = model(sample) # trained model
            out_list += outputs.tolist()
            n_diff += torch.mean(torch.abs(outputs-labels))
            n_samples += 1

        acc = n_diff/n_samples
        print(f"accuracy = {acc}")

    return out_list, label_list


if __name__ == "__main__":
    path = "data/realisations=5000_xy_len=1000.npy"
    realisations = 100
    xy_len = 1000
    batch_size = 10
    data, label = data_gen1.load_data(path)

    data, label = data[:realisations, :xy_len, :], label[:realisations]

    # creating test/train data
    data_train, data_test, label_train, label_test = tts(data, label, test_size=0.1)

    train_data = models.NumbersDataset(data_train, label_train)
    test_data = models.NumbersDataset(data_test, label_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # train
    model_trained = train(train_loader, 0.5, realisations, xy_len, batch_size)

    # test
    output_test, label_test = test(model_trained, test_loader, batch_size)

    plt.plot(output_test, "*", label="Model output")
    plt.plot(label_test, "*", label="Label")
    plt.legend()
    plt.show()






