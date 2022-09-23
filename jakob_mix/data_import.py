# %%
import sys, os

sys.path.append(os.getcwd())
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data_gen1

# import models

np.random.seed(69)
# torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('cpu')


# _test, mutual_information = data_gen_1.gen_list(3000, 10, 10, 1)
# x, y = _test[:, 0, :], _test[:, 1, :]

# plt.scatter(x, y, alpha=.1)
# plt.show()
# print(f"Mutual information is approx {mutual_information:.2f} \n")


def create_data(num_points, point_size=1000, min_var0=0.1, max_var0=10, min_var1=0.1, max_var1=10, dim=1):
    x0_var = np.random.uniform(min_var0, max_var0, num_points)
    x1_var = np.random.uniform(min_var1, max_var1, num_points)
    data = np.zeros([num_points, point_size, 2, dim])
    label = np.zeros([num_points])

    for i in range(num_points):
        data[i], label[i] = data_gen1.gen_list(
            point_size, x0_var[i], x1_var[i], dim)

    return data, label


# Test data import

class Article_nn(nn.Module):
    def __init__(self, input_size):
        super(Article_nn, self).__init__()
        self.l1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.sig(self.l2(out))
        return out


class NumbersDataset(Dataset):
    def __init__(self, num_points, point_size):
        self.samples, self.labels = create_data(num_points, point_size)
        self.samples = torch.from_numpy(self.samples).to(torch.float32)
        self.labels = torch.from_numpy(self.labels).to(torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #sample = {'feature': self.samples[idx], 'label': self.labels[idx]}
        return self.samples, self.labels


def train(train_loader, learning_rate, num_epoch, input_size):
    model = Article_nn(input_size)
    # print(train_loader)
    # loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            sample = images.reshape(-1, 2).to(device)
            #labels = torch.ones(
            #    (len(sample), 1)) * labels.item()  # reshaping because labels needs to be same size as output
            labels = labels.to(device)

            print(images.shape, labels, labels.shape)
            # forward
            output = model(sample)
            loss = criterion(output, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i) % 1 == 0:
                print(
                    f'epoch {epoch} / {num_epoch}, step {i}/{n_total_steps} loss = {loss.item():.4f}')


if __name__ == '__main__':
    input_size = 100
    train_data = NumbersDataset(20, input_size)
    dataloader = DataLoader(train_data, batch_size=5, shuffle=True)
    # test = next(iter(dataloader))
    train(dataloader, 0.5, 20, input_size)
# %%
