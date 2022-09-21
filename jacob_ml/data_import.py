# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data_gen1
import models

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
class NumbersDataset(Dataset):
    def __init__(self, num_points, point_size):
        self.samples, self.labels = create_data(num_points, point_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train(train_loader, learning_rate, num_epoch, input_size):
    model = models.Article_nn(input_size)
    print(train_loader)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epoch):
        for i, (sample, labels) in enumerate(train_loader):
            # 100, 1, 28, 28
            # 100, 784
            sample = sample.to(device)
            labels = labels.to(device)

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
    dataset = NumbersDataset(10, input_size)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    # test = next(iter(dataloader))
    train(dataloader, 0.5, 20, input_size)
# %%
