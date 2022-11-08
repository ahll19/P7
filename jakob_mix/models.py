import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable


class sin_fun(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(input) / input * torch.abs(torch.sin(input))


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.fun_act = sin_fun()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.fun_act(out)
        out = self.l2(out)  # NOT USE SOFTMAX SINCE WE USE nn.CrossEntropyLoss
        return out


class CNN(nn.Module):
    def __init__(self, input_size, batch_size):
        super(CNN, self).__init__()
        self.out_channel = 6
        self.kernel_size = 2
        self.magic_nr = (input_size*self.out_channel)-(self.kernel_size*3)

        self.conv1 = nn.Conv2d(batch_size, self.out_channel, self.kernel_size)
        self.pool = nn.MaxPool2d(1, 1)
        self.fc1 = nn.Linear(self.magic_nr, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, batch_size)
        self.batch = batch_size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.magic_nr)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(self.batch, -1)


class FNN(nn.Module):
    def __init__(self, input_size):
        super(FNN, self).__init__()
        self.l1 = nn.Linear(input_size*2, 125)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(125, 550)
        self.l3 = nn.Linear(550, 1)

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)
        return out


class RNN(nn.Module):
    '''
    Inspiration from:
    https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch/notebook

    Module used to create a recurrent neural network. 
    '''
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, batch_size):
        super(RNN, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.input_dim = input_dim
        self.batch = batch_size

        # RNN
        self.rnn = nn.RNN(2, hidden_dim, layer_dim,
                          batch_first=True, nonlinearity='relu')

        # fc: fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, self.batch, self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)

        # Since RNN has different structure it only uses the last
        # entrance of the sequence length for the output hence [:, -1, :]
        out = self.fc(out[:, -1, :])
        return out


class Article_nn2(nn.Module):
    def __init__(self, input_size):
        super(Article_nn2, self).__init__()
        self.l1 = nn.Linear(input_size*2, 64)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.leaky(self.l1(x))
        out = self.leaky(self.l2(out))
        out = self.leaky(self.l3(out))
        out = self.leaky(self.l4(out))
        out = self.l5(out)
        return out


class NumbersDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.from_numpy(samples).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]