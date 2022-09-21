import torch
import torch.nn as nn

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


class Article_nn(nn.Module):
    def __init__(self, input_size):
        super(Article_nn, self).__init__()
        self.l1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.sig(out)
        return out
