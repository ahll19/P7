from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np

def load_data(file_path):
    """
    Takes the path to where the .npy file is located.

    :param file_path: The path with the path to the folder where the
    data is saved.
    :return: data, labels

    Example using savedata/loaddata:
    path = "test"
    save_data(path, data, labels) # saves data as .npy file
    data, labels = load_data(path) # loads .npy file
    """
    if file_path[-4:] != '.npy':
        file_path += '.npy'

    with open(file_path, 'rb') as f:
        data = np.load(f)
        label = np.load(f)

    return data, label

class NumbersDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.from_numpy(samples).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class Network(nn.Module):
    def __init__(self, hidden_layer_dims: list[int], input_size: int, device: torch.device) -> None:
        super(Network, self).__init__()
        self.device = device
        
        self.relu = nn.ReLU().to(device)
        self.l1 = nn.Linear(input_size, hidden_layer_dims[0]).to(device)
        self.ln = nn.Linear(hidden_layer_dims[-1], 2).to(device)
        self.ls = [self.l1]
        for i in range(len(hidden_layer_dims) - 1):
            self.ls.append(
                nn.Linear(hidden_layer_dims[i], hidden_layer_dims[i+1]).to(device)
            )
        self.ls.append(self.ln)
        
    def forward(self, x: np.array) -> np.array:
        out = self.relu(self.ls[0](x))
        for i, l in enumerate(self.ls[1:-1]):
            out = self.relu(l(out))
        out = self.ls[-1](out)
        
        return out


def add_noise(data, SNR):
    # Calculate the noise power
    P_noise = np.mean(data**2, axis=1)/np.full((data.shape[0], data.shape[2]), SNR)

    # Initialize the noise array
    noise = np.zeros(data.shape)

    # Generate noise for each sample in the data
    mu = np.zeros((data.shape[1], 1))
    for i in range(len(data)):
        Xi_noise = np.random.normal(mu.copy(), P_noise[i][0], (data.shape[1], 1))
        Yi_noise = np.random.normal(mu.copy(), P_noise[i][1], (data.shape[1], 1))

        # Concatenate the noise for the X and Y dimensions
        noise[i] = np.concatenate([Xi_noise, Yi_noise], axis=1)

    # Add the noise to the data
    noisy_data = data + noise

    # Return the noisy data
    return noisy_data
