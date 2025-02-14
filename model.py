import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_node, hidden_N, hidden_L, output_N=3):
        super(Net, self).__init__()
        self.hidden_N = hidden_N
        self.hidden_L = hidden_L
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_node, hidden_N))
        for _ in range(hidden_L):
            self.layers.append(nn.Linear(hidden_N, hidden_N))

        self.dropouts = nn.ModuleList()
        for _ in range(hidden_L):
            self.dropouts.append(nn.Dropout(0.3))

        self.batches = nn.ModuleList()
        for _ in range(hidden_L):
            self.batches.append(nn.BatchNorm1d(hidden_N))

        self.output = nn.Linear(hidden_N, output_N)
    def forward(self, x):
        z = x
        for layer, dropout, batch_norm in zip(self.layers, self.dropouts, self.batches):
            z = layer(z)
            z = batch_norm(z)
            z = F.leaky_relu(z, 0.05)
            z = dropout(z)

        z = torch.sigmoid(self.output(z))
        return z