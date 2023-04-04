import torch
from torch import nn


class CrossNetVector(nn.Module):
    def __init__(self, d_in, n_cross=2):
        super().__init__()
        self.n_cross = n_cross
        self.linears = nn.ModuleList([nn.Linear(d_in, 1, bias=False) for i in range(self.n_cross)])
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_in)) for i in range(self.n_cross)])

    def forward(self, x):
        x0 = x
        xi = x
        for i in range(self.n_cross):
            xi = x0 * self.linears[i](xi) + self.biases[i] + xi
        return xi