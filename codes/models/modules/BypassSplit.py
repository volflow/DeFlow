import torch
from torch import nn as nn


class BypassSplit(nn.Module):
    def __init__(self, n_split):
        super().__init__()
        self.n_split = n_split

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, bypass=None):
        if not reverse:
            return self.encode(bypass, input, logdet)
        else:
            return self.decode(bypass, input, logdet)

    def decode(self, bypass, input, logdet):
        assert bypass is None, "In flow direction the bypass is returned"
        z = input
        return {'z': z[:, :-self.n_split], 'bypass': z[:, -self.n_split:]}, logdet

    def encode(self, bypass, input, logdet):
        assert bypass is not None, "In reverse flow direction the bypass is concatenated"
        z = input
        return torch.cat([z, bypass], dim=1), logdet
