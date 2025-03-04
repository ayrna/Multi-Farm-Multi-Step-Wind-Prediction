import torch
import torch.nn as nn
from torch.nn.functional import softplus, elu


class Elusplus2L(nn.Module):
    def __init__(self):
        super(Elusplus2L, self).__init__()
        self.initialized = False

    def _initialize(self, device):
        self._lambda = nn.Parameter(torch.tensor(0.5, device=device), requires_grad=True)
        self._alpha = nn.Parameter(torch.rand(1, device=device), requires_grad=True)
        self.initialized = True

    def __call__(self, x):
        if not self.initialized:
            raise ValueError("The module has not been initialized yet.")

        output = (elu(x) * torch.clip(self._lambda, 0, 1)) + (
            (softplus(x) - self._alpha) * (1 - torch.clip(self._lambda, 0, 1))
        )
        return output
