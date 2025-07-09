import torch
import torch.optim as optim
import torch.nn as nn

LOSSES = {
    "l1": torch.nn.L1Loss(),
    "l2": torch.nn.MSELoss(),
}

OPTIMIZERS = {
    "adam": optim.Adam
}

SOLVERS = [
    "dopri5",
    "adams",
    "euler"
]

LOSSES = {
    "l1": torch.nn.L1Loss(),
    "l2": torch.nn.MSELoss(),
}

NONLINEARITIES = {
    "relu": nn.ReLU(),
}
