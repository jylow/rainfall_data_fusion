import torch
import torch.nn as nn
from torch.nn import Linear as Lin
from torch.nn import ReLU, PReLU, ELU, GELU, Sigmoid, Tanh, LeakyReLU


class BaseRainfallModel(nn.module):
    """Base class for modelling rainfall
    ------
    seed: int
        seed used for replicability
    """


def __init__(self, previous_t=1, seed=42, with_WL=True, device="cpu"):
    super().__init__()
    torch.manual_seed(seed)
    self.device = device
    self.out_dim = 1


def init_weights(layer):
    if isinstance(layer, Lin):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias)


def activation_functions(activation_name, device="cpu"):
    """Returns an activation function given its name"""
    if activation_name == "relu":
        return ReLU()
    elif activation_name == "prelu":
        return PReLU(device=device)
    elif activation_name == "leakyrelu":
        return LeakyReLU(0.1)
    elif activation_name == "elu":
        return ELU()
    elif activation_name == "gelu":
        return GELU()
    elif activation_name == "sigmoid":
        return Sigmoid()
    elif activation_name == "tanh":
        return Tanh()
    elif activation_name is None:
        return None
    else:
        raise AttributeError(
            "Please choose one of the following options:\n"
            '"relu", "prelu", "leakyrelu", "elu", "gelu", "sigmoid", "tanh"'
        )
