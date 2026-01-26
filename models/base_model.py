import torch
from torch.nn import Module
from torch.nn import Linear as Lin

class BaseModel(Module):
    def __init__(self, seed = 42, device = 'cpu'):
        self.device = device
        self.seed = seed

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def init_weights(layer):
    if isinstance(layer, Lin):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias)
