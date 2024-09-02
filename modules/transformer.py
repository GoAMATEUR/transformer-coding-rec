import torch
from torch import nn
from configs.base_config import BaseConfig


class Transformer(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(Transformer, self).__init__()
        self.cfg = cfg
        
        self.positional_encoding = None

    def forward(self, x):
        pass

    