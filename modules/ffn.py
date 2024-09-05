import numpy as np

import torch.nn as nn
import torch
from configs.base_config import BaseConfig

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_dim = cfg.dim_ffn
        
        self.linear1 = nn.Linear(cfg.dim_embed, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, cfg.dim_embed)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        
        

    def forward(self, x):
        """Perform positional encoding on input tensor x

        Args:
            x (torch.Tensor): size (batch_size, seq_len, dim_embed)
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
        