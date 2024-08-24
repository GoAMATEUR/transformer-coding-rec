import numpy as np

import torch.nn as nn
import torch
from configs.base_config import BaseConfig

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_dim = cfg.dim_ffn
        
        self.positional_encoding = torch.zeros(cfg.max_len, cfg.dim_embed)
        
        # self.positional_encoding.requires_grad = False
        self.register_buffer("pe", self.positional_encoding)
        

    def forward(self, x):
        """Perform positional encoding on input tensor x

        Args:
            x (torch.Tensor): size (batch_size, seq_len, dim_embed)
        """
        self.pe 
        return 
        