import torch
import torch.nn as nn
from configs.base_config import BaseConfig

class PositionalEncoding(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(cfg.dropout)
        
        
        # self.dropout = nn.Dropout(p=dropout)

        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        # pe[:, 0::2] = torch.sin(position.float() * div_term)
        # pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)