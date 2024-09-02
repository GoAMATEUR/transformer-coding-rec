import torch
from torch import nn
from configs.base_config import BaseConfig


class PositionalEncoding(nn.Module):
    def __init__(self, cfg: BaseConfig) -> None:
        super(PositionalEncoding, self).__init__()
        self.cfg = cfg
        self.positional_encoding = torch.zeros(cfg.max_len, cfg.dim_embed)
        self.register_buffer("pe", self.positional_encoding)

    def forward(self, x):
        return
