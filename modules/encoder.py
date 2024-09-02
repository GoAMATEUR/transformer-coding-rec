from attention import MultiHeadAttention
from ffn import FeedForwardNetwork
import torch
from torch import nn
from configs.base_config import BaseConfig

class EncoderLayer(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(EncoderLayer, self).__init__()
        self.cfg = cfg
        
        
        self.self_attn = MultiHeadAttention(cfg)
        self.ffn = FeedForwardNetwork(cfg)
        self.layer_norm1 = nn.LayerNorm(cfg.dim_embed)
        self.layer_norm2 = nn.LayerNorm(cfg.dim_embed)

    def forward(self, x, mask=None):
        
        x = x + self.self_attn(x, x, x, mask)
        x = self.layer_norm1(x)
        x = x + self.ffn(x)
        return self.layer_norm2(x)


class Encoder(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.Encoder.num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x