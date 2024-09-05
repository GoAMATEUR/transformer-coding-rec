import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ffn import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super(DecoderLayer, self).__init__()
        self.cfg = cfg

        self.self_attn = MultiHeadAttention(cfg)
        self.src_attn = MultiHeadAttention(cfg)
        self.ffn = FeedForwardNetwork(cfg)
        self.layer_norm1 = nn.LayerNorm(cfg.dim_embed)
        self.layer_norm2 = nn.LayerNorm(cfg.dim_embed)
        self.layer_norm3 = nn.LayerNorm(cfg.dim_embed)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = x + self.self_attn(x, x, x, tgt_mask)
        x = self.layer_norm1(x)
        x = x + self.src_attn(x, memory, memory, src_mask)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return self.layer_norm3(x)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.Decoder.num_layers)])

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x