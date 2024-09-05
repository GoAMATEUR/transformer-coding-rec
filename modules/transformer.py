import torch
from torch import nn
from configs.base_config import BaseConfig
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(Transformer, self).__init__()
        self.cfg = cfg
        
        self.register_buffer('pe', self._get_positional_encoding())
        
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def _get_positional_encoding(self):
        positional_encoding = torch.zeros(self.cfg.max_len, self.cfg.dim_embed, requires_grad=False)
        # positional_encoding.requires_grad = False
        
        pos = torch.arange(self.cfg.max_len)
        order = torch.arange(self.cfg.dim_embed // 2)
        pos, order = torch.meshgrid(pos, order)

        positional_encoding[:, ::2] = torch.sin(pos / (10000 ** ((2 * order) / self.cfg.dim_embed)))
        positional_encoding[pos, 1::2] = torch.cos(pos / (10000 ** ((2 * order) / self.cfg.dim_embed)))
        return positional_encoding

    def forward(self, x):
        """

        Args:
            x (_type_): (batch_size, seq_len, dim_embed)
        """
        x = x + self.pe[:x.size(1), :]
        memory = self.encoder(x)
        output = self.decoder(x, memory)
        # TODO: Implement the forward pass
        
        return output
