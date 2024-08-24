import torch
import torch.nn as nn
import numpy as np

from configs.base_config import BaseConfig


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        
        # self.temperature = np.power(cfg.dim_head, 0.5)

    def forward(self, q, k, v, mask=None):
        """_summary_

        Args:
            q (_type_): (batch_size, num_heads, num_q, dim_q)
            k (_type_): (batch_size, num_heads, num_k, dim_k)
            v (_type_): (batch_size, num_heads, num_v, dim_v) # num_k == num_v
            
            mask (_type_, optional): _description_. Defaults to None.
        """
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9) # large negative value so that e^x -> 0
        attention = torch.softmax(scores, dim=-1) # (batch_size, num_heads, num_q, num_k) # softmax along the keys
        output = torch.matmul(attention, v) # (batch_size, num_heads, num_q, dim_v)
        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = cfg.dim_e

if __name__ == "__main__":
    a = torch.zeros(1, 2, 3)
    
    a = a.transpose(-1, -2)
    print(a.size(0))
        