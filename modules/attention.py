import torch
import torch.nn as nn
import numpy as np

from configs.base_config import BaseConfig


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        # self.temperature = np.power(cfg.dim_head, 0.5)

    def forward(self, Q, K, V, mask=None, temperature=1.0):
        """_summary_

        Args:
            Q (_type_): (batch_size, num_heads, len_seq, dim_k)
            K (_type_): (batch_size, num_heads, len_seq, dim_k)
            V (_type_): (batch_size, num_heads, len_seq, dim_v) 
            
            mask (_type_, optional): _description_. Defaults to None.
        """
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9) # large negative value so that e^x -> 0
        attention = torch.softmax(scores / temperature, dim=-1) # (batch_size, num_heads, len_seq, len_seq) # softmax along tokens in the sequence
        output = torch.matmul(attention, V) # (batch_size, num_heads, len_seq, dim_v)
        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = cfg.SelfAtten.num_heads
        
        self.dim_k = cfg.SelfAtten.dim_k
        self.dim_q = cfg.SelfAtten.dim_k
        self.dim_v = cfg.SelfAtten.dim_v
        
        self.dim_model = cfg.dim_embed

        self.linear_q = nn.Linear(self.dim_model, self.num_heads * self.dim_q, bias=False)
        self.linear_k = nn.Linear(self.dim_model, self.num_heads * self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_model, self.num_heads * self.dim_v, bias=False)
        
        self.linear_output = nn.Linear(self.num_heads * self.dim_v, self.dim_model, bias=False)
        
        self.scaled_dot_product_attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        """_summary_

        Args:
            Q (_type_): (batch_size, len_seq, dim_model)
            K (_type_): (batch_size, len_seq, dim_model)
            V (_type_): (batch_size, len_seq, dim_model)

            mask (_type_, optional): _description_. Defaults to None.
        """
        # batch_size = Q.size(0)
        Q = self.linear_q(Q).view(Q.size(0), -1, self.num_heads, self.dim_q).transpose(1, 2) # (batch_size, num_heads, len_seq, dim_q)
        K = self.linear_k(K).view(K.size(0), -1, self.num_heads, self.dim_k).transpose(1, 2) # (batch_size, num_heads, len_seq, dim_k)
        V = self.linear_v(V).view(V.size(0), -1, self.num_heads, self.dim_v).transpose(1, 2) # (batch_size, num_heads, len_seq, dim_v)
        
        output, _ = self.scaled_dot_product_attention(Q, K, V, mask=mask) # (batch_size, num_heads, len_seq, dim_v)
        
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.num_heads * self.dim_v) # aggragate heads
        output = self.linear_output(output)
        return output


if __name__ == "__main__":
    a = torch.zeros(1, 2, 3)
    
    a = a.transpose(-1, -2)
    print(a.size(0))
        