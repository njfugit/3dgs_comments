import numpy as np
import torch
import torch.nn as nn
import math

#实现多头attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, dropout=0.1, bias=True):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else (d_model // num_heads)
        if head_dim is None:
            assert d_model % num_heads == 0, "d_model 必须能整除 num_heads（未显式指定 head_dim 时）"
        
        inner_dim = self.head_dim * self.num_heads
        
        self.q_proj = nn.Linear(d_model, inner_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=bias)
        self.out_proj = nn.Linear(inner_dim, d_model, bias=bias)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        计算缩放点积注意力 (与单头注意力实现完全相同)
        """
        #q,k,v:  (batch, num_heads, seq_len, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * self.scale


        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (torch.Tensor): shape (batch_size, seq_len_q, d_model)
            key (torch.Tensor): shape (batch_size, seq_len_k, d_model)
            value (torch.Tensor): shape (batch_size, seq_len_v, d_model)
            mask (torch.Tensor, optional): shape (batch_size, 1, seq_len_q, seq_len_k). Defaults to None.

        Returns:
            torch.Tensor: shape (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # 1) Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. 重塑 Q, K, V 以进行多头并行计算
        # (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算缩放点积注意力
        # context: (batch, num_heads, seq_len_q, head_dim)
        context, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 4. 拼接多头输出并进行最终的线性变换
        # context.transpose(1, 2) -> (batch, seq_len_q, num_heads, head_dim)
        # .contiguous().view(...) -> (batch, seq_len_q, inner_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        # out_proj(...) -> (batch, seq_len_q, d_model)
        output = self.out_proj(context)

        return output

if __name__ == '__main__':
    # Parameters
    d_model = 512  # Dimension of model
    num_heads = 8  # Number of attention heads
    head_dim = 96
    dropout = 0.1
    batch_size = 4
    seq_length = 10

    # Create a MultiHeadAttention instance
    multi_head_attention = MultiHeadAttention(d_model, num_heads, head_dim)

    # Create dummy input tensors
    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)

    # Create a dummy mask (e.g., for padding)
    # mask = torch.ones(batch_size, 1, seq_length, seq_length)
    # mask[:, :, :, 5:] = 0 # Mask out last 5 tokens

    # Forward pass
    output = multi_head_attention(query, key, value, mask=None)

    # Print output shape
    print("Input shape:", query.shape)
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, seq_length, d_model)
    print("Multi-head attention test passed!")
