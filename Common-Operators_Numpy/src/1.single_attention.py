import numpy as np
import torch
import torch.nn as nn

#实现单头attention
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, head_dim,  dropout=0.1, bias=True):
        """
        初始化函数
        参数:
            d_model (int): 输入词嵌入的维度 (embed_dim)
            head_dim (int): 单个注意力头的维度 (d_k or d_v)
        """
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, head_dim, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5
        #根据需求增加
        self.out_proj = nn.Linear(head_dim, d_model, bias=bias)

    def forward(self, query, key, value, mask=None):
        # 1. 线性投影 Q, K, V
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, head_dim)
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 2. 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores = torch.bmm(Q, K.transpose(-2, -1)) * self.scale
        # scores = torch.einsum('bik,bjk->bij', Q, K) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = self.softmax(scores) 
        attention = self.dropout(attention)
        out = torch.matmul(attention, V)
        # out = torch.bmm(attention, V)
        # out = torch.einsum('bij,bjk->bik', attention, V)
        return out
    
# 实现1: 自注意力
class SelfAttention(nn.Module):
    def __init__(self, d_model, head_dim, dropout=0.1, bias=True):
        super().__init__()
        self.attention = SingleHeadAttention(d_model, head_dim, dropout, bias)

    def forward(self, x, mask=None):
        # 在自注意力中, query, key, 和 value 都是同一个输入 x
        return self.attention(x, x, x, mask)

# 实现2: 交叉注意力
class CrossAttention(nn.Module):
    def __init__(self, d_model, head_dim, dropout=0.1, bias=True):
        super().__init__()
        self.attention = SingleHeadAttention(d_model, head_dim, dropout, bias)

    def forward(self, query, key_value, key_padding_mask=None):
        # 在交叉注意力中, query 来自一个源, key 和 value 来自另一个源 (key_value)
        # key_padding_mask 用于屏蔽 key_value 中的填充部分
        return self.attention(query, key_value, key_value, mask=key_padding_mask)

# --- 示例用法 ---
if __name__ == '__main__':
    batch_size = 2
    d_model = 512
    head_dim = 64

    # --- 自注意力示例 ---
    print("--- Self-Attention Example ---")
    seq_len_self = 10
    self_attention = SelfAttention(d_model, head_dim)
    
    # 创建一个随机输入张量
    input_self = torch.randn(batch_size, seq_len_self, d_model)
    
    # (可选) 创建一个掩码来忽略序列中的某些部分
    # 这里我们创建一个未来帧的掩码 (causal mask)
    self_mask = torch.tril(torch.ones(seq_len_self, seq_len_self)).unsqueeze(0)
    
    print("Input shape:", input_self.shape)
    print("Mask shape:", self_mask.shape)
    
    output_self = self_attention(input_self, mask=self_mask)
    print("Output shape:", output_self.shape)
    print("\n" + "="*30 + "\n")

    # --- 交叉注意力示例 ---
    print("--- Cross-Attention with Key Padding Mask Example ---")
    seq_len_query = 5
    seq_len_kv = 8
    cross_attention = CrossAttention(d_model, head_dim)

    # 创建随机的 query 和 key/value 张量
    query_input = torch.randn(batch_size, seq_len_query, d_model)
    kv_input = torch.randn(batch_size, seq_len_kv, d_model)

    # 创建 key_padding_mask
    # 假设在第一个序列中最后2个是padding, 第二个序列中最后3个是padding
    key_padding_mask = torch.ones(batch_size, seq_len_kv)
    key_padding_mask[0, -2:] = 0
    key_padding_mask[1, -3:] = 0
    
    # 调整mask的维度以进行广播: (batch_size, seq_len_kv) -> (batch_size, 1, seq_len_kv)
    key_padding_mask_expanded = key_padding_mask.unsqueeze(1)

    print("Query shape:", query_input.shape)
    print("Key/Value shape:", kv_input.shape)
    print("Key padding mask shape:", key_padding_mask.shape)
    print("Expanded key padding mask shape:", key_padding_mask_expanded.shape)

    output_cross = cross_attention(query_input, kv_input, key_padding_mask=key_padding_mask_expanded)
    print("Output shape:", output_cross.shape)