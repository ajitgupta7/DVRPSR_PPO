import math
import torch.nn as nn
import torch.nn.functional as F


class GraphMultiHeadAttention(nn.Module):
    def __init__(self, num_head, query_size, key_size=None, value_size=None):
        super(GraphMultiHeadAttention, self).__init__()

        self.query_size = query_size
        self.num_head = num_head
        self.key_size = self.query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size

        self.key_dim = self.key_size//self.num_head
        self.value_dim = self.value_size//self.num_head
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.query_embed = nn.Linear(self.query_size, self.num_head*self.key_dim, bias=False)
        self.key_embed = nn.Linear(self.key_size, self.num_head*self.key_dim, bias=False)
        self.value_embed = nn.Linear(self.value_size, self.num_head*self.value_dim, bias=False)

        self.combine = nn.Linear(self.num_head*self.value_dim, self.value_size, bias=False)
        self.init_parameters()

    def init_parameters(self):
        norm_factor_value = 1 / math.sqrt(self.value_dim)
        norm_factor_combine = 1 / math.sqrt(self.value_size)

        nn.init.uniform_(self.query_embed.weight, -self.norm_factor, self.norm_factor)
        nn.init.uniform_(self.key_embed.weight, -self.norm_factor, self.norm_factor)
        nn.init.uniform_(self.value_embed.weight, -norm_factor_value, norm_factor_value)
        nn.init.uniform_(self.combine.weight, -norm_factor_combine, norm_factor_combine)

    def forward(self, queries, keys=None, values=None, edges=None, mask=None):
        batch_size, Q_size, _ = queries.size()
        Q = self.query_embed(queries).view(batch_size, Q_size, self.num_head, self.key_dim).permute(0, 2, 1, 3)

        if keys is None:
            KV_size = Q_size
            K = self.key_embed(queries).view(batch_size, KV_size, self.num_head, self.key_dim).permute(0, 2, 3, 1)
        else:
            KV_size = keys.size(-2)
            K = self.key_embed(keys).view(batch_size, KV_size, self.num_head, self.key_dim).permute(0, 2, 3, 1)

        if values is None:
            V = self.key_embed(queries).view(batch_size, KV_size, self.num_head, self.key_dim).permute(0, 2, 1, 3)
        else:
            V = self.key_embed(values).view(batch_size, KV_size, self.num_head, self.key_dim).permute(0, 2, 1, 3)

        attention = self.norm_factor * Q.matmul(K)

        if mask is not None:
            if mask.numel() * self.num_head == attention.numel():
                m = mask.view(-1, 1, Q_size, KV_size).expand_as(attention)
            else:
                m = mask.view(-1, 1, 1, KV_size).expand_as(attention)
            attention[m.bool()] = -float('inf')
        attention = F.softmax(attention, dim=-1)
        attention = attention.matmul(V).permute(0, 2, 1, 3).contiguous().view(-1, Q_size, self.num_head * self.value_dim)
        return self.combine(attention)