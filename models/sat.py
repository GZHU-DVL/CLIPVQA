import torch
from torch import nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import sys
sys.path.append("../")
from clip.model import QuickGELU,LayerNorm


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SpatiotemporalAggregationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers=1,):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        # if embed_dim==512 :
        #     MFC=13
        # else:
        #     MFC=25
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])
        # scale = embed_dim** -0.5
        # self.class_embedding = nn.Parameter( scale* torch.randn(embed_dim))
        self.apply(self._init_weights)
        self.alpha = nn.Parameter(torch.ones(embed_dim))
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        ori_x = x
        x_mos = x[:,0:self.T,:]
        x_mos = x_mos + self.positional_embedding
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),x], dim=1)

        # x = x.permute(1, 0, 2)
        x_mos= self.resblocks(x_mos)
        x_fusion = x[:,self.T:,:]
        x=torch.cat([x_fusion*self.alpha ,x_mos],1)      # x = x.permute(1, 0, 2)
        # x = x.type(ori_x.dtype)+ ori_x
        # x = self.head(x.permute(0, 2, 1))     # x.view(x.shape[0],x.shape[1])
        return x.mean(dim=1, keepdim=False)    # x.mean(dim=1, keepdim=False)