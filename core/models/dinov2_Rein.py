# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 09:35:36 2025

@author: 15642
"""

# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    并联 LoRA：y = Wx + scale * B(Ax)；只训练 A/B，原 Linear 冻结。
    """
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.r = int(r)
        self.scaling = (alpha / r) if r and r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        if self.r > 0:
            self.A = nn.Linear(base_linear.in_features, r, bias=False)
            self.B = nn.Linear(r, base_linear.out_features, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        else:
            self.A = None; self.B = None
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        y = self.base(x)
        if self.r and self.r > 0:
            y = y + self.scaling * self.B(self.A(self.dropout(x)))
        return y


def apply_lora_to_dinov2(vit: nn.Module, r=8, alpha=16, dropout=0.0,
                         targets=("qkv", "proj", "fc1", "fc2")) -> nn.Module:
    """
    在 ViT blocks 内的线性层上注入并联 LoRA。
    典型命名：.attn.qkv / .attn.proj / .mlp.fc1 / .mlp.fc2
    """
    for blk in getattr(vit, "blocks", []):
        for name, mod in list(blk.named_modules()):
            if isinstance(mod, nn.Linear):
                last = name.split(".")[-1]
                if last in targets:
                    parent = blk
                    for p in name.split(".")[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, last, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
    return vit


class ReinModule(nn.Module):
    def __init__(self, num_layers: int, c: int, m: int = 64, r: int = 8, cq: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.c = c
        self.m = m
        self.r = r
        self.cq = cq

        # 低秩 tokens：每层独立 Ai, Bi
        self.As = nn.ParameterList([nn.Parameter(torch.empty(m, r)) for _ in range(num_layers)])
        self.Bs = nn.ParameterList([nn.Parameter(torch.empty(r, c)) for _ in range(num_layers)])
        for A, B in zip(self.As, self.Bs):
            nn.init.uniform_(A, -0.02, 0.02)
            nn.init.uniform_(B, -0.02, 0.02)

        self.WT = nn.Linear(c, c, bias=True)     # 对 tokens 做空间变换（用于 Δf̄ 的右项）
        self.Wf = nn.Linear(c, c, bias=True)     # Δf 最终线性映射
        self.WQi = nn.Linear(c, cq, bias=True)   # 每层生成 Qi
        self.WQ  = nn.Linear(3 * cq, cq, bias=True)  # [Qmax, Qavg, Q_last] → Q

        nn.init.zeros_(self.Wf.weight)
        nn.init.zeros_(self.Wf.bias)

        self._cached_T: List[torch.Tensor] = [None for _ in range(num_layers)]

    def make_tokens(self, i: int) -> torch.Tensor:
        Ti = self.As[i] @ self.Bs[i]       # [m, c]
        self._cached_T[i] = Ti
        return Ti

    def refine(self, i: int, fi: torch.Tensor) -> torch.Tensor:
        """
        输入:
            fi: [B, N, C]，进入第 i 个 block 的 token（patch tokens）
        输出:
            fi + Δfi  （Eq.(2) 的加法作用在 block 之前）
        """
        B, N, C = fi.shape
        Ti = self.make_tokens(i)                    # [m, c]
        # S_i = softmax(fi @ Ti^T / sqrt(c))  over last dim=m
        Si = torch.matmul(fi, Ti.t()) / math.sqrt(C)  # [B, N, m]
        Si = F.softmax(Si, dim=-1)

        if self.m > 1:
            Si_sel = Si[..., 1:]                    # [B, N, m-1]
            Ti_sel = Ti[1:, :]                      # [m-1, c]
        else:
            return fi

        # 右项：Ti_sel @ W_T  → [m-1, c]
        Ti_mapped = self.WT(Ti_sel)                 # [m-1, c]

        # Δf̄ = Si_sel @ Ti_mapped
        delta_bar = torch.matmul(Si_sel, Ti_mapped) # [B, N, c]

        # Δf = (Δf̄ + fi) @ W_f   （Eq.(7)）
        delta = self.Wf(delta_bar + fi)             # [B, N, c]

        return fi + delta                            # [B, N, c]

    @torch.no_grad()
    def _stack_Q_layers(self) -> torch.Tensor:
        """
        由缓存的各层 Ti 生成 Qi，并做 max / avg / last 汇聚，得到最终 Q: [m, cq]
        """
        Ts = [T if T is not None else (A @ B) for T, A, B in zip(self._cached_T, self.As, self.Bs)]
        Q_layers = [self.WQi(T) for T in Ts]            # list of [m, cq]
        Q_stack = torch.stack(Q_layers, dim=0)          # [L, m, cq]
        Qmax = Q_stack.max(dim=0).values                # [m, cq]
        Qavg = Q_stack.mean(dim=0)                      # [m, cq]
        Qlast = Q_layers[-1]                            # [m, cq]
        Q = self.WQ(torch.cat([Qmax, Qavg, Qlast], dim=-1))  # [m, cq]
        return Q


class Mask2FormerLite(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, cq: int = 256, num_feats: Optional[int] = None):
        super().__init__()
        self.cq = cq
        self.num_classes = num_classes

        # 不在这里固定数量，留到 forward 再按 feats 动态构建
        self.proj: Optional[nn.ModuleList] = None
        self.proj_bn: Optional[nn.ModuleList] = None

        self.q2kernel = nn.Linear(cq, cq, bias=False)
        self.q2class  = nn.Linear(cq, num_classes, bias=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(cq, cq, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cq),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, feats: List[torch.Tensor], Q: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        n = len(feats)
        assert n >= 1, "decode head expects at least one feature map"
    
        if self.proj is None or self.proj_bn is None:
            self.proj    = nn.ModuleList([nn.Conv2d(feats[0].shape[1], self.cq, kernel_size=1, bias=False) for _ in range(n)])
            self.proj_bn = nn.ModuleList([nn.BatchNorm2d(self.cq) for _ in range(n)])
            self.proj.to(feats[0].device)
            self.proj_bn.to(feats[0].device)
        elif len(self.proj) != n:
            if len(self.proj) > n:
                self.proj    = nn.ModuleList(list(self.proj)[:n])
                self.proj_bn = nn.ModuleList(list(self.proj_bn)[:n])
            else:
                add = n - len(self.proj)
                for _ in range(add):
                    self.proj.append(nn.Conv2d(feats[0].shape[1], self.cq, kernel_size=1, bias=False).to(feats[0].device))
                    self.proj_bn.append(nn.BatchNorm2d(self.cq).to(feats[0].device))
    
        ref_h, ref_w = feats[0].shape[-2:]
        proj_feats = []
        for x, conv, bn in zip(feats, self.proj, self.proj_bn):
            if conv.in_channels != x.shape[1]:
                idx = self.proj.index(conv)
                new_conv = nn.Conv2d(x.shape[1], self.cq, kernel_size=1, bias=False).to(x.device)
                self.proj[idx] = new_conv
                conv = new_conv
            y = bn(conv(x))
            if y.shape[-2:] != (ref_h, ref_w):
                y = F.interpolate(y, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            proj_feats.append(y)
    
        P = self.fuse(torch.stack(proj_feats, dim=0).sum(dim=0))  # [B,cq,ref_h,ref_w]
    
        kernels = self.q2kernel(Q).unsqueeze(-1).unsqueeze(-1)   # [m,cq,1,1]
        B = feats[0].shape[0]
        P_grouped = P.view(B, self.cq, ref_h, ref_w)
    
        mask_logits = []
        for j in range(Q.shape[0]):  # m
            w_j = kernels[j].to(P_grouped.device)  # [cq,1,1]  (注意：上面是 [m,cq,1,1]，这里索引后是 [cq,1,1])
            w_j = w_j.unsqueeze(0)  # [1,cq,1,1]
            y = F.conv2d(P_grouped, w_j, bias=None)
            mask_logits.append(y.squeeze(1))
        mask_logits = torch.stack(mask_logits, dim=1)  # [B,m,H,W]
    
        q_cls = self.q2class(Q)                        # [m,K]
        tau = 1.0  # 温度，可调 0.7~1.5
        attn = F.softmax(mask_logits / tau, dim=1)         # [B,m,H,W]
        sem_logits = torch.einsum('bmhw,mk->bkhw', attn, q_cls)

    
        if (ref_h, ref_w) != out_hw:
            sem_logits = F.interpolate(sem_logits, size=out_hw, mode='bilinear', align_corners=False)
        return sem_logits



class DINOv2ReinSeg(nn.Module):
    """
    DINOv2 + Rein + Query-based 解码头（Mask2Former-Lite）

    关键点：
      - 在每个 Transformer Block 前，通过 forward_pre_hook 注入 Δfi（fi 由该层输入 tokens 给出）
      - 训练时默认冻结 backbone；仅训练 Rein 与 decode head
      - 默认抽取第 7/11/15/23 层的 tokens 转成特征图，作为多层特征输入解码头
    """
    def __init__(self,
                 hub_dir='dinov2-main/',
                 model_name='dinov2_vitb14',
                 num_classes: int = 19,
                 m_tokens: int = 64,
                 rank_r: int = 8,
                 cq: int = 256,
                 layer_indices: Tuple[int, ...] = (6, 10, 14, 22), 
                 use_lora: bool = False,           
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.0,
                 freeze_backbone: bool = True):
        super().__init__()

        self.vit = torch.hub.load(hub_dir, model_name, source='local', pretrained=True)
        self.patch = getattr(self.vit, "patch_size", 14) or 14

        embed_dim = getattr(self.vit, "embed_dim", None)
        if embed_dim is None:
            embed_dim = 768
        se