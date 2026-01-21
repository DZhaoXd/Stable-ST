# -*- coding: utf-8 -*-

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
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

    for blk in getattr(vit, "blocks", []):
        for name, mod in list(blk.named_modules()):
            if isinstance(mod, nn.Linear):
                last = name.split(".")[-1]
                if last in targets:
                    # 找到父模块并替换属性
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
        B, N, C = fi.shape
        Ti = self.make_tokens(i)                    # [m, c]
        Si = torch.matmul(fi, Ti.t()) / math.sqrt(C)  # [B, N, m]
        Si = F.softmax(Si, dim=-1)

        if self.m > 1:
            Si_sel = Si[..., 1:]                    # [B, N, m-1]
            Ti_sel = Ti[1:, :]                      # [m-1, c]
        else:
            return fi

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
    def __init__(self,
                 hub_dir='dinov2-main/',
                 model_name='dinov2_vitb14',
                 num_classes: int = 19,
                 m_tokens: int = 64,
                 rank_r: int = 8,
                 cq: int = 256,
                 layer_indices: Tuple[int, ...] = (6, 10, 14, 22),  # 0-based → 第7/11/15/23层
                 use_lora: bool = False,           # 与论文“少参数”不完全一致，默认 False
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
        self.out_ch = embed_dim

        if use_lora and lora_r > 0:
            apply_lora_to_dinov2(self.vit, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        if freeze_backbone:
            for n, p in self.vit.named_parameters():
                if use_lora and (".A." in n or ".B." in n):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        self.num_blocks = len(getattr(self.vit, "blocks", []))
        self.rein = ReinModule(num_layers=self.num_blocks, c=self.out_ch, m=m_tokens, r=rank_r, cq=cq)

        self.decode = Mask2FormerLite(in_channels=self.out_ch, num_classes=num_classes, cq=cq, num_feats=len(layer_indices))

        self.layer_indices = tuple([i for i in layer_indices if i < self.num_blocks])
        assert len(self.layer_indices) > 0, "layer_indices 不能为空，且需 < num_blocks"

        self._pre_hooks = []
        self._post_hooks = []
        self._feat_tokens: Dict[int, torch.Tensor] = {}
        self._register_hooks()


    def _register_hooks(self):
        def make_pre(i):
            def _pre(module, inputs):
                (x,) = inputs  # [B,N,C]
                x_new = self.rein.refine(i, x)
                return (x_new,)
            return _pre

        def make_post(i):
            def _post(module, inputs, output):
                # output: [B,N,C]
                if i in self.layer_indices:
                    self._feat_tokens[i] = output.detach() if not self.training else output
                return output
            return _post

        for i, blk in enumerate(self.vit.blocks):
            self._pre_hooks.append(blk.register_forward_pre_hook(make_pre(i)))
            self._post_hooks.append(blk.register_forward_hook(make_post(i)))

    def remove_hooks(self):
        for h in self._pre_hooks + self._post_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._pre_hooks.clear()
        self._post_hooks.clear()

    @torch.no_grad()
    def _infer_special_tokens(self, N: int, H_pad: int, W_pad: int) -> int:
        has_cls = 1 if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None else 0
        num_reg = None
        if hasattr(self.vit, "num_register_tokens"):
            try:
                num_reg = int(self.vit.num_register_tokens)
            except Exception:
                num_reg = None
        if num_reg is None and hasattr(self.vit, "register_tokens") and self.vit.register_tokens is not None:
            try:
                num_reg = int(self.vit.register_tokens.shape[1])
            except Exception:
                num_reg = None
        if num_reg is not None:
            return has_cls + num_reg
    
        h = H_pad // self.patch
        w = W_pad // self.patch
        spatial = h * w
        extra = N - spatial
        if extra < 0:
            extra = 0
        return extra
    
    @torch.no_grad()
    def _tokens_to_featmap(self, tokens: torch.Tensor, H_pad: int, W_pad: int) -> torch.Tensor:
        """
        将 block 输出 tokens [B, N, C] → [B, C, h, w]
        自动适配：有/无 register，有/无 CLS，非方形输入
        """
        B, N, C = tokens.shape
        h = H_pad // self.patch
        w = W_pad // self.patch
        spatial = h * w
    
        if not hasattr(self, "_num_special_tokens") or self._num_special_tokens is None:
            self._num_special_tokens = self._infer_special_tokens(N, H_pad, W_pad)
    
        extra = int(self._num_special_tokens)
    
        if N < extra + spatial:
            extra_dyn = max(0, N - spatial)
            if abs(extra_dyn - extra) <= 5:
                extra = extra_dyn
                self._num_special_tokens = extra
            else:
                raise RuntimeError(
                    f"[tokens_to_featmap] tokens 计数异常: N={N}, spatial={spatial}, "
                    f"cached_extra={self._num_special_tokens}, dyn_extra={extra_dyn}"
                )
    
        if extra > 0:
            tokens = tokens[:, extra: extra + spatial, :]
        else:
            tokens = tokens[:, :spatial, :]
    
        fmap = tokens.transpose(1, 2).contiguous().view(B, C, h, w)
        return fmap



    def _pad_to_patch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
        B, C, H, W = x.shape
        p = self.patch
        newH = math.ceil(H / p) * p
        newW = math.ceil(W / p) * p
        pad_h = newH - H
        pad_w = newW - W
        if pad_h > 0 or pad_w > 0:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        else:
            pad_left = pad_right = pad_top = pad_bottom = 0
        return x, (pad_left, pad_right, pad_top, pad_bottom)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        输入: x [B, 1/3, H, W]
        输出: {'logits': [B, num_classes, H, W]}
        """
        H, W = x.shape[-2:]
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x_pad, (pl, pr, pt, pb) = self._pad_to_patch(x)
        H_pad, W_pad = x_pad.shape[-2:]

        self._feat_tokens.clear()

        _ = self.vit.forward_features(x_pad)  # 不直接用返回；多层 tokens 已在 hook 中存下

        feats = []
        for i in self.layer_indices:
            tokens = self._feat_tokens.get(i, None)
            if tokens is None:
                raise RuntimeError(f"未捕获到第 {i} 层的 tokens，请检查 layer_indices 是否越界/命名是否一致")
            fmap = self._tokens_to_featmap(tokens, H_pad, W_pad)  # [B,C,h,w]
            feats.append(fmap)

        Q = self.rein._stack_Q_layers()  # [m, cq]
 
        if Q.shape[0] > 1:               # 丢掉第 1 个 no-change
            Q = Q[1:, :]                 # [m-1, cq]


        logits = self.decode(feats, Q, out_hw=(H_pad // 1, W_pad // 1))  # [B,K,H_pad,W_pad]

        if any(v > 0 for v in (pl, pr, pt, pb)):
            logits = logits[..., pt: H_pad - pb, pl: W_pad - pr]  # [B,K,H,W]

        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return {'logits': logits}


def build_dinov2_rein_seg(
    hub_dir='dinov2-main/',
    model_name='dinov2_vitb14',
    num_classes=19,
    m_tokens=64,
    rank_r=8,
    cq=256,
    layer_indices=(6, 10, 14, 22),
    use_lora=False,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    freeze_backbone=True,
) -> nn.Module:
    return DINOv2ReinSeg(
        hub_dir=hub_dir,
        model_name=model_name,
        num_classes=num_classes,
        m_tokens=m_tokens,
        rank_r=rank_r,
        cq=cq,
        layer_indices=layer_indices,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        freeze_backbone=freeze_backbone,
    )
