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


class PPMHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 256, num_classes: int = 19,
                 bins: Tuple[int, ...] = (1, 2, 3, 6), dropout: float = 0.1):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
            for b in bins
        ])
        out_in = in_channels + len(bins) * mid_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_in, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        H, W = out_size
        feats = [x]
        for s in self.stages:
            y = s(x)
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(y)
        feat = torch.cat(feats, dim=1)
        feat = self.bottleneck(feat)
        logits = self.classifier(feat)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


class DINOv2SegPPM(nn.Module):
    def __init__(self,
                 hub_dir='dinov2-main/',
                 model_name='dinov2_vitb14',
                 num_classes=19,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.0,
                 in_ch_override=None,
                 freeze_backbone=True):
        super().__init__()
        import timm

        self.vit = torch.hub.load(hub_dir, model_name, source='local', pretrained=True)
        self.patch = 14
        
        # import timm
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=512)
        # self.vit.reset_classifier(0)
        # path = "/data1/interim/VFM_KD/DINOv2_large2base_training_deit_IMNET_FM_bias_head_2_epoch100_citys_epoch300_512.pth"
        # state_dict = torch.load(path, map_location="cpu")
        # self.vit.load_state_dict(state_dict, strict=False)
        # self.patch = 16
        
        

        embed_dim = getattr(self.vit, "embed_dim", None)
        if embed_dim is None:
            embed_dim = in_ch_override or 768
        self.out_ch = embed_dim

        if lora_r and lora_r > 0:
            apply_lora_to_dinov2(self.vit, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        if freeze_backbone:
            for n, p in self.vit.named_parameters():
                if (".A." in n) or (".B." in n):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        self.seg_head = PPMHead(
            in_channels=self.out_ch,
            mid_channels=256,
            num_classes=num_classes
        )


    @torch.no_grad()
    def _to_featmap(self, x, out_dict):
        if 'x_norm_patchtokens' in out_dict and out_dict['x_norm_patchtokens'] is not None:
            tokens = out_dict['x_norm_patchtokens']  # [B, L, C]
            B, L, C = tokens.shape
            h = int(round(x.shape[-2] / self.patch))
            w = int(round(x.shape[-1] / self.patch))
            if h * w != L:
                hw = int(math.sqrt(L))
                h = w = hw
            fmap = tokens.transpose(1, 2).contiguous().view(B, C, h, w)
            return fmap

        if 'feats' in out_dict and isinstance(out_dict['feats'], (list, tuple)) and len(out_dict['feats']) > 0:
            fm = out_dict['feats'][-1]
            if fm.dim() == 4 and fm.shape[1] < 8 and fm.shape[-1] != x.shape[-1]:
                fm = fm.permute(0, 3, 1, 2).contiguous()
            return fm

        raise RuntimeError("DINOv2 forward_features did not return expected keys")

    def forward_backbone(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

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

        # dinov2 前向
        out = self.vit.forward_features(x)
        fm = self._to_featmap(x, out)  # [B, C, h, w]
        return fm

    def forward(self, x):

        H, W = x.shape[-2:]
        fm = self.forward_backbone(x)
        logits = self.seg_head(fm, (H, W))
        return {'logits': logits}

