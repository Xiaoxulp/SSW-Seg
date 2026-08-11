import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
    ):
        super().__init__()
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

# class VSSBlock(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)
#
#         # --- 新增 linear + conv 特征增强 ---
#         self.pre_linear = nn.Linear(hidden_dim, hidden_dim)  # 不改变通道
#         self.pre_conv = nn.Conv2d(
#             in_channels=hidden_dim,
#             out_channels=hidden_dim,
#             kernel_size=3,
#             padding=1,
#             groups=hidden_dim  # 可选深度卷积
#         )
#         self.act = nn.SiLU()
#         self.pre_dropout = nn.Dropout(0.1)  # 可选 dropout 防止过拟合
#
#         # 原来的 SS2D 注意力
#         self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
#         self.drop_path = DropPath(drop_path)
#
#     def forward(self, input: torch.Tensor):
#         # LayerNorm
#         x = self.ln_1(input)  # [B,H,W,C]
#
#         # ---- Linear + Conv 特征增强 ----
#         x_enhance = self.pre_linear(x)          # [B,H,W,C]
#         x_enhance = x_enhance.permute(0, 3, 1, 2)  # [B,C,H,W]
#         x_enhance = self.pre_conv(x_enhance)
#         x_enhance = self.act(x_enhance)
#         x_enhance = x_enhance.permute(0, 2, 3, 1)  # [B,H,W,C]
#         x_enhance = self.pre_dropout(x_enhance)
#
#         # ---- SS2D 注意力 ----
#         x_attn = self.self_attention(x)  # [B,H,W,C]
#
#         # ---- 融合增强 + 注意力输出 + 残差 ----
#         out = input + self.drop_path(x_attn + x_enhance)
#         return out


# class FFN_Lite(nn.Module):
#     def __init__(self, embed_dim, ffn_ratio=4.0, act_layer=nn.GELU, dropout=0.0):
#         super().__init__()
#         hidden_dim = int(embed_dim * ffn_ratio)
#
#         # 通道扩展
#         self.fc1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
#         self.act = act_layer()
#
#         # 局部卷积增强
#         self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
#
#         # 投影回原通道
#         self.fc2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#
#     def forward(self, x):
#         shortcut = x
#         x = self.fc1(x)
#         x = self.act(x)
#         x = x + self.dwconv(x)  # 残差增强局部特征
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         x = x + shortcut  # 全局残差
#         return x

class FFN_Lite(nn.Module):
    def __init__(self, embed_dim, ffn_ratio=4.0, act_layer=nn.GELU, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * ffn_ratio)

        self.fc1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.act = act_layer()

        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1,
            groups=hidden_dim
        )

        self.fc2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, H, W, C]
        shortcut = x

        # NHWC → NCHW
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        # NCHW → NHWC
        x = x.permute(0, 2, 3, 1).contiguous()

        x = x + shortcut
        return x

class CNN2MambaAdapter(nn.Module):
    def __init__(self, in_dim, mamba_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, mamba_dim, 1, bias=False)
        self.norm = nn.LayerNorm(mamba_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)   # [B, H, W, C]
        x = self.norm(x)
        return x

class Mamba2CNNAdapter(nn.Module):
    def __init__(self, mamba_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(mamba_dim, out_dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2)   # [B, C, H, W]
        x = self.proj(x)
        x = self.norm(x)
        return x

class VSSAdapter(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        ffn_ratio=4.0,
        drop_path=0.0,
        d_state=16,
        attn_drop_rate=0.0,
        **kwargs
    ):
        super().__init__()


        mamba_dim = out_dim if out_dim is not None else dim


        self.pre_adapter = CNN2MambaAdapter(dim, mamba_dim)
        self.vss = VSSBlock(hidden_dim=mamba_dim, **kwargs)
        self.ffn = FFN_Lite(mamba_dim, ffn_ratio=ffn_ratio)

        self.post_adapter = Mamba2CNNAdapter(mamba_dim, dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x_m = self.pre_adapter(x)    # CNN → Mamba
        x_m = self.vss(x_m)          # Mamba / VAABlock
        # x_m = self.ffn(x_m)
        x_m = self.vss(x_m)  # Mamba / VAABlock
        x_m = self.vss(x_m)  # Mamba / VAABlock

        x_m = self.ffn(x_m)
        x = self.post_adapter(x_m)   # Mamba → CNN
        return x





class FFN(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            act_layer=nn.GELU,
            dropout=0,
    ):
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x




