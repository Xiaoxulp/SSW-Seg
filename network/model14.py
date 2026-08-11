import math
import torch
import torch.nn as nn
from network.resnet import resnet50  # Fix import to match package path when running from project root.
import torch.nn.functional as F

from network.wtconv import WTConv2d


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class dilated_conv(nn.Module):
    """
    Wavelet Multi-Branch Module (3 branches, different kernel sizes)
    用小波卷积替代 dilated_conv 的多分支结构
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # 🔹 分支1：kernel=3
        self.b1 = nn.Sequential(
            WTConv2d(in_c, in_c, kernel_size=1, wt_levels=1),
            CBR(in_c, out_c, kernel_size=1, padding=0)
        )

        # 🔹 分支2：kernel=5
        self.b2 = nn.Sequential(
            WTConv2d(in_c, in_c, kernel_size=3, wt_levels=2),
            CBR(in_c, out_c, kernel_size=1, padding=0)
        )

        # 🔹 分支3：kernel=7
        self.b3 = nn.Sequential(
            WTConv2d(in_c, in_c, kernel_size=5, wt_levels=3),
            CBR(in_c, out_c, kernel_size=1, padding=0)
        )

        # 🔹 融合
        self.fuse = CBR(out_c * 3, out_c, kernel_size=3, padding=1, act=False)

        # 🔹 残差（稳定训练）
        self.res = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)

        # 🔹 空间注意力
        self.sa = spatial_attention()

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)

        # 多尺度拼接
        xc = torch.cat([x1, x2, x3], dim=1)

        # 融合
        xc = self.fuse(xc)

        # 残差连接
        xr = self.res(x)

        x = self.relu(xc + xr)

        # 注意力强化
        x = self.sa(x)

        return x

"""Decouple Layer"""


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc


"""Auxiliary Head"""


class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()

        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc


class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5


        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted


# class decoder_block(nn.Module):
#     def __init__(self, in_c, out_c, scale=2):
#         super().__init__()
#         self.scale = scale
#         self.relu = nn.ReLU(inplace=True)
#
#         self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
#         self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)
#         self.c2 = CBR(out_c, out_c, act=False)
#         self.c3 = CBR(out_c, out_c, act=False)
#         self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)
#         self.ca = channel_attention(out_c)
#         self.sa = spatial_attention()
#
#     def forward(self, x, skip):
#         x = self.up(x)
#         x = torch.cat([x, skip], axis=1)
#
#         x = self.c1(x)
#
#         s1 = x
#         x = self.c2(x)
#         x = self.relu(x + s1)
#
#         s2 = x
#         x = self.c3(x)
#         x = self.relu(x + s2 + s1)
#
#         s3 = x
#         x = self.c4(x)
#         x = self.relu(x + s3 + s2 + s1)
#
#         x = self.ca(x)
#         x = self.sa(x)
#         return x

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


# class decoder_block(nn.Module):
#     def __init__(self, in_c, out_c, wt_levels=1, wt_type='db1'):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#
#         # 用于 skip 通道对齐
#         self.skip_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
#
#         # 融合卷积
#         self.c1 = nn.Conv2d(in_c + out_c, out_c, kernel_size=1, bias=False)
#         self.c1_no_skip = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
#
#         # 小波卷积残差块
#         self.c2 = WTConv2d(out_c, out_c, kernel_size=3, wt_levels=wt_levels, wt_type=wt_type)
#         self.c3 = WTConv2d(out_c, out_c, kernel_size=3, wt_levels=wt_levels, wt_type=wt_type)
#         self.c4 = WTConv2d(out_c, out_c, kernel_size=1, wt_levels=1, wt_type=wt_type)
#
#         # CBAM 注意力
#         self.cbam = CBAM(out_c)
#
#     def forward(self, x, skip=None):
#         # -------- skip 拼接 --------
#         if skip is not None:
#             # 上采样到 skip 尺寸
#             x = F.interpolate(
#                 x, size=(skip.shape[2], skip.shape[3]),
#                 mode='bilinear', align_corners=True
#             )
#
#             # 通道对齐
#             if skip.shape[1] != x.shape[1]:
#                 skip = self.skip_conv(skip)
#
#             # 融合
#             x = torch.cat([x, skip], dim=1)
#             x = self.c1(x)
#         else:
#             # 无 skip，直接卷积
#             x = self.c1_no_skip(x)
#
#         # -------- 残差卷积块 --------
#         s = x
#         x = self.relu(self.c2(x) + s)
#         s = x
#         x = self.relu(self.c3(x) + s)
#         s = x
#         x = self.relu(self.c4(x) + s)
#
#         # -------- CBAM 注意力 --------
#         x = self.cbam(x)
#
#         # -------- 最后上采样（无 skip 时） --------
#         if skip is None:
#             # 自动上采样到输入的 2 倍尺寸（保持 decoder 层级关系）
#             H, W = x.shape[2] * 2, x.shape[3] * 2
#             x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#
#         return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        # -------- 上采样 --------
        self.up = nn.Upsample(scale_factor=scale,
                              mode="bilinear",
                              align_corners=True)

        # -------- 融合卷积 --------
        self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c1_no_skip = CBR(in_c, out_c, kernel_size=1, padding=0)

        # -------- 残差卷积块 --------
        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)

        # -------- 注意力 --------
        self.cbam = CBAM(out_c)

    def forward(self, x, skip=None):

        # ===== 上采样 =====
        x = self.up(x)

        # ===== skip 拼接 =====
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.c1(x)
        else:
            x = self.c1_no_skip(x)

        # ===== 残差块 =====
        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        # ===== 注意力 =====
        x = self.cbam(x)

        return x


class output_block(nn.Module):
    def __init__(self, in_c, out_c=1):
        super().__init__()

        self.c1 = CBR(in_c, 128, 3, padding=1)


        self.c2 = CBR(128, 64, 1, padding=0)  # 添加 padding=0

        self.c3 = nn.Conv2d(64, out_c, 1, padding=0)  # kernel_size=1 时 padding=0
        self.sig = nn.Sigmoid()

    def forward(self, x):

        x = self.c1(x)

        x = self.c2(x)

        x = self.c3(x)

        return self.sig(x)



class multiscale_feature_aggregation(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.c11 = CBR(in_c[0], out_c, kernel_size=1, padding=0)
        self.c12 = CBR(in_c[1], out_c, kernel_size=1, padding=0)
        self.c13 = CBR(in_c[2], out_c, kernel_size=1, padding=0)
        self.c14 = CBR(out_c * 3, out_c, kernel_size=1, padding=0)

        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)

    def forward(self, x1, x2, x3):
        x1 = self.up_4x4(x1)
        x2 = self.up_2x2(x2)

        x1 = self.c11(x1)
        x2 = self.c12(x2)
        x3 = self.c13(x3)
        x = torch.cat([x1, x2, x3], axis=1)
        x = self.c14(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        return x


class CDFAPreprocess(nn.Module):

    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x



class ConDSeg14(nn.Module):
    def __init__(self, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W

        """ Backbone: ResNet50 """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [batch_size, 256, h/4, w/4]
        self.layer2 = backbone.layer2  # [batch_size, 512, h/8, w/8]
        self.layer3 = backbone.layer3  # [batch_size, 1024, h/16, w/16]

        """ FEM """
        self.dconv1 = dilated_conv(64, 128)
        self.dconv2 = dilated_conv(256, 128)
        self.dconv3 = dilated_conv(512, 128)
        self.dconv4 = dilated_conv(1024, 128)

        """ Decouple Layer """
        self.decouple_layer = DecoupleLayer(1024, 128)

        """ Adjust the shape of decouple output """
        self.preprocess_fg4 = CDFAPreprocess(128, 128, 1)  # 1/16
        self.preprocess_bg4 = CDFAPreprocess(128, 128, 1)  # 1/16

        self.preprocess_fg3 = CDFAPreprocess(128, 128, 2)  # 1/8
        self.preprocess_bg3 = CDFAPreprocess(128, 128, 2)  # 1/8

        self.preprocess_fg2 = CDFAPreprocess(128, 128,4)  # 1/4
        self.preprocess_bg2 = CDFAPreprocess(128, 128, 4)  # 1/4

        self.preprocess_fg1 = CDFAPreprocess(128, 128,8)  # 1/2
        self.preprocess_bg1 = CDFAPreprocess(128, 128, 8)  # 1/2

        """ Auxiliary Head """
        self.aux_head = AuxiliaryHead(128)

        """ Contrast-Driven Feature Aggregation """
        self.up2X = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cdfa4 = ContrastDrivenFeatureAggregation(128, 128, 4)
        self.cdfa3 = ContrastDrivenFeatureAggregation(128 + 128, 128, 4)
        self.cdfa2 = ContrastDrivenFeatureAggregation(128 + 128, 128, 4)
        self.cdfa1 = ContrastDrivenFeatureAggregation(128 + 128, 128, 4)

        """ Decoder """
        self.dec4 = decoder_block(128, 128)  # 不用 scale=2
        self.dec3 = decoder_block(128, 128)
        self.dec2 = decoder_block(128, 128)
        self.dec1 = decoder_block(128, 128)

        """ Output Block """
        self.output_block = output_block(128, 1)

    def forward(self, image):
        """ Backbone: ResNet50 """
        x0 = image
        x1 = self.layer0(x0)  ## [-1, 64, h/2, w/2]
        x2 = self.layer1(x1)  ## [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)  ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)  ## [-1, 1024, h/16, w/16]

        """ Dilated Conv """
        d1 = self.dconv1(x1)
        d2 = self.dconv2(x2)
        d3 = self.dconv3(x3)
        d4 = self.dconv4(x4)

        """ Decouple Layer """
        f_fg, f_bg, f_uc = self.decouple_layer(x4)
        """ Auxiliary Head """
        mask_fg, mask_bg, mask_uc = self.aux_head(f_fg, f_bg, f_uc)

        """ Contrast-Driven Feature Aggregation """
        f_fg4 = self.preprocess_fg4(f_fg)
        f_bg4 = self.preprocess_bg4(f_bg)
        f_fg3 = self.preprocess_fg3(f_fg)
        f_bg3 = self.preprocess_bg3(f_bg)
        f_fg2 = self.preprocess_fg2(f_fg)
        f_bg2 = self.preprocess_bg2(f_bg)
        f_fg1 = self.preprocess_fg1(f_fg)
        f_bg1 = self.preprocess_bg1(f_bg)

        f4 = self.cdfa4(d4, f_fg4, f_bg4)
        f4_up = self.up2X(f4)
        f_4_3 = torch.cat([d3, f4_up], dim=1)
        f3 = self.cdfa3(f_4_3, f_fg3, f_bg3)
        f3_up = self.up2X(f3)
        f_3_2 = torch.cat([d2, f3_up], dim=1)
        f2 = self.cdfa2(f_3_2, f_fg2, f_bg2)
        f2_up = self.up2X(f2)
        f_2_1 = torch.cat([d1, f2_up], dim=1)
        f1 = self.cdfa1(f_2_1, f_fg1, f_bg1)

        """ Decoder """
        d4 = self.dec4(f4, f3)  # 1/16 → 1/8
        d3 = self.dec3(d4, f2)  # 1/8  → 1/4
        d2 = self.dec2(d3, f1)  # 1/4  → 1/2   ← f1 在这里！
        d1 = self.dec1(d2, None)

        """ Output Block """
        mask = self.output_block(d1)

        return mask, mask_fg, mask_bg, mask_uc






if __name__ == "__main__":
    model = ConDSeg14().cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_tensor)
    print(output.shape)