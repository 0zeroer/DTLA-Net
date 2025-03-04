# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os.path

from os.path import join as pjoin

import timm
import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import networks.vit_seg_configs as configs
from networks.vit_seg_modeling_resnet_skip import ResNetV2
from networks.ConvFormer import Setr_ConvFormer
import torch.nn.functional as F
from networks.attention import LinAngularAttention
from networks.scSE import scSE
from networks.ODConv import ODConv2d
from networks.GAM import GAM_Attention

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)  # (B,3,H,W) -> (B,1024,H/16,W/16)
            # x:(B,1024,H/16,W/16)  (6,1024,14,14)
            # features[0]:(B,512,H/8,W/8)  (6,512,28,28)
            # features[1]:(B,256,H/4,W/4)  (6,256,56,56)
            # features[2]:(B,64,H/2,W/2)  (6,64,112,112)
        else:
            features = None
        embeddings = x
        return embeddings, features

class Encoder(nn.Module):
    def __init__(self, config, vis, n_channels=1024):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(1):
            layer = Setr_ConvFormer(config, n_channels)
            self.layer.append(copy.deepcopy(layer))
        self.conv = nn.Conv2d(n_channels, n_channels // 2, kernel_size=1)

    def forward(self, hidden_states):   # (B, n_patches, 768):(B, 1024, 768)
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = hidden_states  # (B,196,768)
        return encoded, attn_weights    # encoded：(B,196,768)


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):   # (B,3,H,W)
        embedding_output, features = self.embeddings(input_ids)  # (B, 196, 768) # (B,512,H/8,W/8);(B,256,H/4,W/4);(B,64,H/2,W/2)
        encoded, attn_weights = self.encoder(embedding_output)   # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

# class CBAM_Layer(nn.Module):
#     def __init__(self, channel, reduction=16, spatial_kernel=7):
#         super(CBAM_Layer, self).__init__()
#
#         # channel attention 压缩H,W为1
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # shared MLP
#         self.mlp = nn.Sequential(
#             # Conv2d比Linear方便操作
#             # nn.Linear(channel, channel // reduction, bias=False)
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             # inplace=True直接替换，节省内存
#             nn.ReLU(inplace=True),
#             # nn.Linear(channel // reduction, channel,bias=False)
#             nn.Conv2d(channel // reduction, channel, 1, bias=False)
#         )
#
#         # spatial attention
#         self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
#                               padding=spatial_kernel // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.mlp(self.max_pool(x))
#         avg_out = self.mlp(self.avg_pool(x))
#         channel_out = self.sigmoid(max_out + avg_out)
#         x = channel_out * x
#
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
#         x = spatial_out * x
#         return x

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         ratio = in_planes // 32
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


# class CASA_Attention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.CA = ChannelAttention(in_channels)
#         self.SA = SpatialAttention()
#
#     def forward(self, x):
#         c = self.CA(x)
#         s = self.SA(x)
#         x = c + s + x
#         return x

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512        # 解码器的输入通道数
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        if self.config.n_skip != 0:
            skip_channels = [512,256,64,16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]
        self.conv11 = Conv2dReLU(1024, 256, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv12 = Conv2dReLU(256, 256, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv21 = Conv2dReLU(512, 128, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv22 = Conv2dReLU(128, 128, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv31 = Conv2dReLU(192, 64, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv32 = Conv2dReLU(64, 64, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv41 = Conv2dReLU(64, 16, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv42 = Conv2dReLU(16, 16, kernel_size=3, padding=1, use_batchnorm=True)

        self.LAatt0 = LinAngularAttention(in_channels[0])
        # self.LAatt1 = LinAngularAttention(in_channels[1])
        # self.LAatt2 = LinAngularAttention(in_channels[2])
        # self.LAatt3 = LinAngularAttention(in_channels[3])
        # self.ODConv2d = ODConv2d(in_channels[0], in_channels[0], kernel_size=3)
        # self.scSE = scSE(in_channels[0])
        # self.CASA_Attention0 = CASA_Attention(in_channels[0])
        # self.CASA_Attention1 = CASA_Attention(in_channels[1])
        # self.CASA_Attention2 = CASA_Attention(in_channels[2])
        # self.CASA_Attention3 = CASA_Attention(in_channels[3])

    def forward(self, hidden_states, features=None):
        x0 = hidden_states  # (4,512,14,14)
        x0 = self.LAatt0(x0)
        # c = self.CA(x0)
        # s = self.SA(x0)
        # x0 = c + s + x0
        # x0 = self.CASA_Attention0(x0)
        # x0 = self.scSE(x0)
        # x0 = self.ODConv2d(x0)
        x1 = self.up(x0)                            # (4,512,28,28)
        x1 = torch.cat([x1, features[0]], dim=1)    # (4,1024,28,28)
        x1 = self.conv11(x1)                        # (4,256,28,28)
        x1 = self.conv12(x1)
        # x1 = self.CASA_Attention1(x1)
        # x1 = self.LAatt1(x1)
        x2 = self.up(x1)                            # (4,256,56,56)
        x2 = torch.cat([x2, features[1]], dim=1)    # (4,512,56,56)
        x2 = self.conv21(x2)                        # (4,128,56,56)
        x2 = self.conv22(x2)

        # x2 = self.CASA_Attention2(x2)
        # x2 = self.LAatt2(x2)
        x3 = self.up(x2)                            # (4,128,112,112)
        x3 = torch.cat([x3, features[2]], dim=1)    # (4,192,112,112)
        x3 = self.conv31(x3)                        # (4,64,112,112)
        x3 = self.conv32(x3)

        # x3 = self.CASA_Attention3(x3)
        # x3 = self.LAatt3(x3)
        x4 = self.up(x3)                            # (4,64,224,224)
        x4 = self.conv41(x4)                        # (4,16,224,224)
        x4 = self.conv42(x4)

        return x4


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if x.size()[1] == 1:  # 如果图片是灰度图就在其通道方向进行复制从1维转成3维
            x = x.repeat(1, 3, 1, 1)   # (B,3,H,W):(6,3,224,224)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden):(B,196,768)
        x = self.decoder(x, features)       # (B,16,H,W):(B,16,224,224)
        logits = self.segmentation_head(x)  # (B,9,H,W):(B,9,224,224)
        return logits


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}