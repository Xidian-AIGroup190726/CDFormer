import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from torchvision import models
from logging import Logger
from typing import Optional, Any

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)
class INN(nn.Module):
    def __init__(self, in_channels, out_channels, is_HE=True):
        # HP: 高频特征提取器使用的INN
        super(INN, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        if is_HE:
            self.theta_phi = InvertedResidualBlock(inp=in_channels, oup=out_channels, expand_ratio=2)
            self.theta_rho = InvertedResidualBlock(inp=in_channels, oup=out_channels, expand_ratio=2)
            self.theta_eta = InvertedResidualBlock(inp=in_channels, oup=out_channels, expand_ratio=2)

        else:
            self.theta_phi = InvertedResidualBlock(inp=in_channels/2, oup=out_channels/2, expand_ratio=2)
            self.theta_rho = InvertedResidualBlock(inp=in_channels/2, oup=out_channels/2, expand_ratio=2)
            self.theta_eta = InvertedResidualBlock(inp=in_channels/2, oup=out_channels/2, expand_ratio=2)

        # self.shffleconv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)
    def ChannelSeparate(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def checkerboard_mask(self, height, width, device=None, reverse=False,
                          dtype=torch.float32, requires_grad=False):
        """Get a checkerboard mask, such that no two entries adjacent entries
        have the same value. In non-reversed mask, top-left entry is 0.

        Args:
            height (int): Number of rows in the mask.
            width (int): Number of columns in the mask.
            reverse (bool): If True, reverse the mask (i.e., make top-left entry 1).
                Useful for alternating masks in RealNVP.
            dtype (torch.dtype): Data type of the tensor.
            device (torch.device): Device on which to construct the tensor.
            requires_grad (bool): Whether the tensor requires gradient.

        Returns:
            mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width).
        """
        checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
        mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

        if reverse:
            mask = 1 - mask

        # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
        mask = mask.view(1, 1, height, width)

        return mask

    def CheckerBoardSeparate(self, x):
        mask = self.checkerboard_mask(x.size(2), x.size(3), device=x.device)
        z1 = mask * x
        z2 = x - z1
        return z1,z2

    def forward(self, x, is_HE=True):
        if is_HE:
            z1, z2 = self.CheckerBoardSeparate(x)
        else:
            z1, z2 = self.ChannelSeparate(x)
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class LFIE(nn.Module):
    def __init__(self):
        self.rate1 = 0.5
        self.rate2 = 0.5
        self.transformer =
        self.InnBlock