import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from torchvision import models
from logging import Logger
from typing import Optional, Any


def conv1x1(in_channels, out_channels, stride=1, padding=0, *args, **kwargs):
    # type: (int, int, int, int, Any, Any) -> nn.Conv2d
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                     stride=stride, padding=padding, *args, **kwargs)


def conv3x3(in_channels, out_channels, stride=1, padding=1, *args, **kwargs):
    # type: (int, int, int, int, Any, Any) -> nn.Conv2d
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                     stride=stride, padding=padding, *args, **kwargs)


def build_norm_layer(logger, n_feats, norm_type='BN', *args, **kwargs):
    r""" build a normalization layer in [BatchNorm, InstanceNorm]
    Args:
        logger (Logger): logger
        n_feats (int): output channel
        norm_type (str): 'BN' or 'IN'
    Returns:
        nn.Module: expected normalization layer
    """
    if norm_type == 'BN':
        return nn.BatchNorm2d(num_features=n_feats, *args, **kwargs)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(num_features=n_feats, *args, **kwargs)
    else:
        logger.error(f'no such type of norm_layer:{norm_type}')
        raise SystemExit(f'no such type of norm_layer:{norm_type}')


class ResBlock(nn.Module):
    def __init__(self, logger, n_feats, norm_type='BN'):
        # type: (Logger, int, Optional[str]) -> None
        super(ResBlock, self).__init__()
        self.basic = []
        self.basic.append(conv3x3(n_feats, n_feats))
        if norm_type is not None:
            self.basic.append(build_norm_layer(logger, n_feats, norm_type))
        self.basic.append(nn.ReLU(True))
        self.basic.append(conv3x3(n_feats, n_feats))
        if norm_type is not None:
            self.basic.append(build_norm_layer(logger, n_feats, norm_type))
        self.basic = nn.Sequential(*self.basic)

    def forward(self, x):
        temp = self.basic(x)
        return x + temp

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        inp = int(inp)
        oup = int(oup)
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(

            # pw 60w
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # dw 30w
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=16, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # pw-linear 60w
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # conv1x1(hidden_dim, oup),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),

        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class INN(nn.Module):
    def __init__(self, in_channels, out_channels, iter=1):
        super(INN, self).__init__()
        self.n_blocks = iter
        self.theta_phi = nn.ModuleList()
        self.theta_rho = nn.ModuleList()
        self.theta_eta = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.theta_phi.append(InvertedResidualBlock(inp=in_channels, oup=out_channels, expand_ratio=1))
            self.theta_rho.append(InvertedResidualBlock(inp=in_channels, oup=out_channels, expand_ratio=1))
            self.theta_eta.append(InvertedResidualBlock(inp=in_channels, oup=out_channels, expand_ratio=1))

    def forward(self, z1, z2):
        # 进行格式重排(Transformer将数据变换为:[batch, w, h, head]),图像数据为[batch, head/c, w, h]
        z1 = z1.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)

        for i in range(self.n_blocks):
            z2 = z2 + self.theta_phi[i](z1)
            z1 = z1 * torch.exp(self.theta_rho[i](z2)) + self.theta_eta[i](z2)
            # z1, z2 = z2, z1

        z1 = z1.permute(0, 2, 3, 1)
        z2 = z2.permute(0, 2, 3, 1)

        return z1, z2

class ResChAttnBlock(nn.Module):
    r"""
        Residual Channel Attention Block
    """
    def __init__(self, logger, n_feats, norm_type='BN'):
        # type: (Logger, int, Optional[str]) -> None
        super(ResChAttnBlock, self).__init__()
        self.conv1_block = []
        self.conv1_block.append(conv3x3(n_feats, n_feats))
        if norm_type is not None:
            self.conv1_block.append(build_norm_layer(logger, n_feats, norm_type))
        self.conv1_block.append(nn.ReLU(True))
        self.conv1_block.append(conv3x3(n_feats, n_feats))
        if norm_type is not None:
            self.conv1_block.append(build_norm_layer(logger, n_feats, norm_type))
        self.conv1_block = nn.Sequential(*self.conv1_block)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn_block = []
        self.attn_block.append(nn.Linear(n_feats, n_feats // 2))
        self.attn_block.append(nn.ReLU(True))
        self.attn_block.append(nn.Linear(n_feats // 2, n_feats))
        self.attn_block.append(nn.Sigmoid())
        self.attn_block = nn.Sequential(*self.attn_block)

        self.conv2_block = []
        self.conv2_block.append(conv3x3(n_feats * 2, n_feats))
        if norm_type is not None:
            self.conv2_block.append(build_norm_layer(logger, n_feats, norm_type))
        self.conv2_block = nn.Sequential(*self.conv2_block)

    def forward(self, x):
        y = self.conv1_block(x)

        attn = self.global_avg_pool(y)
        attn = attn.squeeze(-1).squeeze(-1)
        attn = self.attn_block(attn)
        attn = attn.unsqueeze(-1).unsqueeze(-1)

        return self.conv2_block(torch.cat([attn * y, y], dim=1)) + x


class Pixel_Discriminator(nn.Module):
    def __init__(self, logger, in_channels, n_feats, norm_type='BN'):
        # type: (Logger, int, int, Optional[str]) -> None
        super(Pixel_Discriminator, self).__init__()
        self.netD = []
        self.netD.append(conv1x1(in_channels, n_feats))  # 256 * 256 * 64
        self.netD.append(nn.LeakyReLU(0.2, True))
        self.netD.append(conv1x1(n_feats, n_feats * 2))  # 256 * 256 * 128
        if norm_type is not None:
            self.netD.append(build_norm_layer(logger, n_feats * 2, norm_type))
        self.netD.append(nn.LeakyReLU(0.2, True))
        self.netD.append(conv1x1(n_feats * 2, 1))  # 256 * 256 * 128
        self.netD = nn.Sequential(*self.netD)

    def forward(self, x):
        return self.netD(x)


class Patch_Discriminator(nn.Module):
    def __init__(self, logger, in_channels, n_feats=64, n_layers=3, norm_type='BN'):
        # type: (Logger, int, int, int, Optional[str]) -> None
        r""" if n_layers=0, then use pixelGAN (rf=1)
             else rf is 16 if n_layers=1
                        34 if n_layers=2
                        70 if n_layers=3
                        142 if n_layers=4
                        286 if n_layers=5
                        574 if n_layers=6
        """
        super(Patch_Discriminator, self).__init__()
        self.netD = []
        self.netD.append(nn.Conv2d(in_channels, n_feats, 4, 2, 1))  # 256 * 256 * 64
        self.netD.append(nn.LeakyReLU(0.2, True))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.netD.append(nn.Conv2d(n_feats * nf_mult_prev, n_feats * nf_mult, 4, 2, 1))
            if norm_type is not None:
                self.netD.append(build_norm_layer(logger, n_feats * nf_mult, norm_type))
            self.netD.append(nn.LeakyReLU(0.2, True))

        # N * N * (ndf*M)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.netD.append(nn.Conv2d(n_feats * nf_mult_prev, n_feats * nf_mult, 4, 1, 1))  # (N-1) * (N-1) * (ndf*M*2)
        if norm_type is not None:
            self.netD.append(build_norm_layer(logger, n_feats * nf_mult, norm_type))
        self.netD.append(nn.LeakyReLU(0.2, True))

        self.netD.append(nn.Conv2d(n_feats * nf_mult, 1, 4, 1, 1))  # (N-2) * (N-2) * 1
        self.netD = nn.Sequential(*self.netD)

    def forward(self, x):
        return self.netD(x)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class VGG_Feat(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1, shift_enable=True):
        super(VGG_Feat, self).__init__()
        self.shift_enable = shift_enable
        # use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        # LTE 是pretrained vgg的各个阶段，不同级别的特征图，越靠后channel越大，H*W越小
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        if self.shift_enable:
            x = self.sub_mean(x)

        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3


class VGG_Discriminator(nn.Module):
    def __init__(self, in_channel=3, in_size=160):
        super().__init__()
        self.conv_head = nn.Sequential(
            conv3x3(in_channel, 32),
            nn.LeakyReLU(0.2, True),
            conv3x3(32, 32, 2),
            nn.LeakyReLU(0.2, True),
            conv3x3(32, 64),
            nn.LeakyReLU(0.2, True),
            conv3x3(64, 64, 2),
            nn.LeakyReLU(0.2, True),
            conv3x3(64, 128),
            nn.LeakyReLU(0.2, True),
            conv3x3(128, 128, 2),
            nn.LeakyReLU(0.2, True),
            conv3x3(128, 256),
            nn.LeakyReLU(0.2, True),
            conv3x3(256, 256, 2),
            nn.LeakyReLU(0.2, True),
            conv3x3(256, 512),
            nn.LeakyReLU(0.2, True),
            conv3x3(512, 512, 2),
            nn.LeakyReLU(0.2, True),
        )

        self.fc_tail = nn.Sequential(
            nn.Linear((in_size // 32) ** 2 * 512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.conv_head(x)
        x = self.fc_tail(x.view(x.size(0), -1))
        return x


class SFTLayer(nn.Module):
    def __init__(self, n_feats=32):
        super(SFTLayer, self).__init__()

        self.SFT_scale_conv0 = conv1x1(n_feats, n_feats//2)
        self.SFT_scale_conv1 = conv1x1(n_feats//2, n_feats)
        self.SFT_shift_conv0 = conv1x1(n_feats, n_feats//2)
        self.SFT_shift_conv1 = conv1x1(n_feats//2, n_feats)

    def forward(self, features, conditions):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(conditions), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(conditions), 0.1, inplace=True))
        return features * (scale + 1) + shift


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, cross_attn, h_attn, inn_iter):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.cross_attn = cross_attn
        if self.cross_attn:
            self.inn = INN(inner_dim, inner_dim, inn_iter)
        self.h_attn = h_attn

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        # nn.Linear 层实际上可以处理任意数量的维度的输入
        if not self.cross_attn:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_q = nn.Linear(dim, inner_dim, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, y=None):

        if self.shifted:
            x = self.cyclic_shift(x)
            if self.cross_attn:
                y = self.cyclic_shift(y)

        b, n_h, n_w, _, h = *x.shape, self.heads
        # print('forward-x: ', x.shape)   # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
        if not self.cross_attn:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            # [N, H//downscaling_factor, W//downscaling_factor, head_dim * heads] * 3
        else:
            kv = self.to_kv(x).chunk(2, dim=-1)
            k = kv[0]
            v = kv[1]
            q = self.to_q(y)

            # np_x = x.cpu().detach().numpy()
            # np_q = q.cpu().detach().numpy()
            # np_k = k.cpu().detach().numpy()
            # np_v = v.cpu().detach().numpy()

            # 加入INN；Q恒等映射,K有变换
            k_inn, q_inn = self.inn(k, q)
            # 高频映射
            if self.h_attn:
                k = k - k_inn
                q = q_inn
            else:
                k = k_inn
                q = q_inn

            qkv = (q, k, v)

        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        # print('forward-q: ', q.shape)   # [N, num_heads, num_win, win_area, hidden_dim/num_heads]
        # print('forward-k: ', k.shape)
        # print('forward-v: ', v.shape)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale  # q * k / sqrt(d)

        # 更改.relative_indices类型
        self.relative_indices = self.relative_indices.long()
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        # [N, H//downscaling_factor, W//downscaling_factor, head_dim * head]
        out = self.to_out(out)
        # [N, H//downscaling_factor, W//downscaling_factor, dim]
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, cross_attn, h_attn, inn_iter):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     cross_attn=cross_attn,
                                                                     h_attn=h_attn,
                                                                     inn_iter=inn_iter)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x, y=None):
        x = self.attention_block(x, y=y)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x  # [N, H//downscaling_factor, W//downscaling_factor, out_channels]

# 在 Swin Transformer 中，regular block 和 shifted block 交替执行是为了提高模型对多尺度信息的感知。这样的设计考虑到了图像中不同位置的特征可能具有不同的重要性。
# 通过交替执行 regular block 和 shifted block，模型能够有效地捕捉到全局和局部的特征。
# 1. **Regular Block（正常块）**：Regular block 使用常规的注意力机制，适用于捕捉全局信息。这种块有助于处理较大尺度的特征。
# 2. **Shifted Block（位置偏移块）**：Shifted block 在注意力机制中引入了位置偏移，通过这种方式，模型更容易捕捉到局部信息。位置偏移可以使得模型更加关注邻近的特征，有助于处理较小尺度的特征。
# 通过这种交替的设计，Swin Transformer 可以在多个尺度上有效地捕捉图像特征，提高模型的感知能力。这对于处理不同尺寸的对象或图像中的结构变化非常有帮助。
class SwinModule(nn.Module):
    def __init__(self, in_channels=4, hidden_dimension=32, layers=2, downscaling_factor=1, num_heads=4, head_dim=16, window_size=4,
                 relative_pos_embedding=True, cross_attn=False, h_attn=False, inn_iter=1):
        r"""
        Args:
            in_channels(int): 输入通道数
            hidden_dimension(int): 隐藏层维数，patch_partition提取patch时有个Linear学习的维数
            layers(int): swin block数，必须为2的倍数，连续的，regular block和shift block
            downscaling_factor: H,W上下采样倍数
            num_heads: multi-attn 的 attn 头的个数
            head_dim:   每个attn 头的维数
            window_size:    窗口大小，窗口内进行attn运算
            relative_pos_embedding: 相对位置编码
            cross_attn: 交叉注意力机制(有两个输入时一定要使用)
            h_attn: 第二个输入是否为高频信息.若为高频,则对INN的结果需要进行反转
        """
        super().__init__()
        # assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        layers = layers * 2
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn, h_attn=h_attn, inn_iter=inn_iter),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn, h_attn=h_attn, inn_iter=inn_iter),
            ]))

    def forward(self, x, y=None):
        if y is None:
            x = self.patch_partition(x)  # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
            for regular_block, shifted_block in self.layers:
                x = regular_block(x)
                x = shifted_block(x)
            return x.permute(0, 3, 1, 2)
            # [N, hidden_dim,  H//downscaling_factor, W//downscaling_factor]
        else:
            x = self.patch_partition(x)
            # np_x = x.cpu().detach().numpy()
            y = self.patch_partition(y)
            for regular_block, shifted_block in self.layers:
                x = regular_block(x, y)
                x = shifted_block(x, y)
            return x.permute(0, 3, 1, 2)

if __name__ == '__main__':
    x = torch.ones([1, 1, 64, 64])
    print(x.shape)
    vgg_feat = VGG_Feat()
    x_lv1, x_lv2, x_lv3 = vgg_feat(x)
    print(x_lv1.shape, x_lv2.shape, x_lv3.shape)
