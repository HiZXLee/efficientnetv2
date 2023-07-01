import torch
from torch import nn


class ConvBlock(nn.Module):
    """A class for model block consists of Convolution, Batch Normalization and Activation.

    Args:
        n_in (int): Number of input channels.
        n_out (int): Number of output channels.
        k_size (int, optional): Kernel size. Defaults to 3.
        stride (int, optional): Stride. Defaults to 1.
        padding (int, optional): Padding. Defaults to 0.
        groups (int, optional): _description_. Defaults to 1.
        act (bool, optional): Apply activation or not. Defaults to True.
        bn (bool, optional): Apply batch normalization or not. Defaults to True.
        bias (bool, optional): With bias or not. Defaults to False.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True,
        bn: bool = True,
        bias: bool = False,
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            n_in,
            n_out,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.activation = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x


class SEBLock(nn.Module):
    """Squeeze and excitation block.
    Args:
        n_in (int): Number of input channels.
        r (float, optional): Reduction factor. Defaults to 4.
    """

    def __init__(self, n_in: int, r: float = 4):
        super(SEBLock, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(n_in, n_in // r, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(n_in // r, n_in, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)

        return x * y


class StochasticDepth(nn.Module):
    """A class for stochastic depth. Use to shrink the depth of network during training (for testing, the depth remains unchanged).

    Args:
        survival_prob (float, optional): Probability to keep the block. Defaults to 0.8.
    """

    def __init__(
        self,
        survival_prob: float = 0.8,
    ):
        super(StochasticDepth, self).__init__()
        self.p = survival_prob

    def forward(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p

        return torch.div(x, self.p) * binary_tensor


class MBConvN(nn.Module):
    """A class of Inverted Residual Block.

    Args:
        n_in (int): Number of input channels.
        n_out (int): Number of output channels.
        k_size (int, optional): Kernel size. Defaults to 3.
        stride (int, optional): Stride. Defaults to 1.
        expansion_factor (int, optional): Expansion factor for efficientnet. Defaults to 4.
        reduction_factor (int, optional): Reduction factor for SE block. Defaults to 4.
        survival_prob (float, optional): Probability to keep the block. Defaults to 0.8.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        k_size: int = 3,
        stride: int = 1,
        expansion_factor: int = 4,
        reduction_factor: int = 4,
        survival_prob: float = 0.8,
    ):
        super(MBConvN, self).__init__()

        expanded_dim = int(expansion_factor * n_in)
        padding = (k_size - 1) // 2

        self.use_residual = (n_in == n_out) and (stride == 1)
        self.expand = (
            nn.Identity()
            if (expansion_factor == 1)
            else ConvBlock(n_in, expanded_dim, k_size=1)
        )
        self.depthwise_conv = ConvBlock(
            expanded_dim,
            expanded_dim,
            k_size,
            stride=stride,
            padding=padding,
            groups=expanded_dim,
        )
        self.se = SEBLock(expanded_dim, reduction_factor)
        self.drop_layers = StochasticDepth(survival_prob)
        self.pointwise_conv = ConvBlock(expanded_dim, n_out, k_size=1, act=False)

    def forward(self, x):
        residual = x.clone()
        x = self.expand(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)

        if self.use_residual:
            x = self.drop_layers(x)
            x += residual

        return x


class FusedMBConvN(nn.Module):
    """A class of FusedMBConv. To replace the depthwise 3x3 convolution and expansion 1x1 convolution in MBConv with a regular 3x3 convolution.

    Args:
        n_in (int): Number of input channels.
        n_out (int): Number of output channels.
        k_size (int, optional): Kernel size. Defaults to 3.
        stride (int, optional): Stride. Defaults to 1.
        expansion_factor (int, optional): Expansion factor for efficientnet. Defaults to 4.
        survival_prob (float, optional): Probability to keep the block. Defaults to 0.8.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        k_size: int = 3,
        stride: int = 1,
        expansion_factor: int = 4,
        survival_prob=0.8,
    ):
        super(FusedMBConvN, self).__init__()

        expanded_dim = int(expansion_factor * n_in)
        padding = (k_size - 1) // 2

        self.use_residual = (n_in == n_out) and (stride == 1)
        self.conv = ConvBlock(
            n_in, expanded_dim, k_size, stride=stride, padding=padding, groups=1
        )

        self.drop_layers = StochasticDepth(survival_prob)
        self.pointwise_conv = (
            nn.Identity()
            if (expansion_factor == 1)
            else ConvBlock(expanded_dim, n_out, k_size=1, act=False)
        )

    def forward(self, x):
        residual = x.clone()
        x = self.conv(x)
        x = self.pointwise_conv(x)

        if self.use_residual:
            x = self.drop_layers(x)
            x += residual

        return x
