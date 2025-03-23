import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from KANConv import KAN_Convolutional_Layer

class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation (SE) block enhances feature representation
    by adaptively recalibrating channel-wise feature responses.
    """
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        reduced_ch = max(1, int(in_ch * se_ratio))
        self.fc1 = nn.utils.weight_norm(nn.Conv2d(in_ch, reduced_ch, 1))
        self.fc2 = nn.utils.weight_norm(nn.Conv2d(reduced_ch, in_ch, 1))

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.silu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

class Involution(nn.Module):
    """
    Implementation of `Involution: Inverting the Inherence of Convolution for Visual Recognition`.
    """
    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3, stride=1, reduction_ratio=2):
        super().__init__()
        channels_reduced = max(1, in_channels // reduction_ratio)
        padding = kernel_size // 2

        self.reduce = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(in_channels, channels_reduced, 1)),
            nn.BatchNorm2d(channels_reduced),
            nn.ReLU(inplace=True))

        self.span = nn.utils.weight_norm(nn.Conv2d(channels_reduced, kernel_size * kernel_size * groups, 1))
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        self.resampling = None if in_channels == out_channels else nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, 1))

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, input_tensor):
        # print(f"Involution Input: {input_tensor.shape}")
        _, _, height, width = input_tensor.size()
        if self.stride > 1:
            out_size = lambda x: (x + 2 * self.padding - self.kernel_size) // self.stride + 1
            height, width = out_size(height), out_size(width)
        uf_x = rearrange(self.unfold(input_tensor), 'b (g d k j) (h w) -> b g d (k j) h w',
                         g=self.groups, k=self.kernel_size, j=self.kernel_size, h=height, w=width)

        if self.stride > 1:
            input_tensor = F.adaptive_avg_pool2d(input_tensor, (height, width))
        kernel = rearrange(self.span(self.reduce(input_tensor)), 'b (k j g) h w -> b g (k j) h w',
                           k=self.kernel_size, j=self.kernel_size)

        out = rearrange(torch.einsum('bgdxhw, bgxhw -> bgdhw', uf_x, kernel), 'b g d h w -> b (g d) h w')
        if self.resampling:
            out = self.resampling(out)

        # print(f"Involution Output: {out.shape}")
        return out.contiguous()

class ConvInvolutionBlock(nn.Module):
    """
    This block combines both convolution and involution to leverage their strengths.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, expansion=1):
        super().__init__()
        hidden_dim = int(in_channels * expansion)
        self.expand = nn.utils.weight_norm(nn.Conv2d(in_channels, hidden_dim, 1)) if expansion > 1 else nn.Identity()
        self.expand_bn = nn.BatchNorm2d(hidden_dim) if expansion > 1 else nn.Identity()
        self.expand_act = nn.SiLU(inplace=True) if expansion > 1 else nn.Identity()

        self.conv = KAN_Convolutional_Layer(in_channels=hidden_dim,
                                            out_channels=hidden_dim,
                                            kernel_size=(kernel_size,kernel_size),
                                            stride=(stride,stride),
                                            padding=(padding,padding),
                                            dilation=(dilation,dilation),
                                            spline_order=3)
        self.conv_bn = nn.BatchNorm2d(hidden_dim)
        # self.conv_act = nn.SiLU(inplace=True)

        self.involution = Involution(hidden_dim, out_channels, kernel_size=kernel_size, stride=1)
        self.inv_bn = nn.BatchNorm2d(out_channels)
        self.inv_act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        # print(f"ConvInvolutionBlock Input: {x.shape}")
        x = self.expand(x)
        x = self.expand_bn(x)
        x = self.expand_act(x)
        # print(f"After Expansion: {x.shape}")

        x = self.conv(x)
        x = self.conv_bn(x)
        # x = self.conv_act(x)
        # print(f"After Conv: {x.shape}")

        x = self.involution(x)
        x = self.inv_bn(x)
        x = self.inv_act(x)
        # print(f"After Involution: {x.shape}")

        return x

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, expansion, se_ratio, num_repeats=1, dropout_rate=0.2):
        super().__init__()
        layers = []
        for _ in range(num_repeats):
            layers.append(ConvInvolutionBlock(in_ch, out_ch, kernel_size, stride, kernel_size//2, dilation, expansion))
            in_ch = out_ch
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            # print(f"MBConv Input: {x.shape}")
            x = block(x)
            # print(f"MBConv Output: {x.shape}")
        return x

class EfficientNetV2S_WithInvolution(nn.Module):
    def __init__(self, num_classes=1486):
        super().__init__()
        self.stem = ConvInvolutionBlock(3, 24, 3, stride=2, padding=1, dilation=1, expansion=1)
        self.blocks = nn.Sequential(
            MBConv(24, 48, 3, 2, 1, 4, 0.0, num_repeats=5),
            MBConv(48, 64, 3, 2, 1, 4, 0.25, num_repeats=5),
            MBConv(64, 128, 3, 2, 1, 4, 0.25, num_repeats=5),
            MBConv(128, 160, 3, 1, 1, 6, 0.25, num_repeats=3),
            MBConv(160, 256, 3, 2, 1, 6, 0.25, num_repeats=1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x