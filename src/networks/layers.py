import numpy as np
import torch
import torch.nn.functional as F


class Downsample2x2x2(torch.nn.Module):
    def __init__(self, fast=False):
        super().__init__()

        self.fast = fast

        # create the convolution kernel
        np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
        np_k2 = np_k @ np_k.T
        np_k = (np_k @ np_k2.reshape(1, -1)).reshape(5, 5, 5)
        np_k /= np_k.sum()
        np_k = np.reshape(np_k, (1, 1, 5, 5, 5))
        self.register_buffer('blur', torch.from_numpy(np_k))

    def forward(self, x):
        kernel = self.blur
        pad = int(kernel.shape[-1])//2
        N, C, D, H, W = x.shape
        assert x.stride()[-1] == 1
        if self.fast:
            x = torch.nn.functional.conv3d(x.view(N*C, 1, D, H, W), kernel, stride=2, padding=pad)
        else:
            x = torch.nn.functional.pad(x.view(N*C, 1, D, H, W), (pad, pad, pad, pad, pad, pad), 'replicate')
            # compute the convolution
            x = torch.nn.functional.conv3d(x, kernel, stride=2)
        return x.view(N, C, (D+1)//2, (H+1)//2, (W+1)//2)


class AvgPool2x2x2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.avg_pool3d(x, kernel_size=3, stride=2, padding=1)


class MaxPool2x2x2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.max_pool3d(x, kernel_size=3, stride=2, padding=1)


class Interpolate2x2x2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, output_shape=None):
        return torch.nn.functional.interpolate(x, size=output_shape[2:], mode="trilinear", align_corners=False)


class Upsample2x2x2(torch.nn.Module):
    def __init__(self, fast=False):
        super().__init__()

        self.fast = fast

        # create the convolution kernel
        np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
        np_k2 = np_k @ np_k.T
        np_k = (np_k @ np_k2.reshape(1, -1)).reshape(5, 5, 5)
        np_k /= np_k.sum()
        np_k *= 8
        np_k = np.reshape(np_k, (1, 1, 5, 5, 5))
        self.register_buffer('blur', torch.from_numpy(np_k))

    def forward(self, x, output_shape=None):
        # determine the amount of padding
        if output_shape is not None:
            output_padding = (
                output_shape[2] - ((x.shape[2]-1)*2+1),
                output_shape[3] - ((x.shape[3]-1)*2+1),
                output_shape[4] - ((x.shape[4]-1)*2+1)
            )
        else:
            output_padding = 0

        kernel = self.blur
        pad = int(kernel.shape[-1])//4

        N, C, D, H, W = x.shape
        assert x.stride()[-1] == 1
        if self.fast:
            x = torch.nn.functional.conv_transpose3d(x.view(N*C, 1, D, H, W), kernel, stride=2, padding=2*pad,
                                                     output_padding=output_padding)
        else:
            x = torch.nn.functional.pad(x.view(N*C, 1, D, H, W), (pad, pad, pad, pad, pad, pad), 'replicate')
            # compute the convolution
            x = torch.nn.functional.conv_transpose3d(x, kernel, stride=2, padding=4*pad, output_padding=output_padding)
        return x.view(N, C, *output_shape[2:])


class Conv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 bias: bool = True, fast: bool = False):
        super().__init__()

        self.c = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                                 padding=kernel_size//2,
                                 padding_mode='zeros' if fast or (kernel_size == 1) else 'replicate',
                                 bias=bias)

    def forward(self, x):
        return self.c(x)


class FastShortcut(torch.nn.Module):
    """ Implementation of a 1x1x1 convolution using a Linear layer.
        Required due to missing CPU implementation of pytorch!
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        if x.is_cuda:
            # avoid permuting of tensor on GPU
            x = F.conv3d(
                x,
                weight=self.linear.weight.view(self.linear.out_features, self.linear.in_features, 1, 1, 1),
                bias=self.linear.bias
            )
            return x
        else:
            # convert to channels last
            x = x.permute(0, 2, 3, 4, 1)
            x = self.linear(x)
            # convert back to channels first
            return x.permute(0, 4, 1, 2, 3)


class FastConv(torch.nn.Module):
    """ Reimplementation of a 1x1x1 convolution.
        Required due to missing CPU implementation of pytorch!
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.c = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if x.is_cuda:
            # avoid permuting of tensor on GPU
            x = F.conv3d(x, weight=self.c.weight, bias=self.c.bias)
            return x
        else:
            # convert to channels last
            x = x.permute(0, 2, 3, 4, 1)
            x = torch.matmul(x, self.c.weight[..., 0, 0, 0].T) + self.c.bias[None, None, None, None, :]
            # convert back to channels first
            return x.permute(0, 4, 1, 2, 3)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, cemb):
        return x


def nonlinearity(inplace=False):
    return torch.nn.SiLU(inplace=inplace)


class GuidedUpsample(torch.nn.Module):
    def __init__(self, in_channels: int, fast: bool = False):
        super().__init__()

        self.mask = torch.nn.Sequential(
            Conv(in_channels, 2*in_channels, kernel_size=3, fast=fast),
            nonlinearity(),
            Conv(2*in_channels, 8*27, kernel_size=1),
        )

    def _unfold(self, x):
        B, C, D, H, W = x.shape
        # pad with zeros
        pad = 1
        size = 3
        x = F.pad(x, (pad, pad, pad, pad, pad, pad), mode='replicate')
        x = x.unfold(2, size, 1)
        x = x.unfold(3, size, 1)
        x = x.unfold(4, size, 1)  # [B, C, D, H, W, S, S, S]
        return x.reshape(B, C, D, H, W, 27)

    def forward(self, x, features):
        B, C, D, H, W = x.shape

        mask = self.mask(features).view(B, 1, 2, 2, 2, 27, D, H, W)
        mask = mask.permute(0, 1, 6, 2, 7, 3, 8, 4, 5).contiguous()  # [B, 1, D, 2, H, 2, W, 2, 27]
        mask = torch.softmax(mask, dim=-1)

        x = self._unfold(x)  # [B, C, D, H, W, 27]
        x = x.view(B, C, D, 1, H, 1, W, 1, 27)

        x = torch.sum(mask * x, dim=-1)
        return x.view(B, C, 2*D, 2*H, 2*W)
