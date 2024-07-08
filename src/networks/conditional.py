import torch
import torch.nn.functional as F

from .base import Model
from . import layers


def normalization(norm_fn: str, channels: int, num_groups: int = 1):
    if norm_fn == 'group':
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    elif norm_fn == 'batch':
        return torch.nn.BatchNorm3d(channels)

    elif norm_fn == 'instance':
        return torch.nn.InstanceNorm3d(channels)

    elif norm_fn == 'none':
        return torch.nn.Identity()


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, cemb_channels: int = 512,
                 pre_conv: bool = False, norm_fn='none',
                 fast: bool = False):
        super().__init__()

        if pre_conv:
            self.pre_conv = layers.Conv(in_channels, out_channels, fast=fast)
            in_channels = out_channels
        else:
            self.pre_conv = torch.nn.Identity()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = normalization(norm_fn, in_channels)
        self.act1 = layers.nonlinearity(inplace=False)
        self.conv1 = layers.Conv(in_channels,
                                 out_channels,
                                 fast=fast)
        self.actcemb = layers.nonlinearity(inplace=False)
        self.temb_proj = torch.nn.Linear(cemb_channels,
                                         out_channels)
        self.norm2 = normalization(norm_fn, out_channels)
        self.act2 = layers.nonlinearity(inplace=True)
        self.conv2 = layers.Conv(out_channels,
                                 out_channels,
                                 fast=fast)
        if self.in_channels != self.out_channels:
            self.in_shortcut = layers.FastShortcut(in_channels,
                                                   out_channels)
        else:
            self.in_shortcut = torch.nn.Identity()

        if fast:
            self.add_to = lambda x, y: x.add_(y)
        else:
            self.add_to = lambda x, y: x.add(y)

    def forward(self, x, cemb):
        x = self.pre_conv(x)
        h = x
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.add_to(h, self.temb_proj(self.actcemb(cemb))[:, :, None, None, None])

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        x = self.in_shortcut(x)
        return self.add_to(x, h)


class LocalAttentionBlock(torch.nn.Module):
    def __init__(self, dim: int, heads: int = 1, size: int = 3, fast: bool = False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        assert dim % heads == 0
        self.size = size
        self.ws = size*size*size

        self.scale = (self.dim//self.heads) ** (-.5)

        self.q = torch.nn.Linear(self.dim, self.dim)
        self.k = torch.nn.Linear(self.dim, self.dim)
        self.v = torch.nn.Linear(self.dim, self.dim)
        self.softmax = torch.nn.Softmax(dim=-1)

        pad = int(self.size//2)
        if fast:
            self.add_to = lambda x, y: x.add_(y)

            def padding(x):
                B, D, H, W, C = x.shape
                xp = x.new_zeros((B, D+2*pad, H+2*pad, W+2*pad, C))
                xp[:, pad:-pad, pad:-pad, pad:-pad, :] = x
                return xp
            self.pad = padding
        else:
            self.add_to = lambda x, y: x.add(y)

            self.pad = lambda x: F.pad(x.permute(0, 4, 1, 2, 3), (pad, pad, pad, pad, pad, pad),
                                       mode='replicate').permute(0, 2, 3, 4, 1)

    def _unfold(self, x):
        B, D, H, W, C = x.shape
        x = self.pad(x)
        x = x.unfold(1, self.size, 1)
        x = x.unfold(2, self.size, 1)
        x = x.unfold(3, self.size, 1)  # [B, D, H, W, C, S, S, S]
        x = x.permute(0, 1, 2, 3, 5, 6, 7, 4)  # [B, D, H, W, S, S, S, C]
        x = x.reshape(B, D*H*W, self.ws, self.heads, C//self.heads,).transpose(2, 3)  # [B, DHW, Nh, S**3, C/Nh]
        return x

    def forward(self, x, cemb):
        # transform to channels last
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        B, D, H, W, C = x.shape
        assert C == self.dim

        # compute query, key and value
        q = self.q(x).view(B, D*H*W, self.heads, 1, C//self.heads)  # [B, DHW, Nh, 1, C/Nh]
        k = self._unfold(self.k(x))  # [B, DHW, Nh, S**3, C/Nh]
        v = self._unfold(self.v(x))  # [B, DHW, Nh, S**3, C/Nh]
        # compute local attention
        q = q * self.scale
        attn = torch.einsum("...ij,...kj->...ik", q, k)  # q @ k.T: [B, DHW, Nh, 1, S**3]
        # TODO: maybe add relative position
        attn = self.softmax(attn)
        # compute output
        h = (attn @ v).view(B, D, H, W, C)  # [B, D, H, W, C]
        return self.add_to(x, h).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]


class CondUNetModel(Model):

    def __init__(self, config, fast: bool = False):
        super().__init__(config)

        # conditional embedding
        in_cond_channels = self.config["in_cond_channels"]
        cond_channels = self.config["cond_channels"]
        cond_layers = self.config["cond_layers"]
        cond_linear = [torch.nn.Linear(cond_channels, cond_channels) for _ in range(cond_layers)]
        cond_nonlinear = [layers.nonlinearity(inplace=True) for _ in range(cond_layers)]
        self.cond_emb = torch.nn.Sequential(
            # start with first linear layer
            torch.nn.Linear(in_cond_channels, cond_channels),
            # alternate a linear and nonlinear layer
            *[x for y in zip(cond_nonlinear, cond_linear) for x in y]
        )

        # network architecture
        in_channels = self.config["in_channels"]
        channels = self.config["channels"]
        out_channels = self.config["out_channels"]
        self.fast = fast
        pool_type = self.config.get("pool_type", "avg")
        norm_fn = self.config.get("norm_fn", "none")

        attn_enable = self.config.get("attn_enable", False)
        attn_heads = self.config.get("attn_heads", 4)
        attn_size = self.config.get("attn_size", 3)

        self.conv1 = layers.Conv(in_channels, channels, kernel_size=3, stride=1, fast=self.fast)
        self.head = layers.Conv(channels, out_channels, fast=self.fast)

        if pool_type == "max":
            self.pool = layers.MaxPool2x2x2()
        elif pool_type == "avg":
            self.pool = layers.AvgPool2x2x2()
        else:
            raise RuntimeError(f"unsupported pool_type '{pool_type}'")

        self.up = layers.Interpolate2x2x2()

        self.b01 = torch.nn.ModuleList([
            ResnetBlock(channels, channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            ResnetBlock(channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])
        self.b02 = torch.nn.ModuleList([
            ResnetBlock(2*channels, channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            ResnetBlock(channels, channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])

        self.b11 = torch.nn.ModuleList([
            ResnetBlock(channels, 2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            ResnetBlock(2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])
        self.b12 = torch.nn.ModuleList([
            ResnetBlock(4*channels, 2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            ResnetBlock(2*channels, channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])

        self.b21 = torch.nn.ModuleList([
            ResnetBlock(2*channels, 4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            LocalAttentionBlock(4*channels, heads=attn_heads, size=attn_size, fast=self.fast) if attn_enable
            else layers.Identity(),
            ResnetBlock(4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])
        self.b22 = torch.nn.ModuleList([
            ResnetBlock(8*channels, 4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            LocalAttentionBlock(4*channels, heads=attn_heads, size=attn_size, fast=self.fast) if attn_enable
            else layers.Identity(),
            ResnetBlock(4*channels, 2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])

        self.b41 = torch.nn.ModuleList([
            ResnetBlock(4*channels, 8*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            LocalAttentionBlock(8*channels, heads=attn_heads, size=attn_size, fast=self.fast) if attn_enable
            else layers.Identity(),
            ResnetBlock(8*channels, 4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])

    def forward(self, x, condition, mean=None, std=None):

        x_diff = x[:, :1].clone()
        if self.config.get('normalize_input', False):
            x[:, :1] = (x[:, :1] - mean) / torch.clamp(std, min=1e-6) * self.config.get("ref_std", 1.)

        x0 = self.conv1(x)
        cemb = self.cond_emb(condition)

        for b in self.b01:
            x0 = b(x0, cemb)
        x1 = self.pool(x0)
        for b in self.b11:
            x1 = b(x1, cemb)
        x2 = self.pool(x1)
        for b in self.b21:
            x2 = b(x2, cemb)
        x4 = self.pool(x2)
        for b in self.b41:
            x4 = b(x4, cemb)
        x4 = self.up(x4, output_shape=x2.shape)
        x2 = torch.cat([x2, x4], 1)
        for b in self.b22:
            x2 = b(x2, cemb)
        x2 = self.up(x2, output_shape=x1.shape)
        x1 = torch.cat([x1, x2], 1)
        for b in self.b12:
            x1 = b(x1, cemb)
        x1 = self.up(x1, output_shape=x0.shape)
        x0 = torch.cat([x1, x0], 1)
        for b in self.b02:
            x0 = b(x0, cemb)

        residual = self.head(x0)
        if self.config.get('normalize_input', False):
            residual = residual * torch.clamp(std, min=1e-6) / self.config.get("ref_std", 1.)

        # residual connection
        return x_diff + residual

    def extra_repr(self):
        return f"fast={self.fast}"


class GatedAddition(torch.nn.Module):
    def __init__(self, in1_ch: int, in2_ch: int, out_ch: int, fast: bool = False):
        super().__init__()

        self.conv_g1 = layers.FastConv(in1_ch + in2_ch, out_ch)
        self.conv_w1 = layers.FastConv(in1_ch, out_ch)

        self.conv_g2 = layers.FastConv(in1_ch + in2_ch, out_ch)
        self.conv_w2 = layers.FastConv(in2_ch, out_ch)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        m1 = torch.sigmoid(self.conv_g1(x))
        m2 = torch.sigmoid(self.conv_g2(x))

        return m1 * self.conv_w1(x1) + m2 * self.conv_w2(x2)


class CondGatedUNetModel(Model):

    def __init__(self, config, fast: bool = False):
        super().__init__(config)

        # conditional embedding
        in_cond_channels = self.config["in_cond_channels"]
        cond_channels = self.config["cond_channels"]
        cond_layers = self.config["cond_layers"]
        cond_linear = [torch.nn.Linear(cond_channels, cond_channels) for _ in range(cond_layers)]
        cond_nonlinear = [layers.nonlinearity(inplace=True) for _ in range(cond_layers)]
        self.cond_emb = torch.nn.Sequential(
            # start with first linear layer
            torch.nn.Linear(in_cond_channels, cond_channels),
            # alternate a linear and nonlinear layer
            *[x for y in zip(cond_nonlinear, cond_linear) for x in y]
        )

        # network architecture
        in_channels = self.config["in_channels"]
        self.in_channels = in_channels
        channels = self.config["channels"]
        out_channels = self.config["out_channels"]
        self.fast = fast
        pool_type = self.config.get("pool_type", "avg")
        norm_fn = self.config.get("norm_fn", "none")

        attn_enable = self.config.get("attn_enable", False)
        attn_heads = self.config.get("attn_heads", 4)
        attn_size = self.config.get("attn_size", 3)

        self.conv1 = layers.Conv(in_channels, channels, kernel_size=3, stride=1, fast=self.fast)
        self.head = layers.Conv(channels, out_channels, fast=self.fast)

        if pool_type == "max":
            self.pool = layers.MaxPool2x2x2()
        elif pool_type == "avg":
            self.pool = layers.AvgPool2x2x2()
        elif pool_type == "conv":
            self.pool = layers.Downsample2x2x2()
        else:
            raise RuntimeError(f"unsupported pool_type '{pool_type}'")

        self.up = layers.Interpolate2x2x2()

        self.b01 = torch.nn.ModuleList([
            ResnetBlock(in_channels+channels, channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            ResnetBlock(channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])
        self.merge0 = GatedAddition(in_channels+channels, channels, channels)
        self.b02 = torch.nn.ModuleList([
            ResnetBlock(channels, channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            ResnetBlock(channels, channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])

        self.b11 = torch.nn.ModuleList([
            ResnetBlock(in_channels+channels, 2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            ResnetBlock(2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])
        self.merge1 = GatedAddition(in_channels+2*channels, 2*channels, 2*channels)
        self.b12 = torch.nn.ModuleList([
            ResnetBlock(2*channels, 2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            ResnetBlock(2*channels, channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])

        self.b21 = torch.nn.ModuleList([
            ResnetBlock(in_channels+2*channels, 4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            LocalAttentionBlock(4*channels, heads=attn_heads, size=attn_size, fast=self.fast) if attn_enable
            else layers.Identity(),
            ResnetBlock(4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])
        self.merge2 = GatedAddition(in_channels+4*channels, 4*channels, 4*channels)
        self.b22 = torch.nn.ModuleList([
            ResnetBlock(4*channels, 4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            LocalAttentionBlock(4*channels, heads=attn_heads, size=attn_size, fast=self.fast) if attn_enable
            else layers.Identity(),
            ResnetBlock(4*channels, 2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])

        self.b41 = torch.nn.ModuleList([
            ResnetBlock(in_channels+4*channels, 8*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
            LocalAttentionBlock(8*channels, heads=attn_heads, size=attn_size, fast=self.fast) if attn_enable
            else layers.Identity(),
            ResnetBlock(8*channels, 4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
        ])

    def forward(self, x, condition, mean=None, std=None):

        if self.config.get('normalize_input', False):
            x[:, :1] = (x[:, :1] - mean) / torch.clamp(std, min=1e-6) * self.config.get("ref_std", 1.)

        x0 = self.conv1(x)
        # push the input channels forward
        x0 = torch.cat([x, x0], 1)
        cemb = self.cond_emb(condition)

        x0_res = x0.clone()
        for b in self.b01:
            x0_res = b(x0_res, cemb)
        x0 = torch.cat([x0[:, :self.in_channels], x0_res], 1)
        x1 = self.pool(x0)

        x1_res = x1.clone()
        for b in self.b11:
            x1_res = b(x1_res, cemb)
        x1 = torch.cat([x1[:, :self.in_channels], x1_res], 1)
        x2 = self.pool(x1)

        x2_res = x2.clone()
        for b in self.b21:
            x2_res = b(x2_res, cemb)
        x2 = torch.cat([x2[:, :self.in_channels], x2_res], 1)
        x4 = self.pool(x2)

        for b in self.b41:
            x4 = b(x4, cemb)

        x4 = self.up(x4, output_shape=x2.shape)
        x2 = self.merge2(x2, x4)
        for b in self.b22:
            x2 = b(x2, cemb)

        x2 = self.up(x2, output_shape=x1.shape)
        x1 = self.merge1(x1, x2)
        for b in self.b12:
            x1 = b(x1, cemb)

        x1 = self.up(x1, output_shape=x0.shape)
        x0 = self.merge0(x0, x1)
        for b in self.b02:
            x0 = b(x0, cemb)

        res = self.head(x0)
        if self.config.get('normalize_input', False):
            res = res * torch.clamp(std, min=1e-6) / self.config.get("ref_std", 1.)

        return res

    def extra_repr(self):
        return f"fast={self.fast}"
