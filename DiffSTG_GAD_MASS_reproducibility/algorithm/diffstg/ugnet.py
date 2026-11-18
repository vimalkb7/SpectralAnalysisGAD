# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from .graph_algo import asym_adj, sym_adj


def TimeEmbedding(timesteps: torch.Tensor, embedding_dim: int):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super().__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        if len(Lk.shape) == 2:
            Lk = Lk.unsqueeze(0)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)


class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, :-self.chomp_size]


class TcnBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation_size=1, droupout=0.0):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation_size
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(3, kernel_size), padding=(1, self.padding), dilation=(1, dilation_size))
        self.chomp = Chomp(self.padding)
        self.drop = nn.Dropout(droupout)
        self.net = nn.Sequential(self.conv, self.chomp, self.drop)
        self.shortcut = nn.Conv2d(c_in, c_out, (1, 1)) if c_in != c_out else None

    def forward(self, x):
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)
        return out + x_skip


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, config, kernel_size=3):
        super().__init__()
        self.tcn1 = TcnBlock(c_in, c_out, kernel_size=kernel_size)
        self.tcn2 = TcnBlock(c_out, c_out, kernel_size=kernel_size)
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1,1))
        self.t_conv = nn.Conv2d(config.d_h, c_out, (1,1))
        self.spatial = SpatialBlock(config.supports_len, c_out, c_out)
        self.norm = nn.LayerNorm([config.V, c_out])

    def forward(self, x, t, supports):
        h = self.tcn1(x)
        h += self.t_conv(t[:, :, None, None])
        h = self.tcn2(h)
        h = self.norm(h.transpose(1,3)).transpose(1,3)
        h = h.transpose(2,3)
        h = self.spatial(h, supports).transpose(2,3)
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        super().__init__(); self.res = ResidualBlock(c_in, c_out, config)
    def forward(self, x, t, supports): return self.res(x, t, supports)


class Downsample(nn.Module):
    def __init__(self, c_in):
        super().__init__(); self.conv = nn.Conv2d(c_in, c_in, (1,3), stride=(1,2), padding=(0,1))
    def forward(self, x, t, supports): return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        super().__init__(); self.res = ResidualBlock(c_in + c_out, c_out, config)
    def forward(self, x, t, supports): return self.res(x, t, supports)


class Upsample(nn.Module):
    def __init__(self, c_in):
        super().__init__(); self.conv = nn.ConvTranspose2d(c_in, c_in, (1,4), (1,2), (0,1))
    def forward(self, x, t, supports): return self.conv(x)


class MiddleBlock(nn.Module):
    def __init__(self, c_in, config):
        super().__init__(); self.res1 = ResidualBlock(c_in, c_in, config); self.res2 = ResidualBlock(c_in, c_in, config)
    def forward(self, x, t, supports): return self.res2(self.res1(x, t, supports), t, supports)


class UGnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_h = config.d_h
        self.T_p, self.T_h = config.T_p, config.T_h
        T, self.F = self.T_p + self.T_h, config.F
        self.n_blocks = config.get('n_blocks', 2)
        self.mas_profile_condition = bool(config.get('mas_profile_condition', False))

        graph_operator = config.get('gnn_graph_operator', 'author_directed')
        if graph_operator == 'author_directed':
            supports_np = np.stack([np.asarray(asym_adj(config.A)), np.asarray(asym_adj(np.transpose(config.A)))]).astype(np.float32)
        elif graph_operator == 'mass_symmetric':
            A = np.maximum(np.asarray(config.A), np.asarray(config.A).T)
            supports_np = np.stack([np.asarray(sym_adj(A), dtype=np.float32)]).astype(np.float32)
        else:
            raise ValueError(f"Unknown gnn_graph_operator={graph_operator}")
        config.supports_len = int(supports_np.shape[0])
        self.register_buffer('supports', torch.from_numpy(supports_np))

        n_resolutions = len(config.channel_multipliers)
        down, in_channels = [], self.d_h
        out_channels = in_channels
        for i in range(n_resolutions):
            out_channels = in_channels * config.channel_multipliers[i]
            for _ in range(self.n_blocks):
                down.append(DownBlock(in_channels, out_channels, config)); in_channels = out_channels
            if i < n_resolutions - 1: down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(out_channels, config)

        up, in_channels = [], out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(self.n_blocks): up.append(UpBlock(in_channels, out_channels, config))
            out_channels = in_channels // config.channel_multipliers[i]
            up.append(UpBlock(in_channels, out_channels, config)); in_channels = out_channels
            if i > 0: up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        self.x_proj = nn.Conv2d(self.F, self.d_h, (1,1))
        self.out = nn.Sequential(nn.Conv2d(self.d_h, self.F, (1,1)), nn.Linear(2*T, T))
        if self.mas_profile_condition:
            self.profile_proj = nn.Sequential(
                nn.Linear(5, self.d_h), nn.SiLU(), nn.Linear(self.d_h, self.d_h)
            )
        else:
            self.profile_proj = None

    def forward(self, x: torch.Tensor, t: torch.Tensor, c, noise_profile=None):
        x_masked, pos_w, pos_d = c
        x = torch.cat((x, x_masked), dim=3)
        x = self.x_proj(x)
        t_emb = TimeEmbedding(t, self.d_h)
        if self.profile_proj is not None and noise_profile is not None:
            t_emb = t_emb + self.profile_proj(noise_profile.float())

        h = [x]
        supports = self.supports
        for m in self.down:
            x = m(x, t_emb, supports); h.append(x)
        x = self.middle(x, t_emb, supports)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t_emb, supports)
            else:
                s = h.pop(); x = torch.cat((x, s), dim=1); x = m(x, t_emb, supports)
        return self.out(x)
