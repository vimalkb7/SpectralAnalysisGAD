from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn

from graphs import adjacency_to_laplacian


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    if t.ndim != 1:
        raise ValueError(f"Expected t to have shape [B], got {tuple(t.shape)}")
    half = dim // 2
    if half == 0:
        return t[:, None]
    freqs = torch.exp(
        -math.log(float(max_period)) * torch.arange(0, half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class GenericScoreMixin:
    has_spectral_gains: bool = False

    def score_from_prediction(
        self,
        xt: torch.Tensor,
        V: torch.Tensor,
        t: torch.Tensor,
        a: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        xhat = self.forward(xt, V, t)
        xt_tilde = xt @ V
        xhat_tilde = xhat @ V
        return (a * xhat_tilde - xt_tilde) / (nu + 1e-12)



def _scaled_laplacian_from_dense(L: torch.Tensor) -> torch.Tensor:
    evals = torch.linalg.eigvalsh(L)
    lmax = torch.clamp(evals.max(), min=torch.tensor(1.0, device=L.device, dtype=L.dtype))
    I = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
    return (2.0 / lmax) * L - I



def _dense_laplacian_from_adj(W: torch.Tensor) -> torch.Tensor:
    W_np = W.detach().cpu().numpy().astype(np.float64)
    L_np = adjacency_to_laplacian(W_np, normalized=False)
    return torch.tensor(L_np, device=W.device, dtype=W.dtype)


class ChebFiLMResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        time_hidden_dim: int,
        cheb_order: int = 3,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cheb_order = int(cheb_order)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, dtype=dtype) for _ in range(self.cheb_order + 1)])
        self.time_to_alpha = nn.Sequential(
            nn.Linear(time_hidden_dim, time_hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, self.cheb_order + 1, dtype=dtype),
        )
        self.time_to_bias = nn.Linear(time_hidden_dim, hidden_dim, dtype=dtype)
        self.time_to_film = nn.Linear(time_hidden_dim, 2 * hidden_dim, dtype=dtype)
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def _cheb_basis(self, x: torch.Tensor, L_scaled: torch.Tensor) -> list[torch.Tensor]:
        basis = [x]
        if self.cheb_order == 0:
            return basis
        T1 = torch.einsum("ij,bjc->bic", L_scaled, x)
        basis.append(T1)
        for _ in range(2, self.cheb_order + 1):
            Tk = 2.0 * torch.einsum("ij,bjc->bic", L_scaled, basis[-1]) - basis[-2]
            basis.append(Tk)
        return basis

    def forward(self, h: torch.Tensor, L_scaled: torch.Tensor, t_h: torch.Tensor) -> torch.Tensor:
        alpha = self.time_to_alpha(t_h)  # [B, K+1]
        bias = self.time_to_bias(t_h)[:, None, :]
        film_scale, film_shift = torch.chunk(self.time_to_film(t_h), 2, dim=-1)
        film_scale = film_scale[:, None, :]
        film_shift = film_shift[:, None, :]

        out = torch.zeros_like(h)
        for k, Tk in enumerate(self._cheb_basis(h, L_scaled)):
            out = out + alpha[:, k].view(-1, 1, 1) * self.linears[k](Tk)
        out = out + bias
        out = (1.0 + film_scale) * out + film_shift
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return h + out


class TimeConditionedGCF(GenericScoreMixin, nn.Module):
    """Time-conditioned polynomial graph filter denoiser.

    This is a dense Chebyshev-style graph filter network intended for fixed-graph
    snapshot denoising x_t -> x_0. It matches the phase-1 setting where each
    timestamp is one scalar graph signal sample.
    """

    def __init__(
        self,
        W: Optional[torch.Tensor] = None,
        L: Optional[torch.Tensor] = None,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_embed_dim: int = 64,
        cheb_order: int = 3,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if L is None:
            if W is None:
                raise ValueError("Provide either adjacency W or Laplacian L.")
            W = W.to(dtype=dtype)
            L = _dense_laplacian_from_adj(W)
        else:
            L = L.to(dtype=dtype)
        if W is None:
            W = torch.zeros_like(L)

        self.time_embed_dim = int(time_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.cheb_order = int(cheb_order)

        self.register_buffer("laplacian", L)
        self.register_buffer("laplacian_scaled", _scaled_laplacian_from_dense(L))

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, dtype=dtype),
        )
        self.input_proj = nn.Linear(1, self.hidden_dim, dtype=dtype)
        self.blocks = nn.ModuleList([
            ChebFiLMResidualBlock(
                hidden_dim=self.hidden_dim,
                time_hidden_dim=self.hidden_dim,
                cheb_order=self.cheb_order,
                dropout=dropout,
                use_layernorm=use_layernorm,
                dtype=dtype,
            )
            for _ in range(max(self.num_layers, 1))
        ])
        self.out_norm = nn.LayerNorm(self.hidden_dim) if use_layernorm else nn.Identity()
        self.out_proj = nn.Linear(self.hidden_dim, 1, dtype=dtype)

    def forward(self, xt: torch.Tensor, V: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del V
        t_emb = sinusoidal_time_embedding(t.to(dtype=xt.dtype), self.time_embed_dim)
        t_h = self.time_mlp(t_emb)
        h = self.input_proj(xt.unsqueeze(-1))
        h = h + t_h[:, None, :]
        for block in self.blocks:
            h = block(h, self.laplacian_scaled, t_h)
        h = self.out_norm(h)
        h = torch.nn.functional.silu(h)
        return self.out_proj(h).squeeze(-1)


# Backward-compatible alias used by the original scripts.
TimeConditionedGNN = TimeConditionedGCF
