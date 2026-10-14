# networks/netmamba.py
#
# NetMamba adattata per traffic classification su biflussi TON-IoT
# Input: tensore (B, num_pkts, num_fields) — es. (64, 10, 6) con fields=[PL,IAT,DIR,WIN,TTL,FLG]
# Compatibile con main.py: net(pretrained, num_pkts, num_fields, out_features_size, scale)

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

# ─────────────────────────────────────────────────────────────────────────────
# Importazioni opzionali (kernel CUDA di mamba_ssm); fallback puro-PyTorch
# ─────────────────────────────────────────────────────────────────────────────
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn = mamba_inner_fn = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm = layer_norm_fn = rms_norm_fn = None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("pe", self._build_pe(max_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _extend(self, new_len: int):
        if new_len > self.pe.size(1):
            self.pe = self._build_pe(new_len, self.d_model).to(self.pe.device, self.pe.dtype)

    def forward(self, x):
        """
        Args:
            x: tensor (B, L, d_model)
        """
        L = x.size(1)
        if L > self.pe.size(1):
            self._extend(L)

        pe = self.pe[:, :L].to(dtype=x.dtype, device=x.device)
        x = x * math.sqrt(self.d_model) + pe
        return self.dropout(x)

# ─────────────────────────────────────────────────────────────────────────────
# Mamba block (identico al tuo mamba_simple.py, versione unidirezionale)
# ─────────────────────────────────────────────────────────────────────────────
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        self.d_inner  = int(self.expand * self.d_model)
        self.dt_rank  = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx     = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (B, L, D)  →  (B, L, D)"""
        batch, seqlen, dim = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        if self.use_fast_path and mamba_inner_fn is not None and inference_params is None:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,   # B (None = computed from input)
                None,   # C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            # Fallback puro-PyTorch
            x, z = xz.chunk(2, dim=1)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            if selective_scan_fn is not None:
                y = selective_scan_fn(
                    x, dt, A, B, C, self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )
            else:
                # Fallback puro-PyTorch (nessuna dipendenza CUDA)
                dt_soft = F.softplus(dt + self.dt_proj.bias.float().unsqueeze(-1))  # (B, d_inner, L)
                y = self._selective_scan_ref(x, dt_soft, A, B, C, self.D, z)

            y   = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)

        return out

    # ── fallback scan (nessun kernel CUDA) ─────────────────────────────────
    @staticmethod
    def _selective_scan_ref(u, delta, A, B, C, D, z):
        """Implementazione di riferimento in PyTorch puro (slow ma corretta)."""
        # u, delta: (B, d_inner, L)
        # A: (d_inner, d_state)
        # B, C: (B, d_state, L)
        # D: (d_inner,)
        # z: (B, d_inner, L)
        B_sz, d_inner, L = u.shape
        d_state = A.shape[1]

        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))  # (B, d, L, n)
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)   # (B, d, L, n)

        x = torch.zeros(B_sz, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            ys.append(y)

        y = torch.stack(ys, dim=2)  # (B, d_inner, L)
        y = y + u * D.unsqueeze(-1)
        y = y * F.silu(z)
        return y


# ─────────────────────────────────────────────────────────────────────────────
# Block: LayerNorm + Mamba + residual (come nel tuo mamba_simple.py)
# ─────────────────────────────────────────────────────────────────────────────
class MambaBlock(nn.Module):
    def __init__(self, dim, layer_idx=None):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.mixer = Mamba(d_model=dim, layer_idx=layer_idx)

    def forward(self, x: Tensor, residual: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        residual = (x + residual) if residual is not None else x
        hidden   = self.norm(residual)
        out      = self.mixer(hidden)
        return out, residual


# ─────────────────────────────────────────────────────────────────────────────
# NetMamba  — backbone per traffic classification su biflussi
# ─────────────────────────────────────────────────────────────────────────────
class NetMamba(nn.Module):
    """
    Backbone per traffic classification su biflussi.

    Parametri accettati da main.py:
        pretrained       – ignorato (nessun pretraining disponibile)
        num_pkts         – lunghezza della sequenza (es. 10)
        num_fields       – numero di feature per pacchetto (es. 6 = PL,IAT,DIR,WIN,TTL,FLG)
        out_features_size – dim. dell'embedding finale (None → usa d_model)
        scale            – fattore di scala per d_model (default 1.0)

    Architettura:
        [input embedding]  num_fields → d_model   (Linear)
        [pos embedding]    num_pkts × d_model
        [Mamba blocks]     depth blocchi SSM
        [pool]             mean pooling sulla sequenza
        [proj]             d_model → out_features_size
        [head]             aggiunta dall'esterno tramite add_head()
    """

    def __init__(
        self,
        pretrained: bool = False,
        num_pkts: int = 10,
        num_fields: int = 6,
        out_features_size: Optional[int] = None,
        scale: float = 1.0,
        d_model: int = 128,
        depth: int = 4,
        d_state: int = 16,
    ):
        super().__init__()

        # Scala d_model con il parametro --scale
        d_model = max(32, int(d_model * scale))

        self.num_pkts  = num_pkts
        self.num_fields = num_fields
        self.d_model   = d_model
        self.out_features_size = out_features_size if out_features_size is not None else d_model

        # 1) Input projection: (B, T, num_fields) → (B, T, d_model)
        self.input_proj = nn.Linear(num_fields, d_model)

        # 2) Positional embedding apprendibile
        self.pos_embed = PositionalEncoding(d_model=d_model, max_len=num_pkts)

        # 3) Stack di blocchi Mamba
        self.blocks = nn.ModuleList([
            MambaBlock(dim=d_model, layer_idx=i) for i in range(depth)
        ])

        # 4) Norma finale
        self.norm_f = nn.LayerNorm(d_model)

        # 5) Proiezione verso out_features_size
        self.fc1 = nn.Linear(d_model, self.out_features_size)
        self.fc = nn.Linear(self.out_features_size, 10)  # placeholder, sostituito da LLL_Net
        self.head_var = 'fc'

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x):
        out = F.gelu(self.extract_features(x))
        out = self.fc(out)
        return out
    
    def extract_features(self, x):

        # Il loader passa (B, 1, num_pkts, num_fields) → rimuovi la dim canale
        if x.dim() == 4:
            x = x.squeeze(1)  # → (B, num_pkts, num_fields)
        
        # Porta tutto a (B, T=num_pkts, C=num_fields)
        if x.dim() == 3:
            if x.shape[1] == self.num_fields and x.shape[2] == self.num_pkts:
                x = x.permute(0, 2, 1)
        elif x.dim() == 2:
            x = x.view(x.size(0), self.num_pkts, self.num_fields)
    
        x = x.float()
    
        x = self.input_proj(x)      # (B, T, d_model)
        x = self.pos_embed(x)       # positional encoding
    
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual)
    
        x = self.norm_f(x + residual if residual is not None else x)
        x = x.mean(dim=1)           # (B, d_model)
        x = self.fc1(x)             # (B, out_features_size)
        return x
