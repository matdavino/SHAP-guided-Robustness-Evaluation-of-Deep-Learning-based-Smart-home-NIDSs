import math
import torch
import torch.nn.functional as F
from torch import nn


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
    

class BiflowTransformer(nn.Module):
    """Single-layer Transformer-based classifier with input shape (n_batches, 1, num_pkts, num_fields)."""

    def __init__(self, num_pkts, num_fields, num_classes=10, dropout=0.1, **kwargs):
        super().__init__()

        d_model = kwargs.get('d_model', 128)  # Dimensionality of the embedding
        nhead = kwargs.get('nhead', 4)  # Number of attention heads
        dim_feedforward = kwargs.get('dim_feedforward', 512)  # Feedforward network dimension
        out_features_size = kwargs.get('out_features_size', 200)
        if out_features_size is None:
            out_features_size = 200

        self.in_proj = nn.Linear(num_fields, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # self.pre_norm = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=num_pkts + 1, dropout=dropout)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            dim_feedforward=dim_feedforward, 
            activation='gelu',
            batch_first=True,
            # norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=4) # TODO: provare 2
        print(d_model, out_features_size)
        self.fc1 = nn.Linear(d_model, out_features_size)
        self.fc = nn.Linear(out_features_size, num_classes)
        self.head_var = 'fc'

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.gelu(self.extract_features(x))
        out = self.fc(out)
        return out

    def extract_features(self, x):
        # Input x is of shape (n_batches, 1, num_pkts, num_fields)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, d_model)
        x = x.squeeze(1)
        B, _, _ = x.shape

        # Build pad mask 
        flag = x[..., 0]
        pad_mask = (flag == 0)  # (B, L) boolean
        pad_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1
        )  # (B, L+1) boolean

        x = self.in_proj(x)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_pkts+1, d_model)
        # Positional encoding
        # x = self.pre_norm(x)
        x = self.positional_encoding(x)
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)  # (B, num_pkts+1, d_model)
        cls_output = x[:, 0]  # (B, d_model)
        x = self.fc1(cls_output)  # (B, out_features_size)
        return x
