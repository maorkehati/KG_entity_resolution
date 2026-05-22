"""Lightweight PyTorch matchers over structured field groups."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class StructuredPairDataset(Dataset):
    def __init__(
        self,
        feature_groups: dict[str, np.ndarray],
        labels: np.ndarray | None = None,
    ):
        self.field_names = sorted(feature_groups.keys())
        self.groups = {k: torch.from_numpy(feature_groups[k].astype(np.float32)) for k in self.field_names}
        self.labels = None if labels is None else torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(next(iter(self.groups.values())))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "groups": {k: self.groups[k][idx] for k in self.field_names},
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def structured_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    field_names = sorted(batch[0]["groups"].keys())
    collated: dict[str, Any] = {
        "groups": {
            k: torch.stack([b["groups"][k] for b in batch], dim=0)
            for k in field_names
        },
    }
    if "label" in batch[0]:
        collated["label"] = torch.stack([b["label"] for b in batch], dim=0)
    return collated


class _MatchHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class FieldAttentionMatcher(nn.Module):
    """Field encoders + learned attention over field groups."""

    def __init__(
        self,
        group_dims: dict[str, int],
        hidden_dim: int = 64,
        field_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.field_names = sorted(group_dims.keys())
        self.field_dim = field_dim
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(group_dims[name], field_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(field_dim, field_dim),
                    nn.ReLU(),
                )
                for name in self.field_names
            }
        )
        self.attn_w = nn.Linear(field_dim, field_dim)
        self.attn_v = nn.Linear(field_dim, 1, bias=False)
        self.head = _MatchHead(field_dim, hidden_dim, dropout)

    def forward(
        self,
        groups: dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = []
        for name in self.field_names:
            encoded.append(self.encoders[name](groups[name]))
        h_stack = torch.stack(encoded, dim=1)
        attn_h = torch.tanh(self.attn_w(h_stack))
        scores = self.attn_v(attn_h).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        h_pair = (alpha.unsqueeze(-1) * h_stack).sum(dim=1)
        logits = self.head(h_pair)
        if return_attention:
            return logits, alpha
        return logits


class StructuredTransformerMatcher(nn.Module):
    """Small TransformerEncoder over field-evidence tokens."""

    def __init__(
        self,
        group_dims: dict[str, int],
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.field_names = sorted(group_dims.keys())
        self.use_cls_token = use_cls_token
        self.model_dim = model_dim
        self.projections = nn.ModuleDict(
            {name: nn.Linear(group_dims[name], model_dim) for name in self.field_names}
        )
        self.field_embeddings = nn.Embedding(len(self.field_names), model_dim)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = _MatchHead(model_dim, hidden_dim, dropout)

    def forward(self, groups: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for i, name in enumerate(self.field_names):
            z = self.projections[name](groups[name]) + self.field_embeddings.weight[i]
            tokens.append(z.unsqueeze(1))
        seq = torch.cat(tokens, dim=1)
        if self.use_cls_token:
            batch = seq.size(0)
            cls = self.cls_token.expand(batch, -1, -1)
            seq = torch.cat([cls, seq], dim=1)
        encoded = self.encoder(seq)
        if self.use_cls_token:
            h_pair = encoded[:, 0, :]
        else:
            h_pair = encoded.mean(dim=1)
        return self.head(h_pair)
