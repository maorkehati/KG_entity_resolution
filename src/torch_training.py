"""Training and inference helpers for structured PyTorch matchers."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.torch_models import StructuredPairDataset, structured_collate_fn


@dataclass
class TorchTrainConfig:
    model_type: str
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    hidden_dim: int = 64
    dropout: float = 0.1
    random_state: int = 42
    device: str = "auto"
    class_weight: bool = True
    field_dim: int = 64
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    y = np.asarray(y).astype(int)
    num_pos = max(int((y == 1).sum()), 1)
    num_neg = int((y == 0).sum())
    return torch.tensor([num_neg / num_pos], dtype=torch.float32)


def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            groups = {k: v.to(device) for k, v in batch["groups"].items()}
            labels = batch["label"].to(device)
            logits = model(groups)
            loss = criterion(logits, labels)
            total += float(loss.item()) * labels.size(0)
            n += labels.size(0)
    return total / max(n, 1)


def _eval_ranking(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def train_torch_matcher(
    model: nn.Module,
    train_groups: dict[str, np.ndarray],
    y_train: np.ndarray,
    valid_groups: dict[str, np.ndarray],
    y_valid: np.ndarray,
    config: TorchTrainConfig,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train with BCEWithLogitsLoss; early stop on validation loss."""
    set_seed(config.random_state)
    device = get_device(config.device)
    model = model.to(device)

    train_ds = StructuredPairDataset(train_groups, y_train)
    valid_ds = StructuredPairDataset(valid_groups, y_valid)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=structured_collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=structured_collate_fn,
    )

    pos_weight = compute_pos_weight(y_train).to(device) if config.class_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    patience_left = config.patience
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            groups = {k: v.to(device) for k, v in batch["groups"].items()}
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(groups)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * labels.size(0)
            n_train += labels.size(0)

        train_loss /= max(n_train, 1)
        valid_loss = _eval_loss(model, valid_loader, criterion, device)
        valid_scores = predict_torch_matcher(
            model, valid_groups, batch_size=config.batch_size, device=str(device)
        )
        rank = _eval_ranking(y_valid, valid_scores)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_auroc": rank["auroc"],
            "valid_auprc": rank["auprc"],
        }
        history.append(row)

        if valid_loss < best_valid_loss - 1e-6:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, {"history": history, "best_valid_loss": best_valid_loss}


@torch.no_grad()
def predict_torch_matcher(
    model: nn.Module,
    groups: dict[str, np.ndarray],
    batch_size: int = 512,
    device: str = "auto",
    return_attention: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return sigmoid probabilities; optionally attention weights [n, num_fields]."""
    dev = get_device(device)
    model = model.to(dev)
    model.eval()

    ds = StructuredPairDataset(groups, labels=None)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=structured_collate_fn
    )

    scores: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    for batch in loader:
        batch_groups = {k: v.to(dev) for k, v in batch["groups"].items()}
        if return_attention and hasattr(model, "forward"):
            out = model(batch_groups, return_attention=True)
            logits, alpha = out
            attentions.append(alpha.cpu().numpy())
        else:
            logits = model(batch_groups)
        probs = torch.sigmoid(logits).cpu().numpy()
        scores.append(probs)

    all_scores = np.concatenate(scores, axis=0)
    if return_attention:
        return all_scores, np.concatenate(attentions, axis=0)
    return all_scores
