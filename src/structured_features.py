"""Dense structured field-group features for pairwise neural matchers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.features import safe_float, token_containment, token_jaccard
from src.symbolic import (
    STRONG_VARIANT_MODIFIERS,
    extract_model_like_tokens,
    extract_quantities,
    extract_variant_modifiers,
    token_set,
)

_WHITESPACE_RE = re.compile(r"\s+")
_LEN_SCALE = 500.0
_COUNT_SCALE = 10.0


@dataclass
class StructuredFeatureConfig:
    include_title: bool = True
    include_brand: bool = True
    include_description: bool = True
    include_price: bool = True
    include_identifier: bool = True
    include_quantity: bool = True
    max_description_chars: int = 1000


def normalize_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = str(x).strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def _norm_len(n: int) -> float:
    return float(min(n / _LEN_SCALE, 1.0))


def _norm_count(n: int) -> float:
    return float(min(n / _COUNT_SCALE, 1.0))


def _identifier_text(row: pd.Series, side: str, max_desc: int) -> str:
    title = normalize_text(row.get(f"{side}_title", ""))
    desc = normalize_text(row.get(f"{side}_description", ""))[:max_desc]
    return f"{title} {desc}".strip()


def _quantity_text(row: pd.Series, side: str, max_desc: int) -> str:
    return _identifier_text(row, side, max_desc)


def _model_token_conflict_flag(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    if left_tokens & right_tokens:
        return 0.0
    if 1 <= len(left_tokens) <= 3 and 1 <= len(right_tokens) <= 3:
        return 1.0
    return 0.0


def _variant_conflict_flag(left_title: str, right_title: str) -> float:
    left_mod = extract_variant_modifiers(left_title)
    right_mod = extract_variant_modifiers(right_title)
    ta, tb = token_set(left_title), token_set(right_title)
    if not ta or not tb:
        return 0.0
    shared_ratio = len(ta & tb) / min(len(ta), len(tb))
    if shared_ratio < 0.4:
        return 0.0
    left_strong = left_mod & STRONG_VARIANT_MODIFIERS
    right_strong = right_mod & STRONG_VARIANT_MODIFIERS
    return 1.0 if bool(left_strong ^ right_strong) else 0.0


def _quantity_stats(row: pd.Series, max_desc: int) -> dict[str, float]:
    left_q = extract_quantities(_quantity_text(row, "left", max_desc))
    right_q = extract_quantities(_quantity_text(row, "right", max_desc))
    by_family_left: dict[str, list] = {}
    by_family_right: dict[str, list] = {}
    for q in left_q:
        by_family_left.setdefault(q.family, []).append(q)
    for q in right_q:
        by_family_right.setdefault(q.family, []).append(q)

    match_count = 0
    conflict_count = 0
    ambiguous = 0
    families = set(by_family_left) | set(by_family_right)
    for family in families:
        lq = by_family_left.get(family, [])
        rq = by_family_right.get(family, [])
        if len(lq) > 1 or len(rq) > 1:
            ambiguous += 1
            continue
        if len(lq) == 1 and len(rq) == 1:
            if abs(lq[0].value - rq[0].value) < 1e-6:
                match_count += 1
            else:
                conflict_count += 1

    return {
        "left_quantity_count": _norm_count(len(left_q)),
        "right_quantity_count": _norm_count(len(right_q)),
        "quantity_match_count": _norm_count(match_count),
        "quantity_conflict_count": _norm_count(conflict_count),
        "quantity_conflict_flag": float(conflict_count > 0),
        "ambiguous_quantity_flag": float(ambiguous > 0),
    }


def _title_group(row: pd.Series) -> list[float]:
    lt = normalize_text(row.get("left_title", ""))
    rt = normalize_text(row.get("right_title", ""))
    ll, rl = len(lt), len(rt)
    denom = max(ll, rl, 1)
    return [
        fuzz.token_set_ratio(lt, rt) / 100.0,
        fuzz.ratio(lt, rt) / 100.0,
        token_jaccard(lt, rt),
        token_containment(lt, rt),
        _norm_len(ll),
        _norm_len(rl),
        float(abs(ll - rl) / denom),
        float(bool(lt) and bool(rt)),
    ]


def _brand_group(row: pd.Series) -> list[float]:
    lb = normalize_text(row.get("left_brand", ""))
    rb = normalize_text(row.get("right_brand", ""))
    left_miss = float(not lb)
    right_miss = float(not rb)
    both_miss = float(not lb and not rb)
    exact = float(bool(lb and rb and lb == rb))
    return [
        exact,
        fuzz.ratio(lb, rb) / 100.0 if lb and rb else 0.0,
        fuzz.token_set_ratio(lb, rb) / 100.0 if lb and rb else 0.0,
        float(bool(lb) and bool(rb)),
        left_miss,
        right_miss,
        both_miss,
    ]


def _description_group(row: pd.Series, max_chars: int) -> list[float]:
    ld = normalize_text(row.get("left_description", ""))[:max_chars]
    rd = normalize_text(row.get("right_description", ""))[:max_chars]
    ll, rl = len(ld), len(rd)
    denom = max(ll, rl, 1)
    return [
        fuzz.token_set_ratio(ld, rd) / 100.0 if ld or rd else 0.0,
        token_jaccard(ld, rd),
        token_containment(ld, rd),
        _norm_len(ll),
        _norm_len(rl),
        float(abs(ll - rl) / denom),
        float(bool(ld) and bool(rd)),
    ]


def _price_group(row: pd.Series) -> list[float]:
    lp = safe_float(row.get("left_price"))
    rp = safe_float(row.get("right_price"))
    lc = normalize_text(row.get("left_price_currency", ""))
    rc = normalize_text(row.get("right_price_currency", ""))
    left_miss = float(pd.isna(lp))
    right_miss = float(pd.isna(rp))
    both_present = float(not left_miss and not right_miss)
    if both_present:
        abs_diff = abs(lp - rp)
        denom = max(abs(lp), abs(rp), 1e-6)
        rel_diff = min(abs_diff / denom, 10.0) / 10.0
        abs_norm = min(abs_diff / 1000.0, 1.0)
    else:
        abs_norm = 0.0
        rel_diff = 0.0
    same_currency = float(bool(lc and rc and lc == rc))
    currency_both = float(bool(lc) and bool(rc))
    return [
        both_present,
        left_miss,
        right_miss,
        abs_norm,
        rel_diff,
        same_currency,
        currency_both,
    ]


def _identifier_group(row: pd.Series, max_desc: int) -> list[float]:
    lt = _identifier_text(row, "left", max_desc)
    rt = _identifier_text(row, "right", max_desc)
    left_tokens = extract_model_like_tokens(lt)
    right_tokens = extract_model_like_tokens(rt)
    left_var = extract_variant_modifiers(normalize_text(row.get("left_title", "")))
    right_var = extract_variant_modifiers(normalize_text(row.get("right_title", "")))
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    jacc = len(overlap) / len(union) if union else 0.0
    return [
        _norm_count(len(left_tokens)),
        _norm_count(len(right_tokens)),
        _norm_count(len(overlap)),
        float(jacc),
        _model_token_conflict_flag(left_tokens, right_tokens),
        _norm_count(len(left_var)),
        _norm_count(len(right_var)),
        _norm_count(len(left_var & right_var)),
        _variant_conflict_flag(
            normalize_text(row.get("left_title", "")),
            normalize_text(row.get("right_title", "")),
        ),
    ]


def build_structured_feature_groups(
    df: pd.DataFrame,
    config: StructuredFeatureConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return field-group name -> float32 matrix [n_samples, group_dim]."""
    cfg = config or StructuredFeatureConfig()
    builders: list[tuple[str, Any]] = []
    if cfg.include_title:
        builders.append(("title", lambda r: _title_group(r)))
    if cfg.include_brand:
        builders.append(("brand", lambda r: _brand_group(r)))
    if cfg.include_description:
        builders.append(
            ("description", lambda r: _description_group(r, cfg.max_description_chars))
        )
    if cfg.include_price:
        builders.append(("price", lambda r: _price_group(r)))
    if cfg.include_identifier:
        builders.append(
            ("identifier", lambda r: _identifier_group(r, cfg.max_description_chars))
        )
    if cfg.include_quantity:
        builders.append(
            ("quantity", lambda r: list(_quantity_stats(r, cfg.max_description_chars).values()))
        )

    groups: dict[str, np.ndarray] = {}
    for name, fn in builders:
        rows = [fn(df.iloc[i]) for i in range(len(df))]
        groups[name] = np.asarray(rows, dtype=np.float32)
    return groups


def structured_groups_to_tensor_dict(
    groups: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Convert numpy groups to torch tensors (lazy import)."""
    import torch

    return {k: torch.from_numpy(v) for k, v in groups.items()}


def group_dims(groups: dict[str, np.ndarray]) -> dict[str, int]:
    return {k: int(v.shape[1]) for k, v in groups.items()}
