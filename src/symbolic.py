"""Symbolic product-identity constraints for pairwise entity resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

ACCESSORY_TERMS = {
    "case", "cover", "charger", "adapter", "strap", "band",
    "screen protector", "protector", "battery", "replacement",
    "mount", "cable", "sleeve", "stand", "holder", "dock",
    "remote", "bag", "pouch", "skin", "shell", "lens cap",
    "cartridge", "toner",
}

MAIN_PRODUCT_TERMS = {
    "phone", "smartphone", "iphone", "galaxy",
    "camera", "laptop", "notebook", "tablet", "watch",
    "printer", "monitor", "tv", "television",
    "headphones", "speaker", "console", "router",
    "drive", "ssd", "hdd", "flash drive", "keyboard", "mouse",
}

VARIANT_MODIFIERS = {
    "pro", "max", "mini", "plus", "ultra", "lite", "air", "se", "xl",
}

STRONG_VARIANT_MODIFIERS = {"pro", "ultra", "mini", "max", "plus", "lite"}

BUNDLE_TERMS = {
    "bundle", "kit", "combo", "set", "with", "includes",
    "including", "pack", "starter kit", "lens kit",
}

COLOR_TERMS = {
    "black", "white", "red", "blue", "green", "yellow",
    "silver", "gold", "gray", "grey", "pink", "purple",
    "orange", "brown",
}

CATEGORY_KEYWORDS = {
    "phone": {"phone", "smartphone", "iphone", "galaxy"},
    "phone_accessory": {"case", "cover", "screen protector", "charger", "cable"},
    "camera": {"camera", "dslr", "mirrorless"},
    "camera_accessory": {"lens", "battery", "tripod", "lens cap"},
    "computer": {"laptop", "notebook", "desktop", "pc"},
    "storage": {"ssd", "hdd", "flash drive", "usb drive", "memory card"},
    "audio": {"headphones", "earbuds", "speaker", "soundbar"},
    "display": {"monitor", "tv", "television"},
    "printer": {"printer"},
    "printer_supply": {"toner", "cartridge", "ink"},
    "watch": {"watch", "smartwatch"},
    "watch_accessory": {"strap", "band"},
}

BRAND_ALIASES = {
    "hewlett packard": "hp",
    "hp": "hp",
    "lg electronics": "lg",
    "lg": "lg",
    "apple inc": "apple",
    "apple": "apple",
    "samsung electronics": "samsung",
    "samsung": "samsung",
    "sandisk": "sandisk",
    "san disk": "sandisk",
}

BRAND_SUFFIXES = {
    "inc", "ltd", "llc", "corp", "corporation", "co", "company", "gmbh", "plc",
}

_MODEL_EXCLUDE = re.compile(
    r"^(?:\d+gb|\d+tb|\d+mb|\d+inch|\d+in|\d+mm|\d+cm|4k|1080p|usb\d|wifi\d)$",
    re.I,
)
_MODEL_TOKEN_RE = re.compile(r"\b[a-z]*\d[a-z0-9\-]{2,}\b", re.I)

_QUANTITY_PATTERNS: list[tuple[str, str, str]] = [
    (r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b", "storage", "unit"),
    (r"(\d+(?:\.\d+)?)\s*(inch|inches|in)\b", "length", "unit"),
    (r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", "length", "unit"),
    (r"(\d+(?:\.\d+)?)\s*(kg|g)\b", "weight", "unit"),
    (r"(\d+(?:\.\d+)?)\s*(ml|l)\b", "volume", "unit"),
    (r"(\d+(?:\.\d+)?)\s*(w|watt|watts)\b", "power", "unit"),
    (r"(\d+)\s*(pack|pcs|pieces|count|ct)\b", "count", "unit"),
]

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class SymbolicConfig:
    use_brand_conflict: bool = True
    use_accessory_conflict: bool = True
    use_model_token_conflict: bool = True
    use_variant_modifier_conflict: bool = True
    use_capacity_size_conflict: bool = True
    use_bundle_conflict: bool = True
    use_price_conflict: bool = True
    use_color_conflict: bool = True
    use_category_keyword_conflict: bool = True

    price_conflict_as_invalid: bool = False
    color_conflict_as_invalid: bool = False
    bundle_conflict_as_invalid: bool = False
    category_conflict_as_invalid: bool = False

    same_currency_price_rel_diff_threshold: float = 0.75
    require_positive_evidence_for_valid: bool = True
    max_text_chars_for_rules: int = 1000


@dataclass
class SymbolicResult:
    symbolic_status: str
    violated_constraints: list[str] = field(default_factory=list)
    positive_evidence: list[str] = field(default_factory=list)
    uncertain_reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    raw: str
    family: str


def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = str(x).strip().lower()
    return _WHITESPACE_RE.sub(" ", text)


def is_missing_text(x) -> bool:
    return not normalize_text(x)


def normalize_brand(x) -> str:
    text = normalize_text(x)
    text = re.sub(r"[^\w\s]", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t not in BRAND_SUFFIXES]
    return " ".join(tokens).strip()


def canonicalize_brand(x) -> str:
    norm = normalize_brand(x)
    if not norm:
        return ""
    return BRAND_ALIASES.get(norm, norm)


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def safe_float(x) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    cleaned = re.sub(r"[^\d.\-]", "", str(x))
    if not cleaned:
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _pair_text(row: Any, max_chars: int) -> str:
    parts = []
    for col in ("left_title", "right_title", "left_description", "right_description"):
        if hasattr(row, "get"):
            val = row.get(col, "")
        else:
            val = getattr(row, col, "") if col in row.index else ""
        parts.append(normalize_text(val))
    text = " ".join(p for p in parts if p)
    return text[:max_chars]


def _side_text(row: Any, side: str, max_chars: int) -> str:
    title = normalize_text(row.get(f"{side}_title", "") if hasattr(row, "get") else row[f"{side}_title"])
    desc = normalize_text(row.get(f"{side}_description", "") if hasattr(row, "get") else row.get(f"{side}_description", ""))
    combined = f"{title} {desc}".strip()
    return combined[:max_chars]


def check_brand_conflict(row) -> tuple[str | None, dict]:
    left_raw = row.get("left_brand", "") if hasattr(row, "get") else row["left_brand"]
    right_raw = row.get("right_brand", "") if hasattr(row, "get") else row["right_brand"]
    diag = {
        "left_brand_raw": left_raw,
        "right_brand_raw": right_raw,
        "left_brand_canonical": canonicalize_brand(left_raw),
        "right_brand_canonical": canonicalize_brand(right_raw),
    }
    if is_missing_text(left_raw) or is_missing_text(right_raw):
        return None, {**diag, "uncertain": ["brand_missing"], "brand_both_present": False}
    diag["brand_both_present"] = True
    if diag["left_brand_canonical"] == diag["right_brand_canonical"]:
        return None, {**diag, "positive": ["brand_match"], "brand_match": True}
    return "brand_conflict", {**diag, "brand_match": False}


def detect_accessory_terms(text: str) -> set[str]:
    norm = normalize_text(text)
    found = set()
    for term in sorted(ACCESSORY_TERMS, key=len, reverse=True):
        if term in norm:
            found.add(term)
    return found


def detect_main_product_terms(text: str) -> set[str]:
    norm = normalize_text(text)
    found = set()
    for term in sorted(MAIN_PRODUCT_TERMS, key=len, reverse=True):
        if term in norm:
            found.add(term)
    return found


def check_accessory_conflict(row) -> tuple[str | None, dict]:
    left_text = _side_text(row, "left", 500)
    right_text = _side_text(row, "right", 500)
    left_acc = detect_accessory_terms(left_text)
    right_acc = detect_accessory_terms(right_text)
    left_main = detect_main_product_terms(left_text)
    right_main = detect_main_product_terms(right_text)
    diag = {
        "left_accessory_terms": sorted(left_acc),
        "right_accessory_terms": sorted(right_acc),
        "left_main_terms": sorted(left_main),
        "right_main_terms": sorted(right_main),
    }
    left_is_acc = bool(left_acc) and not left_main
    right_is_acc = bool(right_acc) and not right_main
    left_is_main = bool(left_main) and not left_acc
    right_is_main = bool(right_main) and not right_acc
    if (left_is_acc and right_is_main) or (right_is_acc and left_is_main):
        return "accessory_main_product_conflict", diag
    return None, diag


def _normalize_model_token(tok: str) -> str:
    return tok.lower().replace(" ", "").replace("-", "")


def extract_model_like_tokens(text: str) -> set[str]:
    norm = normalize_text(text)
    tokens = set()
    for match in _MODEL_TOKEN_RE.findall(norm):
        compact = _normalize_model_token(match)
        if len(compact) < 4:
            continue
        if _MODEL_EXCLUDE.match(compact):
            continue
        if compact.isdigit():
            continue
        if not any(c.isalpha() for c in compact) or not any(c.isdigit() for c in compact):
            continue
        tokens.add(compact)
    return tokens


def check_model_token_conflict(row) -> tuple[str | None, dict]:
    left_text = _side_text(row, "left", 800)
    right_text = _side_text(row, "right", 800)
    left_tokens = extract_model_like_tokens(left_text)
    right_tokens = extract_model_like_tokens(right_text)
    overlap = left_tokens & right_tokens
    diag = {
        "left_model_tokens": sorted(left_tokens),
        "right_model_tokens": sorted(right_tokens),
        "model_token_overlap": sorted(overlap),
    }
    if not left_tokens or not right_tokens:
        return None, diag
    if overlap:
        return None, {**diag, "positive": ["model_token_overlap"]}
    if 1 <= len(left_tokens) <= 3 and 1 <= len(right_tokens) <= 3:
        return "model_token_conflict", diag
    return None, {**diag, "uncertain": ["model_tokens_no_overlap"]}


def extract_variant_modifiers(text: str) -> set[str]:
    tokens = token_set(text)
    return {t for t in tokens if t in VARIANT_MODIFIERS}


def _title_overlap_ratio(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def check_variant_modifier_conflict(row) -> tuple[str | None, dict]:
    left_title = normalize_text(row.get("left_title", ""))
    right_title = normalize_text(row.get("right_title", ""))
    left_mod = extract_variant_modifiers(left_title)
    right_mod = extract_variant_modifiers(right_title)
    shared_ratio = _title_overlap_ratio(left_title, right_title)
    diag = {
        "left_variant_modifiers": sorted(left_mod),
        "right_variant_modifiers": sorted(right_mod),
        "shared_title_token_ratio": shared_ratio,
    }
    if shared_ratio < 0.4:
        return None, diag
    left_strong = left_mod & STRONG_VARIANT_MODIFIERS
    right_strong = right_mod & STRONG_VARIANT_MODIFIERS
    if left_strong ^ right_strong:
        return "variant_modifier_conflict", diag
    return None, diag


def _normalize_quantity(value: float, unit: str, family: str) -> float:
    u = unit.lower()
    if family == "storage":
        if u == "tb":
            return value * 1024
        if u == "mb":
            return value / 1024
        return value
    if family == "length":
        if u in ("mm",):
            return value / 25.4
        if u in ("cm",):
            return value / 2.54
        return value
    if family == "weight":
        if u == "kg":
            return value * 1000
        return value
    if family == "volume":
        if u == "l":
            return value * 1000
        return value
    return value


def extract_quantities(text: str) -> list[Quantity]:
    norm = normalize_text(text)
    quantities: list[Quantity] = []
    for pattern, family, _ in _QUANTITY_PATTERNS:
        for match in re.finditer(pattern, norm):
            value = float(match.group(1))
            unit = match.group(2).lower()
            raw = match.group(0)
            quantities.append(
                Quantity(
                    value=_normalize_quantity(value, unit, family),
                    unit=unit,
                    raw=raw,
                    family=family,
                )
            )
    return quantities


def check_capacity_size_conflict(row) -> tuple[str | None, dict]:
    left_q = extract_quantities(_side_text(row, "left", 500))
    right_q = extract_quantities(_side_text(row, "right", 500))
    diag: dict[str, Any] = {
        "left_quantities": [q.raw for q in left_q],
        "right_quantities": [q.raw for q in right_q],
        "quantity_matches": [],
        "quantity_conflicts": [],
    }
    by_family_left: dict[str, list[Quantity]] = {}
    by_family_right: dict[str, list[Quantity]] = {}
    for q in left_q:
        by_family_left.setdefault(q.family, []).append(q)
    for q in right_q:
        by_family_right.setdefault(q.family, []).append(q)

    uncertain: list[str] = []
    for family in set(by_family_left) | set(by_family_right):
        lq = by_family_left.get(family, [])
        rq = by_family_right.get(family, [])
        if len(lq) > 1 or len(rq) > 1:
            uncertain.append("multiple_quantities_ambiguous")
            continue
        if len(lq) == 1 and len(rq) == 1:
            if abs(lq[0].value - rq[0].value) < 1e-6:
                diag["quantity_matches"].append(family)
            else:
                diag["quantity_conflicts"].append(family)
                return "quantity_conflict", diag
    if diag["quantity_matches"]:
        return None, {**diag, "positive": ["quantity_match"]}
    if uncertain:
        return None, {**diag, "uncertain": uncertain}
    return None, diag


def detect_bundle_terms(text: str) -> set[str]:
    norm = normalize_text(text)
    return {t for t in BUNDLE_TERMS if t in norm}


def check_bundle_conflict(row, config: SymbolicConfig) -> tuple[str | None, dict]:
    left_b = detect_bundle_terms(_side_text(row, "left", 500))
    right_b = detect_bundle_terms(_side_text(row, "right", 500))
    diag = {"left_bundle_terms": sorted(left_b), "right_bundle_terms": sorted(right_b)}
    if bool(left_b) ^ bool(right_b):
        if config.bundle_conflict_as_invalid:
            return "bundle_or_kit_conflict", diag
        return None, {**diag, "uncertain": ["bundle_or_kit_mismatch"]}
    return None, diag


def check_price_conflict(row, config: SymbolicConfig) -> tuple[str | None, dict]:
    left_p = safe_float(row.get("left_price"))
    right_p = safe_float(row.get("right_price"))
    left_c = normalize_text(row.get("left_price_currency", ""))
    right_c = normalize_text(row.get("right_price_currency", ""))
    diag = {
        "left_price": left_p,
        "right_price": right_p,
        "left_price_currency": left_c,
        "right_price_currency": right_c,
    }
    if pd.isna(left_p) or pd.isna(right_p):
        return None, {**diag, "uncertain": ["price_missing"]}
    if not left_c or not right_c or left_c != right_c:
        return None, {**diag, "uncertain": ["currency_missing_or_mismatch"]}
    denom = max(abs(left_p), abs(right_p), 1e-6)
    rel_diff = abs(left_p - right_p) / denom
    diag["price_rel_diff"] = rel_diff
    if rel_diff <= config.same_currency_price_rel_diff_threshold:
        return None, {**diag, "positive": ["price_compatible"]}
    if config.price_conflict_as_invalid:
        return "same_currency_price_conflict", diag
    return None, {**diag, "uncertain": ["same_currency_price_conflict"]}


def extract_colors(text: str) -> set[str]:
    tokens = token_set(text)
    return tokens & COLOR_TERMS


def check_color_conflict(row, config: SymbolicConfig) -> tuple[str | None, dict]:
    left_c = extract_colors(_side_text(row, "left", 400))
    right_c = extract_colors(_side_text(row, "right", 400))
    diag = {"left_colors": sorted(left_c), "right_colors": sorted(right_c)}
    if left_c and right_c and not (left_c & right_c):
        if config.color_conflict_as_invalid:
            return "color_conflict", diag
        return None, {**diag, "uncertain": ["color_conflict"]}
    if left_c & right_c:
        return None, {**diag, "positive": ["color_match"]}
    return None, diag


def infer_category_keywords(text: str) -> set[str]:
    norm = normalize_text(text)
    found = set()
    for category, terms in CATEGORY_KEYWORDS.items():
        if any(term in norm for term in terms):
            found.add(category)
    return found


_INCOMPATIBLE_CATEGORIES = {
    frozenset({"phone", "printer"}),
    frozenset({"phone", "printer_supply"}),
    frozenset({"camera", "printer"}),
    frozenset({"watch", "printer"}),
    frozenset({"phone", "camera"}),
    frozenset({"phone_accessory", "printer"}),
    frozenset({"storage", "printer_supply"}),
}


def check_category_keyword_conflict(row, config: SymbolicConfig) -> tuple[str | None, dict]:
    left_cat = infer_category_keywords(_side_text(row, "left", 500))
    right_cat = infer_category_keywords(_side_text(row, "right", 500))
    diag = {"left_categories": sorted(left_cat), "right_categories": sorted(right_cat)}
    if not left_cat or not right_cat:
        return None, diag
    for a in left_cat:
        for b in right_cat:
            pair = frozenset({a, b})
            if pair in _INCOMPATIBLE_CATEGORIES or (
                "phone" in pair and "printer" in pair
            ) or (
                ("phone_accessory" in pair and "phone" not in pair)
                and ("printer" in pair or "printer_supply" in pair)
            ):
                if config.category_conflict_as_invalid:
                    return "category_keyword_conflict", diag
                return None, {**diag, "uncertain": ["category_keyword_mismatch"]}
    if left_cat & right_cat:
        return None, {**diag, "positive": ["category_keyword_overlap"]}
    return None, diag


class ProductConstraintChecker:
    """Ontology-guided product merge validator."""

    def __init__(self, config: SymbolicConfig | None = None):
        self.config = config or SymbolicConfig()

    def check_row(self, row) -> SymbolicResult:
        cfg = self.config
        violations: list[str] = []
        positive: list[str] = []
        uncertain: list[str] = []
        diagnostics: dict[str, Any] = {}

        rules = [
            (cfg.use_brand_conflict, check_brand_conflict),
            (cfg.use_accessory_conflict, check_accessory_conflict),
            (cfg.use_model_token_conflict, check_model_token_conflict),
            (cfg.use_variant_modifier_conflict, check_variant_modifier_conflict),
            (cfg.use_capacity_size_conflict, check_capacity_size_conflict),
            (cfg.use_bundle_conflict, check_bundle_conflict),
            (cfg.use_price_conflict, check_price_conflict),
            (cfg.use_color_conflict, check_color_conflict),
            (cfg.use_category_keyword_conflict, check_category_keyword_conflict),
        ]

        for enabled, fn in rules:
            if not enabled:
                continue
            if fn in (check_bundle_conflict, check_price_conflict, check_color_conflict, check_category_keyword_conflict):
                violation, extra = fn(row, cfg)
            else:
                violation, extra = fn(row)
            rule_name = fn.__name__
            diagnostics[f"rule_{rule_name}"] = {
                k: v for k, v in extra.items() if k not in ("positive", "uncertain")
            }
            if violation:
                violations.append(violation)
            for p in extra.get("positive", []):
                if p not in positive:
                    positive.append(p)
            for u in extra.get("uncertain", []):
                if u not in uncertain:
                    uncertain.append(u)

        if violations:
            status = "invalid"
        elif positive:
            status = "valid"
        elif not cfg.require_positive_evidence_for_valid:
            status = "valid"
        else:
            status = "uncertain"

        return SymbolicResult(
            symbolic_status=status,
            violated_constraints=violations,
            positive_evidence=positive,
            uncertain_reasons=uncertain,
            diagnostics=diagnostics,
        )

    def check_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            result = self.check_row(row)
            rows.append(
                {
                    "pair_id": row.get("pair_id"),
                    "symbolic_status": result.symbolic_status,
                    "violated_constraints": result.violated_constraints,
                    "positive_evidence": result.positive_evidence,
                    "uncertain_reasons": result.uncertain_reasons,
                    "num_violated_constraints": len(result.violated_constraints),
                    "num_positive_evidence": len(result.positive_evidence),
                    "diagnostics_json": json.dumps(result.diagnostics, default=str),
                }
            )
        return pd.DataFrame(rows)


HARD_CONSTRAINTS_CONSERVATIVE = [
    "brand_conflict",
    "accessory_main_product_conflict",
    "model_token_conflict",
    "variant_modifier_conflict",
    "quantity_conflict",
]

HARD_CONSTRAINTS_MODERATE = [
    *HARD_CONSTRAINTS_CONSERVATIVE,
    "bundle_or_kit_conflict",
    "category_keyword_conflict",
]

DIAGNOSTIC_CONSTRAINTS = [
    "same_currency_price_conflict",
    "color_conflict",
]

ALL_ANALYZED_CONSTRAINTS = [
    *HARD_CONSTRAINTS_MODERATE,
    *DIAGNOSTIC_CONSTRAINTS,
]


def hard_constraints_for_profile(profile: str) -> list[str]:
    if profile == "conservative":
        return list(HARD_CONSTRAINTS_CONSERVATIVE)
    if profile == "moderate":
        return list(HARD_CONSTRAINTS_MODERATE)
    raise ValueError(f"Unknown symbolic profile: {profile!r}. Use 'conservative' or 'moderate'.")


def parse_symbolic_list(value: Any) -> list[str]:
    """Parse violated_constraints / positive_evidence from list or JSON string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ("[]", "nan"):
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return []


def recompute_symbolic_status(
    violated: list[str],
    positive: list[str],
    *,
    require_positive_evidence: bool = True,
) -> str:
    if violated:
        return "invalid"
    if positive:
        return "valid"
    if require_positive_evidence:
        return "uncertain"
    return "valid"


def remove_constraint_from_symbolic_results(
    symbolic_df: pd.DataFrame,
    constraint: str,
    *,
    require_positive_evidence: bool = True,
) -> pd.DataFrame:
    """Leave-one-out: drop one hard veto and recompute symbolic_status."""
    out = symbolic_df.copy()
    violations_col: list[list[str]] = []
    status_col: list[str] = []
    for _, row in out.iterrows():
        violated = parse_symbolic_list(row.get("violated_constraints"))
        positive = parse_symbolic_list(row.get("positive_evidence"))
        filtered = [v for v in violated if v != constraint]
        violations_col.append(filtered)
        status_col.append(
            recompute_symbolic_status(
                filtered,
                positive,
                require_positive_evidence=require_positive_evidence,
            )
        )
    out["violated_constraints"] = violations_col
    out["symbolic_status"] = status_col
    out["num_violated_constraints"] = [len(v) for v in violations_col]
    return out


def make_symbolic_config(profile: str) -> SymbolicConfig:
    """Build SymbolicConfig for sweep profiles: conservative or moderate."""
    base = SymbolicConfig(
        use_brand_conflict=True,
        use_accessory_conflict=True,
        use_model_token_conflict=True,
        use_variant_modifier_conflict=True,
        use_capacity_size_conflict=True,
        use_bundle_conflict=True,
        use_price_conflict=True,
        use_color_conflict=True,
        use_category_keyword_conflict=True,
        price_conflict_as_invalid=False,
        color_conflict_as_invalid=False,
        bundle_conflict_as_invalid=False,
        category_conflict_as_invalid=False,
    )
    if profile == "conservative":
        return base
    if profile == "moderate":
        return replace(
            base,
            bundle_conflict_as_invalid=True,
            category_conflict_as_invalid=True,
        )
    raise ValueError(f"Unknown symbolic profile: {profile!r}. Use 'conservative' or 'moderate'.")


def lists_to_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert list columns to JSON strings for CSV export."""
    out = df.copy()
    for col in ("violated_constraints", "positive_evidence", "uncertain_reasons"):
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
    return out


def explode_constraint_counts(
    df: pd.DataFrame,
    column: str,
    split: str,
    kind: str,
) -> pd.DataFrame:
    """Count occurrences of constraint items per split."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for val in df[column]:
        if isinstance(val, str):
            try:
                items = json.loads(val)
            except json.JSONDecodeError:
                items = []
        elif isinstance(val, list):
            items = val
        else:
            items = []
        for item in items:
            counter[str(item)] += 1
    return pd.DataFrame(
        [{"split": split, "kind": kind, "item": k, "count": v} for k, v in counter.items()]
    )
