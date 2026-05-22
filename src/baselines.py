"""Naive and lexical baseline scorers for pairwise entity resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s./%+\-]")
_CAPACITY_RE = re.compile(r"(\d+)\s*(gb|tb|mb)\b", re.I)


@dataclass
class BaselineConfig:
    name: str
    threshold_objective: str = "f1"
    random_state: int = 42
    max_features: int = 50000
    min_df: int = 2
    max_df: float = 0.95
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)


def normalize_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = str(x).strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _CAPACITY_RE.sub(r"\1\2", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_brand(x: Any) -> str:
    return normalize_text(x)


def safe_float(x: Any) -> float:
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


def serialize_record(row: pd.Series, side: str) -> str:
    parts: list[str] = []
    for field, tag in (
        ("title", "title"),
        ("brand", "brand"),
        ("description", "description"),
        ("price", "price"),
        ("price_currency", "currency"),
    ):
        col = f"{side}_{field}"
        if col not in row.index:
            continue
        val = row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        text = normalize_text(val)
        if text:
            parts.append(f"{tag}: {text}")
    return " ".join(parts)


def get_left_right_texts(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    left_texts: list[str] = []
    right_texts: list[str] = []
    for _, row in df.iterrows():
        if "left_text" in df.columns and pd.notna(row.get("left_text")):
            left = normalize_text(row["left_text"])
        else:
            left = serialize_record(row, "left")
        if "right_text" in df.columns and pd.notna(row.get("right_text")):
            right = normalize_text(row["right_text"])
        else:
            right = serialize_record(row, "right")
        left_texts.append(left)
        right_texts.append(right)
    return left_texts, right_texts


def rowwise_cosine(left_matrix, right_matrix) -> np.ndarray:
    product = left_matrix.multiply(right_matrix)
    return np.asarray(product.sum(axis=1)).ravel()


class BaseBaselineScorer:
    name: str = "base"

    def fit(self, train_df: pd.DataFrame) -> "BaseBaselineScorer":
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class AlwaysNegativeBaseline(BaseBaselineScorer):
    name = "always_negative"

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=float)


class ExactTitleMatchBaseline(BaseBaselineScorer):
    name = "exact_title_match"

    def score(self, df: pd.DataFrame) -> np.ndarray:
        scores = []
        for _, row in df.iterrows():
            left = normalize_text(row.get("left_title", ""))
            right = normalize_text(row.get("right_title", ""))
            scores.append(1.0 if left and right and left == right else 0.0)
        return np.asarray(scores, dtype=float)


class NearExactTitleMatchBaseline(BaseBaselineScorer):
    name = "near_exact_title_match"

    def score(self, df: pd.DataFrame) -> np.ndarray:
        scores = []
        for _, row in df.iterrows():
            left = normalize_text(row.get("left_title", ""))
            right = normalize_text(row.get("right_title", ""))
            if not left or not right:
                scores.append(0.0)
            else:
                scores.append(fuzz.token_set_ratio(left, right) / 100.0)
        return np.asarray(scores, dtype=float)


class BagOfWordsCosineBaseline(BaseBaselineScorer):
    name = "bow_cosine"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(name=self.name)
        self.vectorizer_: CountVectorizer | None = None

    def fit(self, train_df: pd.DataFrame) -> "BagOfWordsCosineBaseline":
        left, right = get_left_right_texts(train_df)
        corpus = left + right
        self.vectorizer_ = CountVectorizer(
            ngram_range=self.config.word_ngram_range,
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            lowercase=False,
        )
        self.vectorizer_.fit(corpus)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if self.vectorizer_ is None:
            raise RuntimeError("Call fit() before score().")
        left, right = get_left_right_texts(df)
        left_mat = normalize(self.vectorizer_.transform(left), norm="l2", axis=1)
        right_mat = normalize(self.vectorizer_.transform(right), norm="l2", axis=1)
        return rowwise_cosine(left_mat, right_mat)


class TfidfCosineBaseline(BaseBaselineScorer):
    def __init__(self, analyzer: str = "word", config: BaselineConfig | None = None):
        self.analyzer = analyzer
        self.config = config or BaselineConfig(name=f"tfidf_cosine_{analyzer}")
        self.name = f"tfidf_cosine_{analyzer}"
        self.vectorizer_: TfidfVectorizer | None = None

    def fit(self, train_df: pd.DataFrame) -> "TfidfCosineBaseline":
        left, right = get_left_right_texts(train_df)
        corpus = left + right
        ngram_range = (
            self.config.word_ngram_range
            if self.analyzer == "word"
            else self.config.char_ngram_range
        )
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb" if self.analyzer == "char" else "word",
            ngram_range=ngram_range,
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
        )
        self.vectorizer_.fit(corpus)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if self.vectorizer_ is None:
            raise RuntimeError("Call fit() before score().")
        left, right = get_left_right_texts(df)
        left_mat = self.vectorizer_.transform(left)
        right_mat = self.vectorizer_.transform(right)
        return rowwise_cosine(left_mat, right_mat)


class FieldWiseLexicalHeuristicBaseline(BaseBaselineScorer):
    name = "fieldwise_lexical_heuristic"

    def score(self, df: pd.DataFrame) -> np.ndarray:
        scores = []
        for _, row in df.iterrows():
            left_title = normalize_text(row.get("left_title", ""))
            right_title = normalize_text(row.get("right_title", ""))
            if left_title and right_title:
                title_sim = fuzz.token_set_ratio(left_title, right_title) / 100.0
            else:
                title_sim = 0.0

            left_brand = normalize_brand(row.get("left_brand", ""))
            right_brand = normalize_brand(row.get("right_brand", ""))
            if left_brand and right_brand:
                brand_sim = fuzz.ratio(left_brand, right_brand) / 100.0
            elif not left_brand and not right_brand:
                brand_sim = 0.5
            else:
                brand_sim = 0.0

            left_desc = normalize_text(row.get("left_description", ""))
            right_desc = normalize_text(row.get("right_description", ""))
            if left_desc and right_desc:
                desc_sim = fuzz.token_set_ratio(left_desc, right_desc) / 100.0
            else:
                desc_sim = 0.0

            left_price = safe_float(row.get("left_price"))
            right_price = safe_float(row.get("right_price"))
            left_curr = normalize_text(row.get("left_price_currency", ""))
            right_curr = normalize_text(row.get("right_price_currency", ""))
            if (
                not pd.isna(left_price)
                and not pd.isna(right_price)
                and left_curr
                and right_curr
                and left_curr == right_curr
            ):
                denom = max(abs(left_price), abs(right_price), 1e-6)
                price_sim = max(0.0, 1.0 - abs(left_price - right_price) / denom)
            else:
                price_sim = 0.0

            total = (
                0.60 * title_sim
                + 0.15 * brand_sim
                + 0.15 * desc_sim
                + 0.10 * price_sim
            )
            scores.append(float(np.clip(total, 0.0, 1.0)))
        return np.asarray(scores, dtype=float)


_BASELINE_REGISTRY: dict[str, type[BaseBaselineScorer]] = {
    "always_negative": AlwaysNegativeBaseline,
    "exact_title_match": ExactTitleMatchBaseline,
    "near_exact_title_match": NearExactTitleMatchBaseline,
    "bow_cosine": BagOfWordsCosineBaseline,
    "tfidf_cosine_word": lambda: TfidfCosineBaseline(analyzer="word"),
    "tfidf_cosine_char": lambda: TfidfCosineBaseline(analyzer="char"),
    "fieldwise_lexical_heuristic": FieldWiseLexicalHeuristicBaseline,
}


def available_baselines() -> list[str]:
    return [
        "always_negative",
        "exact_title_match",
        "bow_cosine",
        "tfidf_cosine_word",
        "tfidf_cosine_char",
        "fieldwise_lexical_heuristic",
    ]


def make_baseline(name: str) -> BaseBaselineScorer:
    if name not in _BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline {name!r}. Available: {available_baselines()}"
        )
    factory = _BASELINE_REGISTRY[name]
    if callable(factory) and not isinstance(factory, type):
        return factory()
    return factory()
