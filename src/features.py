"""Pairwise feature extraction for entity matching scorers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loading import serialize_product_pair_row, serialize_product_side

__all__ = [
    "PairFeatureConfig",
    "PairFeatureExtractor",
    "normalize_text",
    "safe_float",
    "token_set",
    "token_jaccard",
    "token_containment",
    "exact_match_or_missing",
    "sparse_rowwise_cosine_similarity",
    "serialize_product_pair_row",
    "serialize_product_side",
]

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PairFeatureConfig:
    use_title_features: bool = True
    use_description_features: bool = True
    use_brand_features: bool = True
    use_price_features: bool = True
    use_text_length_features: bool = True
    use_full_pair_tfidf: bool = True
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    max_word_features: int = 50000
    max_char_features: int = 50000
    min_df: int = 2
    max_df: float = 0.95


def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = str(x).strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def safe_float(x) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    cleaned = re.sub(r"[^\d.\-]", "", str(x))
    if not cleaned or cleaned in {".", "-", "-."}:
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def token_set(x: str) -> set[str]:
    text = normalize_text(x)
    if not text:
        return set()
    return set(text.split())


def token_jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def token_containment(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def exact_match_or_missing(a, b) -> tuple[float, float, float, float]:
    """Return exact_match, both_missing, left_missing, right_missing."""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    left_missing = 1.0 if not a_norm else 0.0
    right_missing = 1.0 if not b_norm else 0.0
    both_missing = 1.0 if left_missing and right_missing else 0.0
    exact_match = 1.0 if a_norm and b_norm and a_norm == b_norm else 0.0
    return exact_match, both_missing, left_missing, right_missing


def sparse_rowwise_cosine_similarity(left_matrix, right_matrix) -> np.ndarray:
    """Row-wise cosine similarity for L2-normalized sparse TF-IDF rows."""
    product = left_matrix.multiply(right_matrix)
    return np.asarray(product.sum(axis=1)).ravel()


def _fuzz_scale(score: float) -> float:
    return float(score) / 100.0


def _len_features(left: str, right: str, prefix: str) -> dict[str, float]:
    left_len = len(left)
    right_len = len(right)
    abs_diff = abs(left_len - right_len)
    denom = max(left_len, right_len, 1)
    return {
        f"{prefix}_left_len": float(left_len),
        f"{prefix}_right_len": float(right_len),
        f"{prefix}_abs_len_diff": float(abs_diff),
        f"{prefix}_rel_len_diff": float(abs_diff / denom),
        f"{prefix}_both_present": float(bool(left) and bool(right)),
    }


def _text_column(df: pd.DataFrame, col: str, fallback: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series([fallback] * len(df), index=df.index)


class PairFeatureExtractor(BaseEstimator, TransformerMixin):
    """Sparse pairwise features: handcrafted + TF-IDF similarities + pair_text TF-IDF."""

    def __init__(self, config: Optional[PairFeatureConfig] = None):
        self.config = config or PairFeatureConfig()
        self._handcrafted_columns: list[str] = []
        self._similarity_columns: list[str] = []
        self._feature_names: list[str] = []
        self.title_word_vectorizer_: TfidfVectorizer | None = None
        self.title_char_vectorizer_: TfidfVectorizer | None = None
        self.full_word_vectorizer_: TfidfVectorizer | None = None
        self.full_char_vectorizer_: TfidfVectorizer | None = None
        self.pair_text_vectorizer_: TfidfVectorizer | None = None

    def _ensure_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if "left_text" not in work.columns:
            work["left_text"] = work.apply(
                lambda r: serialize_product_side(r, "left"), axis=1
            )
        if "right_text" not in work.columns:
            work["right_text"] = work.apply(
                lambda r: serialize_product_side(r, "right"), axis=1
            )
        if "pair_text" not in work.columns:
            work["pair_text"] = work.apply(serialize_product_pair_row, axis=1)
        return work

    def _build_handcrafted_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        work = self._ensure_text_columns(df)
        rows: list[dict[str, float]] = []

        for idx, row in work.iterrows():
            feats: dict[str, float] = {}
            left_title = normalize_text(row.get("left_title", ""))
            right_title = normalize_text(row.get("right_title", ""))
            left_desc = normalize_text(row.get("left_description", ""))
            right_desc = normalize_text(row.get("right_description", ""))
            left_brand = normalize_text(row.get("left_brand", ""))
            right_brand = normalize_text(row.get("right_brand", ""))
            left_text = normalize_text(row.get("left_text", ""))
            right_text = normalize_text(row.get("right_text", ""))
            pair_text = normalize_text(row.get("pair_text", ""))

            if cfg.use_title_features:
                feats["title_fuzz_token_set_ratio"] = _fuzz_scale(
                    fuzz.token_set_ratio(left_title, right_title)
                )
                feats["title_fuzz_ratio"] = _fuzz_scale(fuzz.ratio(left_title, right_title))
                feats["title_token_jaccard"] = token_jaccard(left_title, right_title)
                feats["title_token_containment"] = token_containment(left_title, right_title)
                feats.update(_len_features(left_title, right_title, "title"))

            if cfg.use_description_features:
                feats["description_fuzz_token_set_ratio"] = _fuzz_scale(
                    fuzz.token_set_ratio(left_desc, right_desc)
                )
                feats["description_token_jaccard"] = token_jaccard(left_desc, right_desc)
                feats["description_token_containment"] = token_containment(
                    left_desc, right_desc
                )
                feats.update(_len_features(left_desc, right_desc, "description"))

            if cfg.use_brand_features:
                em, both_m, left_m, right_m = exact_match_or_missing(
                    left_brand, right_brand
                )
                feats["brand_exact_match"] = em
                feats["brand_both_missing"] = both_m
                feats["brand_left_missing"] = left_m
                feats["brand_right_missing"] = right_m
                feats["brand_both_present"] = float(
                    bool(left_brand) and bool(right_brand)
                )
                feats["brand_fuzz_ratio"] = _fuzz_scale(
                    fuzz.ratio(left_brand, right_brand)
                )
                feats["brand_fuzz_token_set_ratio"] = _fuzz_scale(
                    fuzz.token_set_ratio(left_brand, right_brand)
                )

            if cfg.use_price_features:
                left_price = safe_float(row.get("left_price"))
                right_price = safe_float(row.get("right_price"))
                left_curr = normalize_text(row.get("left_price_currency", ""))
                right_curr = normalize_text(row.get("right_price_currency", ""))

                feats["price_left_missing"] = float(pd.isna(left_price))
                feats["price_right_missing"] = float(pd.isna(right_price))
                feats["price_both_present"] = float(
                    not pd.isna(left_price) and not pd.isna(right_price)
                )
                if feats["price_both_present"]:
                    abs_diff = abs(left_price - right_price)
                    denom = max(abs(left_price), abs(right_price), 1e-6)
                    feats["price_abs_diff"] = float(abs_diff)
                    feats["price_rel_diff"] = float(abs_diff / denom)
                else:
                    feats["price_abs_diff"] = 0.0
                    feats["price_rel_diff"] = 0.0

                curr_em, curr_both_m, curr_left_m, curr_right_m = exact_match_or_missing(
                    left_curr, right_curr
                )
                feats["price_same_currency"] = curr_em
                feats["price_currency_both_present"] = float(
                    bool(left_curr) and bool(right_curr)
                )
                feats["price_currency_exact_match"] = curr_em
                feats["price_currency_both_missing"] = curr_both_m

            if cfg.use_text_length_features:
                feats["left_text_len"] = float(len(left_text))
                feats["right_text_len"] = float(len(right_text))
                feats["pair_text_len"] = float(len(pair_text))
                abs_diff = abs(len(left_text) - len(right_text))
                denom = max(len(left_text), len(right_text), 1)
                feats["abs_left_right_text_len_diff"] = float(abs_diff)
                feats["rel_left_right_text_len_diff"] = float(abs_diff / denom)

            rows.append(feats)

        handcrafted = pd.DataFrame(rows, index=work.index).fillna(0.0)
        self._handcrafted_columns = list(handcrafted.columns)
        return handcrafted

    def _fit_vectorizer(
        self,
        left: pd.Series,
        right: pd.Series,
        *,
        analyzer: str,
        max_features: int,
    ) -> TfidfVectorizer:
        corpus = pd.concat([left, right], ignore_index=True)
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=self.config.word_ngram_range
            if analyzer == "word"
            else self.config.char_ngram_range,
            max_features=max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=True,
            norm="l2",
        )
        vectorizer.fit(corpus.astype(str))
        return vectorizer

    def _similarity_from_vectorizer(
        self,
        df: pd.DataFrame,
        vectorizer: TfidfVectorizer,
        left_col: str,
        right_col: str,
    ) -> np.ndarray:
        left = _text_column(df, left_col)
        right = _text_column(df, right_col)
        left_mat = vectorizer.transform(left)
        right_mat = vectorizer.transform(right)
        return sparse_rowwise_cosine_similarity(left_mat, right_mat)

    def fit(self, df: pd.DataFrame, y=None):
        cfg = self.config
        work = self._ensure_text_columns(df)
        self._build_handcrafted_frame(work)

        if cfg.use_title_features:
            left_title = _text_column(work, "left_title").map(normalize_text)
            right_title = _text_column(work, "right_title").map(normalize_text)
            self.title_word_vectorizer_ = self._fit_vectorizer(
                left_title,
                right_title,
                analyzer="word",
                max_features=cfg.max_word_features,
            )
            self.title_char_vectorizer_ = self._fit_vectorizer(
                left_title,
                right_title,
                analyzer="char_wb",
                max_features=cfg.max_char_features,
            )

        if cfg.use_full_pair_tfidf:
            left_text = _text_column(work, "left_text").map(normalize_text)
            right_text = _text_column(work, "right_text").map(normalize_text)
            self.full_word_vectorizer_ = self._fit_vectorizer(
                left_text,
                right_text,
                analyzer="word",
                max_features=cfg.max_word_features,
            )
            self.full_char_vectorizer_ = self._fit_vectorizer(
                left_text,
                right_text,
                analyzer="char_wb",
                max_features=cfg.max_char_features,
            )

            pair_text = _text_column(work, "pair_text").map(normalize_text)
            self.pair_text_vectorizer_ = TfidfVectorizer(
                analyzer="word",
                ngram_range=cfg.word_ngram_range,
                max_features=cfg.max_word_features,
                min_df=cfg.min_df,
                max_df=cfg.max_df,
                sublinear_tf=True,
                norm="l2",
            )
            self.pair_text_vectorizer_.fit(pair_text.astype(str))

        self._similarity_columns = []
        if self.title_word_vectorizer_ is not None:
            self._similarity_columns.extend(
                ["title_word_tfidf_cosine", "title_char_tfidf_cosine"]
            )
        if self.full_word_vectorizer_ is not None:
            self._similarity_columns.extend(
                ["full_text_word_tfidf_cosine", "full_text_char_tfidf_cosine"]
            )

        pair_names = []
        if self.pair_text_vectorizer_ is not None:
            pair_names = [
                f"pair_text_tfidf__{name}"
                for name in self.pair_text_vectorizer_.get_feature_names_out()
            ]

        self._feature_names = (
            self._handcrafted_columns
            + self._similarity_columns
            + pair_names
        )
        return self

    def transform(self, df: pd.DataFrame):
        work = self._ensure_text_columns(df)
        handcrafted = self._build_handcrafted_frame(work)
        blocks: list[sparse.spmatrix] = [
            sparse.csr_matrix(handcrafted.values.astype(np.float64)),
        ]

        similarity_values: list[np.ndarray] = []
        if self.title_word_vectorizer_ is not None:
            similarity_values.append(
                self._similarity_from_vectorizer(
                    work, self.title_word_vectorizer_, "left_title", "right_title"
                )
            )
            similarity_values.append(
                self._similarity_from_vectorizer(
                    work, self.title_char_vectorizer_, "left_title", "right_title"
                )
            )
        if self.full_word_vectorizer_ is not None:
            similarity_values.append(
                self._similarity_from_vectorizer(
                    work, self.full_word_vectorizer_, "left_text", "right_text"
                )
            )
            similarity_values.append(
                self._similarity_from_vectorizer(
                    work, self.full_char_vectorizer_, "left_text", "right_text"
                )
            )

        if similarity_values:
            sim_matrix = np.column_stack(similarity_values)
            blocks.append(sparse.csr_matrix(sim_matrix))

        if self.pair_text_vectorizer_ is not None:
            pair_text = _text_column(work, "pair_text").map(normalize_text)
            blocks.append(self.pair_text_vectorizer_.transform(pair_text.astype(str)))

        return sparse.hstack(blocks, format="csr")

    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names, dtype=object)
