"""Learned pairwise match probability scorer (training/scoring only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features import PairFeatureConfig, PairFeatureExtractor


@dataclass
class ScorerConfig:
    variant_id: str = "pairwise_50_medium_unseen100"
    model_type: str = "logreg"
    class_weight: str | None = "balanced"
    C: float = 1.0
    max_iter: int = 2000
    random_state: int = 42
    probability_calibration: str | None = None


class PairwiseMatchScorer:
    """TF-IDF + handcrafted features + logistic regression match scorer."""

    def __init__(
        self,
        config: ScorerConfig | None = None,
        feature_config: PairFeatureConfig | None = None,
    ):
        self.config = config or ScorerConfig()
        self.feature_config = feature_config or PairFeatureConfig()
        self.feature_extractor_ = PairFeatureExtractor(self.feature_config)
        self.model_: LogisticRegression | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "PairwiseMatchScorer":
        """Fit feature extractor and classifier on training pairs only."""
        if self.config.model_type != "logreg":
            raise ValueError(
                f"Unsupported model_type={self.config.model_type!r}. Use 'logreg'."
            )

        X_train = self.feature_extractor_.fit_transform(train_df)
        y_train = train_df["label"].astype(int).values

        self.model_ = LogisticRegression(
            C=self.config.C,
            class_weight=self.config.class_weight,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            solver="liblinear",
        )
        fit_kwargs: dict = {}
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.shape[0] != len(y_train):
                raise ValueError(
                    f"sample_weight length {sw.shape[0]} != train rows {len(y_train)}"
                )
            fit_kwargs["sample_weight"] = sw
        try:
            self.model_.fit(X_train, y_train, **fit_kwargs)
        except TypeError as exc:
            if sample_weight is not None:
                raise TypeError(
                    f"model_type={self.config.model_type!r} does not support sample_weight"
                ) from exc
            raise
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Raw positive-class confidence scores in [0, 1] (no thresholding)."""
        if self.model_ is None:
            raise RuntimeError("Scorer is not fitted. Call fit() first.")
        X = self.feature_extractor_.transform(df)
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float) -> np.ndarray:
        """Binary predictions at an explicit threshold (for convenience only)."""
        return (self.predict_proba(df) >= threshold).astype(int)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "feature_config": asdict(self.feature_config),
            "feature_extractor": self.feature_extractor_,
            "model": self.model_,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "PairwiseMatchScorer":
        payload = joblib.load(path)
        scorer = cls(
            config=ScorerConfig(**payload["config"]),
            feature_config=PairFeatureConfig(**payload["feature_config"]),
        )
        scorer.feature_extractor_ = payload["feature_extractor"]
        scorer.model_ = payload["model"]
        return scorer
