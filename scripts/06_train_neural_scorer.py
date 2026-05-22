"""Stage 1: train neural/statistical scorer and save raw match scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR, PROJECT_ROOT
from src.data_loading import load_processed_variant, processed_variant_dir
from src.evaluation import build_prediction_dataframe, summarize_feature_importance
from src.features import PairFeatureConfig
from src.scorer import PairwiseMatchScorer, ScorerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train scorer and save raw neural scores (no thresholding)."
    )
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--predict-splits",
        nargs="+",
        default=["train", "valid", "test"],
    )
    parser.add_argument("--model-type", default="logreg", choices=["logreg"])
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument(
        "--class-weight",
        default="balanced",
        choices=["balanced", "none"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run outputs",
    )
    return parser.parse_args()


def _run_dirs(variant_id: str, run_name: str) -> tuple[Path, Path, Path]:
    model_dir = OUTPUTS_DIR / "models" / variant_id / run_name
    pred_dir = OUTPUTS_DIR / "predictions" / variant_id / run_name
    table_dir = OUTPUTS_DIR / "tables" / "neural_scorer" / variant_id / run_name
    return model_dir, pred_dir, table_dir


def main() -> int:
    args = parse_args()
    variant_id = args.variant
    run_name = args.run_name

    if not processed_variant_dir(variant_id).exists():
        print(
            "Processed data not found. Run:\n"
            "  python scripts/03_prepare_variant.py",
            file=sys.stderr,
        )
        return 1

    model_dir, pred_dir, table_dir = _run_dirs(variant_id, run_name)
    model_path = model_dir / "model.joblib"

    if model_path.exists() and not args.overwrite:
        print(
            f"Run already exists at {model_dir}. Use --overwrite to replace.",
            file=sys.stderr,
        )
        return 1

    for d in (model_dir, pred_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    dataset = load_processed_variant(variant_id)
    split_map = {
        "train": dataset.train.df,
        "valid": dataset.valid.df,
        "test": dataset.test.df,
    }

    if args.train_split not in split_map:
        print(
            f"Unknown train split {args.train_split!r}. "
            f"Choose from: {list(split_map)}",
            file=sys.stderr,
        )
        return 1

    for split in args.predict_splits:
        if split not in split_map:
            print(
                f"Unknown predict split {split!r}. Choose from: {list(split_map)}",
                file=sys.stderr,
            )
            return 1

    train_df = split_map[args.train_split]

    print("TRAINING NEURAL SCORER\n")
    print(f"Variant: {variant_id}")
    print(f"Run name: {run_name}")
    print(f"Training split: {args.train_split}")
    print(f"Training rows: {len(train_df)}")
    print("Training labels:")
    for label, count in train_df["label"].value_counts().sort_index().items():
        print(f"  {label}: {count}")

    scorer = PairwiseMatchScorer(
        config=ScorerConfig(
            variant_id=variant_id,
            model_type=args.model_type,
            class_weight=None if args.class_weight == "none" else "balanced",
            random_state=args.random_state,
        ),
        feature_config=PairFeatureConfig(),
    )
    scorer.fit(train_df)
    scorer.save(model_path)
    print(f"\nSaved model:\n  {model_path}")

    prediction_files: dict[str, str] = {}
    print("\nSaved raw predictions:")
    for split in args.predict_splits:
        df = split_map[split]
        scores = scorer.predict_proba(df)
        pred_df = build_prediction_dataframe(df, scores, split=split)
        out_path = pred_dir / f"raw_{split}_predictions.csv"
        pred_df.to_csv(out_path, index=False)
        rel = out_path.relative_to(PROJECT_ROOT)
        prediction_files[split] = str(rel)
        print(f"  {split}: {rel}")

    importance_path = table_dir / "feature_importance.csv"
    summarize_feature_importance(scorer).to_csv(importance_path, index=False)
    print(f"\nSaved feature importance: {importance_path.relative_to(PROJECT_ROOT)}")

    metadata = {
        "variant_id": variant_id,
        "run_name": run_name,
        "model_type": args.model_type,
        "train_split": args.train_split,
        "predict_splits": args.predict_splits,
        "class_weight": args.class_weight,
        "random_state": args.random_state,
        "num_train_rows": len(train_df),
        "train_label_distribution": train_df["label"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
        "output_prediction_files": prediction_files,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "default_threshold_note": (
            "No decision threshold applied in training stage. "
            "Use scripts/08_validate_neural_predictions.py."
        ),
    }
    meta_path = table_dir / "train_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {meta_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
