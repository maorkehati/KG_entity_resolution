"""Run the full entity-resolution experiment pipeline end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

REQUIRED_SCRIPTS = {
    "03_prepare_variant.py",
    "04_dataset_smoke_test.py",
    "06_train_neural_scorer.py",
    "08_validate_neural_predictions.py",
    "09_apply_symbolic_governance.py",
    "10_sweep_decision_configs.py",
    "12_run_naive_baselines.py",
}

OPTIONAL_SCRIPTS = {
    "00_download_data.py",
    "01_inspect_data.py",
    "02_smoke_test.py",
    "05_explore_data.py",
    "11_plot_sweep_results.py",
    "13_plot_main_method_comparison.py",
    "14_analyze_symbolic_rule_impact.py",
    "15_run_soft_risk_experiment.py",
    "16_iterative_symbolic_hard_negative_training.py",
    "17_train_structured_attention_models.py",
    "18_plot_final_comparison_figures.py",
    "19_prepare_ablation_report_results.py",
}

EXPECTED_OUTPUTS = [
    "data/processed/{variant}/train.parquet",
    "data/processed/{variant}/valid.parquet",
    "data/processed/{variant}/test.parquet",
    "outputs/predictions/{variant}/{run_name}/raw_train_predictions.csv",
    "outputs/predictions/{variant}/{run_name}/raw_valid_predictions.csv",
    "outputs/predictions/{variant}/{run_name}/raw_test_predictions.csv",
    "outputs/tables/sweeps/{variant}/{run_name}/decision_config_sweep.csv",
    "outputs/tables/baselines/{variant}/baseline_metrics.csv",
    "outputs/tables/symbolic_analysis/{variant}/{run_name}/constraint_impact.csv",
    "outputs/figures/final_comparison/{variant}/{run_name}/figure1_baselines_vs_proposed_test_f1.png",
    "outputs/figures/final_comparison/{variant}/{run_name}/figure2_proposed_variants_test_f1.png",
    "outputs/figures/final_comparison/{variant}/{run_name}/figure3_intro_motivation_test_f1.png",
    "outputs/tables/report_ablations/{variant}/{run_name}/soft_risk_ablation_table.csv",
    "outputs/tables/report_ablations/{variant}/{run_name}/hard_negative_ablation_table.csv",
    "outputs/tables/report_ablations/{variant}/{run_name}/architecture_ablation_table.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full WDC entity-resolution pipeline end-to-end."
    )
    parser.add_argument("--variant", default="pairwise_50_medium_unseen100")
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument("--hard-negative-experiment", default="symbolic_hn_strong")
    parser.add_argument("--threshold", type=float, default=0.66)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip heavier optional experiments unless explicitly enabled.",
    )
    parser.add_argument("--soft-risk", action="store_true")
    parser.add_argument("--hard-negative-training", action="store_true")
    parser.add_argument("--structured-models", action="store_true")
    parser.add_argument("--final-figures", action="store_true")
    parser.add_argument("--report-ablations", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --overwrite to scripts that support it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--log-file",
        default="outputs/logs/run_all_experiments.log",
    )
    return parser.parse_args()


def ensure_repo_root() -> None:
    if not (ROOT / "scripts").is_dir() or not (ROOT / "src").is_dir():
        raise RuntimeError(
            f"Repository root not found (expected scripts/ and src/ under {ROOT}). "
            "Run this script from the project root."
        )


def script_path(name: str, *, required: bool) -> Path | None:
    path = SCRIPTS_DIR / name
    if path.exists():
        return path
    if required:
        raise FileNotFoundError(f"Required script missing: {path}")
    print(f"WARNING: optional script missing, skipping stage: {path}")
    return None


def run_command(
    cmd: list[str],
    log_file: Path,
    *,
    dry_run: bool = False,
    cwd: Path = ROOT,
) -> None:
    line = " ".join(cmd)
    print(f"\n>>> {line}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 80}\n>>> {line}\n")
    if dry_run:
        return

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    with log_file.open("a", encoding="utf-8") as log:
        for out_line in proc.stdout:
            print(out_line, end="")
            log.write(out_line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}): {line}")


class PipelineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = args.python
        self.log_file = (ROOT / args.log_file).resolve()
        self.commands_run: list[str] = []
        self.warnings: list[str] = []
        self.ran_soft_risk = False
        self.ran_hard_negative = False
        self.ran_structured = False
        self.ran_final_figures = False
        self.ran_report_ablations = False

    def _py(self, script_name: str, *extra: str, required: bool = True) -> list[str] | None:
        path = script_path(script_name, required=required)
        if path is None:
            return None
        return [self.python, str(path), *extra]

    def _maybe_overwrite(self, cmd: list[str]) -> list[str]:
        if self.args.force:
            cmd = [*cmd, "--overwrite"]
        return cmd

    def run(self, cmd: list[str] | None, stage: str) -> None:
        if cmd is None:
            self.warnings.append(f"Skipped stage (missing script): {stage}")
            return
        run_command(cmd, self.log_file, dry_run=self.args.dry_run)
        self.commands_run.append(" ".join(cmd))

    def _run_optional(self, enabled_flag: bool, stage_name: str) -> bool:
        if self.args.skip_optional and not enabled_flag:
            return False
        return enabled_flag or not self.args.skip_optional

    def execute(self) -> None:
        a = self.args
        v, r = a.variant, a.run_name
        thr = str(a.threshold)
        hn = a.hard_negative_experiment

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not a.dry_run:
            self.log_file.write_text(
                f"run_all_experiments started {datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )

        # Stage 0
        self.run(self._py("02_smoke_test.py", required=False), "smoke_test")
        if not a.skip_download:
            self.run(self._py("00_download_data.py", required=False), "download_data")
        else:
            self.warnings.append("Skipped download (--skip-download).")
        self.run(self._py("01_inspect_data.py", required=False), "inspect_data")

        # Stage 1
        self.run(
            self._py("03_prepare_variant.py", "--variant", v, required=True),
            "prepare_variant",
        )
        self.run(
            self._py("04_dataset_smoke_test.py", "--variant", v, required=True),
            "dataset_smoke_test",
        )
        self.run(
            self._py(
                "05_explore_data.py",
                "--variant",
                v,
                "--show-raw",
                "--max-examples",
                "5",
                required=False,
            ),
            "explore_data",
        )

        # Stage 2
        train_cmd = self._py(
            "06_train_neural_scorer.py",
            "--variant",
            v,
            "--run-name",
            r,
            "--train-split",
            "train",
            "--predict-splits",
            "train",
            "valid",
            "test",
            required=True,
        )
        if train_cmd:
            train_cmd = self._maybe_overwrite(train_cmd)
        self.run(train_cmd, "train_neural_scorer")

        for objective, extra in (
            ("f1", ["--objective", "f1"]),
            (
                "recall_at_min_precision",
                [
                    "--objective",
                    "recall_at_min_precision",
                    "--min-precision",
                    "0.90",
                ],
            ),
        ):
            val_cmd = self._py(
                "08_validate_neural_predictions.py",
                "--variant",
                v,
                "--run-name",
                r,
                "--calibration-split",
                "valid",
                *extra,
                required=True,
            )
            self.run(val_cmd, f"validate_neural_{objective}")

        # Stage 3
        self.run(
            self._py(
                "09_apply_symbolic_governance.py",
                "--variant",
                v,
                "--run-name",
                r,
                "--threshold",
                thr,
                "--decision-mode",
                "invalid_blocks",
                required=True,
            ),
            "governance_invalid_blocks",
        )
        self.run(
            self._py(
                "09_apply_symbolic_governance.py",
                "--variant",
                v,
                "--run-name",
                r,
                "--threshold",
                thr,
                "--decision-mode",
                "strict_valid_or_high_confidence",
                "--tau-high",
                "0.90",
                "--uncertain-action",
                "accept",
                required=True,
            ),
            "governance_strict_two_threshold",
        )

        # Stage 4
        sweep_cmd = self._py(
            "10_sweep_decision_configs.py",
            "--variant",
            v,
            "--run-name",
            r,
            required=True,
        )
        if sweep_cmd:
            sweep_cmd = self._maybe_overwrite(sweep_cmd)
        self.run(sweep_cmd, "sweep_decision_configs")

        plot_sweep = self._py(
            "11_plot_sweep_results.py",
            "--variant",
            v,
            "--run-name",
            r,
            required=False,
        )
        if plot_sweep:
            plot_sweep = self._maybe_overwrite(plot_sweep)
        self.run(plot_sweep, "plot_sweep_results")

        # Stage 5
        baseline_cmd = self._py("12_run_naive_baselines.py", "--variant", v, required=True)
        if baseline_cmd:
            baseline_cmd = self._maybe_overwrite(baseline_cmd)
        self.run(baseline_cmd, "naive_baselines")

        # Stage 6
        main_cmp = self._py(
            "13_plot_main_method_comparison.py",
            "--variant",
            v,
            "--run-name",
            r,
            required=False,
        )
        if main_cmp:
            main_cmp = self._maybe_overwrite(main_cmp)
            self.run(main_cmp, "main_method_comparison")
        else:
            self.warnings.append("Main comparison script missing; continuing.")

        # Stage 7
        sym_cmd = self._py(
            "14_analyze_symbolic_rule_impact.py",
            "--variant",
            v,
            "--run-name",
            r,
            "--threshold",
            thr,
            "--symbolic-profile",
            "conservative",
            "--splits",
            "valid",
            "test",
            required=False,
        )
        if sym_cmd:
            sym_cmd = self._maybe_overwrite(sym_cmd)
        self.run(sym_cmd, "symbolic_rule_impact")

        # Stage 8
        if self._run_optional(a.soft_risk, "soft_risk"):
            soft_cmd = self._py(
                "15_run_soft_risk_experiment.py",
                "--variant",
                v,
                "--run-name",
                r,
                "--symbolic-profile",
                "conservative",
                "--calibration-split",
                "valid",
                "--test-split",
                "test",
                required=False,
            )
            if soft_cmd:
                soft_cmd = self._maybe_overwrite(soft_cmd)
                self.run(soft_cmd, "soft_risk")
                self.ran_soft_risk = True
            else:
                self.warnings.append("Soft risk script missing; skipped.")

        # Stage 9
        if self._run_optional(a.hard_negative_training, "hard_negative_training"):
            hn_cmd = self._py(
                "16_iterative_symbolic_hard_negative_training.py",
                "--variant",
                v,
                "--base-run-name",
                r,
                "--experiment-name",
                hn,
                "--iterations",
                "3",
                "--score-threshold",
                thr,
                "--hard-negative-weight",
                "3.0",
                "--constraint-mode",
                "strong_only",
                "--threshold",
                thr,
                required=False,
            )
            if hn_cmd:
                hn_cmd = self._maybe_overwrite(hn_cmd)
                self.run(hn_cmd, "hard_negative_training")
                self.ran_hard_negative = True
            else:
                self.warnings.append("Hard-negative training script missing; skipped.")

        # Stage 10
        if self._run_optional(a.structured_models, "structured_models"):
            struct_cmd = self._py(
                "17_train_structured_attention_models.py",
                "--variant",
                v,
                "--run-name",
                r,
                "--models",
                "field_attention",
                "structured_transformer",
                "--epochs",
                "30",
                "--batch-size",
                "256",
                required=False,
            )
            if struct_cmd:
                struct_cmd = self._maybe_overwrite(struct_cmd)
                self.run(struct_cmd, "structured_models")
                self.ran_structured = True
            else:
                self.warnings.append("Structured models script missing; skipped.")

        # Stage 11
        if self._run_optional(a.final_figures, "final_figures"):
            fig_cmd = self._py(
                "18_plot_final_comparison_figures.py",
                "--variant",
                v,
                "--run-name",
                r,
                "--hard-negative-experiment",
                hn,
                required=False,
            )
            if fig_cmd:
                if self.args.force:
                    fig_cmd.append("--overwrite")
                if self.ran_soft_risk:
                    fig_cmd.append("--soft-risk-enabled")
                if self.ran_structured:
                    fig_cmd.append("--structured-models-enabled")
                self.run(fig_cmd, "final_figures")
                self.ran_final_figures = True
            else:
                self.warnings.append("Final figures script missing; skipped.")

        # Stage 12
        if self._run_optional(a.report_ablations, "report_ablations"):
            abl_cmd = self._py(
                "19_prepare_ablation_report_results.py",
                "--variant",
                v,
                "--run-name",
                r,
                "--hard-negative-experiment",
                hn,
                required=False,
            )
            if abl_cmd:
                if self.args.force:
                    abl_cmd.append("--overwrite")
                self.run(abl_cmd, "report_ablations")
                self.ran_report_ablations = True
            else:
                self.warnings.append("Report ablations script missing; skipped.")


def build_manifest(
    args: argparse.Namespace,
    commands_run: list[str],
    warnings: list[str],
    log_file: Path,
) -> dict[str, Any]:
    expected = [
        p.format(variant=args.variant, run_name=args.run_name)
        for p in EXPECTED_OUTPUTS
    ]
    found: list[str] = []
    missing: list[str] = []
    for rel in expected:
        path = ROOT / rel
        if path.exists():
            found.append(rel)
        else:
            missing.append(rel)

    return {
        "variant": args.variant,
        "run_name": args.run_name,
        "hard_negative_experiment": args.hard_negative_experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commands_run": commands_run,
        "outputs_expected": expected,
        "outputs_found": found,
        "outputs_missing": missing,
        "warnings": warnings,
        "log_file": str(log_file.relative_to(ROOT)).replace("\\", "/"),
        "options": {
            "skip_download": args.skip_download,
            "skip_optional": args.skip_optional,
            "soft_risk": args.soft_risk,
            "hard_negative_training": args.hard_negative_training,
            "structured_models": args.structured_models,
            "final_figures": args.final_figures,
            "report_ablations": args.report_ablations,
            "force": args.force,
            "dry_run": args.dry_run,
        },
    }


def write_manifest(manifest: dict[str, Any], variant: str, run_name: str) -> Path:
    out_dir = ROOT / "outputs" / "tables" / "reproducibility" / variant / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_all_manifest.json"
    import json

    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    ensure_repo_root()

    for name in REQUIRED_SCRIPTS:
        script_path(name, required=True)

    runner = PipelineRunner(args)
    try:
        runner.execute()
    except RuntimeError as exc:
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        if not args.dry_run:
            manifest = build_manifest(args, runner.commands_run, runner.warnings, runner.log_file)
            write_manifest(manifest, args.variant, args.run_name)
        return 1

    manifest = build_manifest(args, runner.commands_run, runner.warnings, runner.log_file)
    manifest_path = write_manifest(manifest, args.variant, args.run_name)

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE" if not args.dry_run else "DRY RUN COMPLETE")
    print("=" * 72)
    print(f"Commands run: {len(runner.commands_run)}")
    print(f"Log file: {runner.log_file}")
    print(f"Manifest: {manifest_path}")
    if manifest["outputs_missing"]:
        print(f"Missing outputs ({len(manifest['outputs_missing'])}):")
        for p in manifest["outputs_missing"]:
            print(f"  - {p}")
    if runner.warnings:
        print("Warnings:")
        for w in runner.warnings:
            print(f"  - {w}")
    if manifest["outputs_missing"] and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
