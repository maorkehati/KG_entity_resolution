# Constraint-Aware Entity Resolution for Knowledge Graph Construction

A small, reproducible neuro-symbolic entity-resolution experiment on the [WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) benchmark.

## Assignment framing

We solve **entity resolution / deduplication** over product records. Each sample is a labeled pair `(e_i, e_j, y_ij)`. A statistical or neural scorer estimates pairwise match probability; a **symbolic governance layer** applies explicit constraints before accepting a merge. Decisions are evaluated with precision, recall, F1, and constraint diagnostics. The goal is not state-of-the-art accuracy but showing that symbolic governance can reduce unsafe false merges and improve interpretability.

## Dataset choice

**WDC Products: A Multi-Dimensional Entity Matching Benchmark** — product offers, pairwise labels, real-world ambiguity, and natural constraints (brand, category, MPN/GTIN, numeric attributes, title conflicts). See the [benchmark page](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) and [GitHub repository](https://github.com/wbsg-uni-mannheim/wdcproducts).

## Setup

```bash
pip install -r requirements.txt
```

Use a virtual environment if your global Python has conflicting NumPy/PyArrow builds (`numpy>=1.26,<2` and `pyarrow>=15` are pinned for compatibility).

From the project root:

```bash
python scripts/00_download_data.py
python scripts/01_inspect_data.py
python scripts/02_smoke_test.py
python scripts/03_prepare_variant.py
python scripts/04_dataset_smoke_test.py
python scripts/05_explore_data.py
python scripts/06_train_neural_scorer.py
python scripts/08_validate_neural_predictions.py --calibration-split valid --objective f1
python scripts/09_apply_symbolic_governance.py --threshold 0.66 --decision-mode invalid_blocks
python scripts/10_sweep_decision_configs.py --variant pairwise_50_medium_unseen100 --run-name neural_logreg
python scripts/11_plot_sweep_results.py --variant pairwise_50_medium_unseen100 --run-name neural_logreg
python scripts/07_lexical_baseline.py
```

The download step saves the benchmark HTML and notes under `data/raw/`, prints candidate download links, and extracts any `.zip` archives found in `data/raw/` into `data/interim/wdc_products/`.

## Reproducing all results

After downloading/extracting the WDC Products files into `data/raw` or `data/interim/wdc_products`, run:

```bash
python scripts/99_run_all_experiments.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --hard-negative-experiment symbolic_hn_strong \
  --soft-risk \
  --hard-negative-training \
  --structured-models \
  --final-figures \
  --report-ablations
```

Or:

```bash
bash run_all_experiments.sh
```

On Windows:

```bat
run_all_experiments.bat
```

This executes dataset preparation, neural scoring, symbolic governance, baselines, sweeps, optional ablations, and report figures.

For a faster run that skips heavier optional experiments:

```bash
python scripts/99_run_all_experiments.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --skip-optional
```

Logs are saved to:

```text
outputs/logs/run_all_experiments.log
```

A reproducibility manifest is saved to:

```text
outputs/tables/reproducibility/{variant_id}/{run_name}/run_all_manifest.json
```

## Selected Benchmark Variant

Main variant:

- Task: pairwise binary entity matching
- Corner-case ratio: 50%
- Development-set size: medium
- Test unseen entities: 100%

Variant ID:

```text
pairwise_50_medium_unseen100
```

Rationale: This setting is challenging enough to expose unsafe false merges, while still feasible for a compact applied research assignment. It evaluates generalization to entirely unseen products and gives the symbolic governance layer a meaningful role.

After extracting `50pair.zip`, the loader discovers files such as:

- `wdcproducts50cc50rnd000un_train_medium.json.gz` (train)
- `wdcproducts50cc50rnd000un_valid_medium.json.gz` (validation)
- `wdcproducts50cc50rnd100un_gs.json.gz` (test, 100% unseen)

If automatic file discovery fails, manually place the downloaded WDC Products pairwise 50% corner-case archive or extracted `.json.gz` files under `data/raw/` or `data/interim/wdc_products/`, then rerun `python scripts/03_prepare_variant.py`.

Switch variants by adding an entry to `WDC_VARIANTS` in `src/config.py` and passing a different `variant_id` to the loader APIs.

## Data exploration

After downloading and preparing the selected WDC Products variant, run:

```bash
python scripts/05_explore_data.py
```

Useful options:

```bash
python scripts/05_explore_data.py --show-raw
python scripts/05_explore_data.py --max-examples 10
python scripts/05_explore_data.py --variant pairwise_50_medium_unseen100
```

This script prints:

- raw file structure,
- normalized pairwise schema,
- label distributions,
- missingness,
- text length statistics,
- brand and price diagnostics,
- representative positive and negative pairs,
- likely hard negatives,
- potentially suspicious positives.

It also saves summary tables under:

```text
outputs/tables/data_exploration/{variant_id}/
```

## Neural-only experiment

The neural/statistical scorer estimates `s_ij = f_theta(phi(e_i, e_j))` using TF-IDF, handcrafted pair features, and logistic regression. It runs in **two stages** so raw scores can be reused with different thresholds.

### Stage 1: train and score

```bash
python scripts/06_train_neural_scorer.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --train-split train \
  --predict-splits train valid test
```

Saves raw match scores (no decision threshold):

```text
outputs/predictions/{variant_id}/{run_name}/raw_{split}_predictions.csv
outputs/models/{variant_id}/{run_name}/model.joblib
```

### Stage 2: validate raw predictions with a threshold

Fixed threshold:

```bash
python scripts/08_validate_neural_predictions.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --threshold 0.5
```

Threshold from validation (e.g. maximize F1):

```bash
python scripts/08_validate_neural_predictions.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --calibration-split valid \
  --objective f1
```

Precision-oriented policy (fewer unsafe false merges):

```bash
python scripts/08_validate_neural_predictions.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --calibration-split valid \
  --objective recall_at_min_precision \
  --min-precision 0.90
```

Validation outputs include thresholded predictions, metrics, threshold sweep, and error analysis under `outputs/tables/neural_scorer/{variant_id}/{run_name}/`.

Optional lexical baseline:

```bash
python scripts/07_lexical_baseline.py
```

The neural scorer is separate from symbolic governance: it learns soft evidence; constraints will validate merges later.

## Symbolic governance layer

The symbolic component is a lightweight product-identity validator for WDC product pairs.

It assigns:

```text
symbolic_status ∈ {valid, invalid, uncertain}
```

where:

- `invalid`: a hard product-identity contradiction was found.
- `valid`: no contradiction and at least one positive symbolic agreement exists.
- `uncertain`: no contradiction, but symbolic evidence is incomplete or weak.

Implemented rules include brand compatibility, accessory/main-product conflict, model-like token conflict, variant modifier conflict, capacity/size quantity conflict, bundle/kit diagnostics, price compatibility diagnostics, color diagnostics, and coarse category keywords.

### Default governed decision rule

```text
if symbolic_status = invalid:
    reject
else if neural_score ≥ τ:
    accept
else:
    reject
```

Here, `uncertain` behaves as `not invalid`.

```bash
python scripts/09_apply_symbolic_governance.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --threshold 0.66 \
  --decision-mode invalid_blocks
```

### Stricter governance rule

```bash
python scripts/09_apply_symbolic_governance.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --threshold 0.66 \
  --decision-mode strict_valid_or_high_confidence \
  --tau-high 0.90 \
  --uncertain-action flag
```

## Naive and lexical baselines

Run:

```bash
python scripts/12_run_naive_baselines.py \
  --variant pairwise_50_medium_unseen100
```

Implemented baselines:

* `always_negative`: predicts all pairs as non-match.
* `exact_title_match`: matches only normalized identical titles.
* `bow_cosine`: raw bag-of-words cosine over serialized records.
* `tfidf_cosine_word`: word TF-IDF cosine over serialized records.
* `tfidf_cosine_char`: character n-gram TF-IDF cosine over serialized records.
* `fieldwise_lexical_heuristic`: fixed weighted title/brand/description/price similarity.

Thresholds are selected on the validation split by default using F1.

Outputs:

```text
outputs/tables/baselines/{variant_id}/baseline_metrics.csv
outputs/tables/baselines/{variant_id}/baseline_test_summary.csv
outputs/predictions/{variant_id}/baselines/
```

## Final comparison figures

After running the baselines, decision sweep, and any optional extension experiments, generate final report figures:

```bash
python scripts/18_plot_final_comparison_figures.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg
```

This creates three figures:

1. Baselines vs the neuro-symbolic method.
2. Variants of the neuro-symbolic method.
3. Introduction motivation figure showing why neural-only and symbolic-only approaches are incomplete.

All methods with tunable configurations are selected using validation F1 and reported using test F1.

Outputs:

```text
outputs/figures/final_comparison/{variant_id}/{run_name}/
outputs/tables/final_comparison/{variant_id}/{run_name}/
```

## Report ablation figures

After running the optional ablation experiments, aggregate the report-ready tables and figures:

```bash
python scripts/19_prepare_ablation_report_results.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --hard-negative-experiment symbolic_hn_strong
```

This produces one table and one figure for each report ablation subsection:

* hard vetoes vs. soft symbolic risk,
* symbolically guided hard-negative training,
* field-aware neural architectures.

Outputs are saved under:

```text
outputs/tables/report_ablations/{variant_id}/{run_name}/
outputs/figures/report_ablations/{variant_id}/{run_name}/
```

## Qualitative error analysis

After training the neural scorer and applying symbolic governance, run:

```bash
python scripts/20_qualitative_error_analysis.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --threshold 0.66
```

The script samples representative test examples from:

* neural false positives blocked by symbolic governance,
* true positives incorrectly blocked by symbolic governance,
* remaining false positives after governance,
* false negatives caused by low neural score or symbolic invalidation.

Outputs are saved under:

```text
outputs/tables/qualitative_error_analysis/{variant_id}/{run_name}/
```

The generated markdown preview (`qualitative_error_examples.md`) can be used to write the qualitative error-analysis section.

## Main comparison plot

After running naive baselines and the decision sweep:

```bash
python scripts/12_run_naive_baselines.py \
  --variant pairwise_50_medium_unseen100

python scripts/10_sweep_decision_configs.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg
```

Generate the final comparison plot:

```bash
python scripts/13_plot_main_method_comparison.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg
```

The script selects the best configuration for each method family using validation F1, then plots the corresponding test F1. This avoids choosing methods directly on the test set.

## Structured neural scorer variants

We compare the existing learned scorer against two lightweight structured neural variants:

1. `field_attention`: encodes field-specific evidence and learns attention over fields.
2. `structured_transformer`: treats field evidence groups as tokens and applies a small TransformerEncoder.

Run:

```bash
python scripts/17_train_structured_attention_models.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg
```

Outputs:

```text
outputs/tables/structured_models/{variant_id}/
outputs/figures/structured_models/{variant_id}/
outputs/predictions/{variant_id}/structured_models/
```

The comparison selects thresholds on validation and reports test metrics.

## Symbolically guided hard-negative training

This is an exploratory integrated neuro-symbolic experiment.

Instead of using symbolic rules only as post-processing, we use them to identify hard negative training examples:

```text
label = 0
neural_score is high
strong symbolic conflict exists
```

These examples are upweighted and the scorer is retrained iteratively.

Run:

```bash
python scripts/16_iterative_symbolic_hard_negative_training.py \
  --variant pairwise_50_medium_unseen100 \
  --base-run-name neural_logreg \
  --experiment-name symbolic_hn_strong \
  --iterations 3
```

Outputs:

```text
outputs/tables/hard_negative_training/{variant_id}/{experiment_name}/
outputs/figures/hard_negative_training/{variant_id}/{experiment_name}/
```

This experiment tests whether symbolic constraints can feed back into learning rather than only vetoing predictions.

## Soft symbolic risk experiment

This is an exploratory variant of symbolic governance.

Instead of treating symbolic violations only as hard vetoes, we compute a graded risk score:

```text
risk(e_i, e_j) = Σ_c w_c · 1[c is violated]
```

We test two soft-governance rules:

```text
governed_score = neural_score - λ · risk
```

and:

```text
accept iff neural_score ≥ τ and risk ≤ ρ
```

Run:

```bash
python scripts/15_run_soft_risk_experiment.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --symbolic-profile conservative
```

Outputs are saved under:

```text
outputs/tables/soft_risk/{variant_id}/{run_name}/
outputs/figures/soft_risk/{variant_id}/{run_name}/
```

This experiment is optional. It tests whether soft symbolic penalties can improve over binary invalid-blocking when symbolic rules are noisy.

## Symbolic rule-impact analysis

After training the neural scorer and running symbolic governance, inspect which symbolic rules contribute most:

```bash
python scripts/14_analyze_symbolic_rule_impact.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg \
  --threshold 0.66 \
  --symbolic-profile conservative
```

This produces:

* constraint impact table,
* leave-one-rule-out ablation metrics,
* constraint overlap table,
* qualitative examples,
* plots showing blocked false positives vs blocked true positives.

Outputs:

```text
outputs/tables/symbolic_analysis/{variant_id}/{run_name}/
outputs/figures/symbolic_analysis/{variant_id}/{run_name}/
```

The goal is to show that the symbolic layer is not decorative: some constraints materially reduce unsafe false merges, while overly aggressive constraints can hurt recall.

## Decision configuration sweep

After training the neural scorer and implementing symbolic governance, run:

```bash
python scripts/10_sweep_decision_configs.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg
```

The sweep compares:

- neural-only thresholding,
- neuro-symbolic default governance (`invalid_blocks`),
- stricter two-threshold governance (`strict_valid_or_high_confidence`).

The sweep varies neural threshold, symbolic hard-rule profile (conservative / moderate), and `tau_high` for uncertain cases in strict mode (accepted when `neural_score >= tau_high`).

Results:

```text
outputs/tables/sweeps/{variant_id}/{run_name}/decision_config_sweep.csv
```

Each row is one configuration on one split (valid and test by default), with classification metrics, governance diagnostics, and deltas vs neural-only at the same threshold.

## Sweep visualization

```bash
python scripts/11_plot_sweep_results.py \
  --variant pairwise_50_medium_unseen100 \
  --run-name neural_logreg
```

Figures: `outputs/figures/sweeps/{variant_id}/{run_name}/`

Key plots: `precision_recall_scatter_{split}`, `fp_vs_recall_{split}`, `fp_reduction_vs_recall_loss_{split}`, `top_configs_{split}`, `valid_vs_test_f1`, `threshold_f1_by_method_{split}` (PNG and PDF).

Selected top configurations: `outputs/tables/sweeps/{variant_id}/{run_name}/plot_selected_configs.csv`

## Project layout

```
data/raw/          # downloaded sources (HTML, archives, notes)
data/interim/      # unpacked or intermediate files
data/processed/    # experiment-ready tables
outputs/           # figures, tables, predictions
src/               # library code
scripts/           # runnable entry points
```
