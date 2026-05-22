# Final Comparison Figures — Results Summary

**Dataset variant:** `pairwise_50_medium_unseen100`  
**Run:** `neural_logreg`  
**Selection protocol:** Each configuration is chosen by **validation F1**; reported metrics are **test-set** performance (precision, recall, F1).

**Figures location:** `outputs/figures/final_comparison/pairwise_50_medium_unseen100/neural_logreg/`

---

## Figure 1 — Baselines vs neuro-symbolic method

**Title:** *Baselines vs neuro-symbolic method*  
**Purpose:** Compare naive and lexical baselines against neural-only scoring and the proposed conservative neuro-symbolic pipeline.

| Method | Description | Test F1 | Test precision | Test recall |
|--------|-------------|---------|----------------|-------------|
| Always negative | Predict no duplicate pairs for every example. | 0.000 | 0.000 | 0.000 |
| Exact title | Predict duplicate only when normalized product titles match exactly. | 0.000 | 0.000 | 0.000 |
| BoW cosine | Bag-of-words cosine similarity with validation-selected threshold (0.29). | 0.317 | 0.212 | 0.632 |
| TF-IDF word | Word-level TF-IDF cosine similarity (threshold 0.21). | 0.370 | 0.255 | 0.674 |
| TF-IDF char | Character n-gram TF-IDF cosine similarity (threshold 0.35). | 0.404 | 0.294 | 0.648 |
| Field-wise heuristic | Hand-crafted lexical agreement over brand, title, model, quantity, and price fields (threshold 0.52). | 0.366 | 0.261 | 0.612 |
| neural-only | Logistic regression on TF-IDF features; accept if score ≥ 0.66 (validation-selected). | 0.410 | 0.298 | 0.656 |
| neuro-symbolic (conservative) | Same neural scorer (0.66) with conservative symbolic governance: invalid constraint blocks force non-duplicate. | **0.452** | **0.380** | 0.556 |

**Takeaway:** Lexical baselines peak near test F1 ≈ 0.40. Neural-only improves recall but remains imprecise. Adding conservative symbolic governance raises precision and test F1 to **0.452**, the best result on this figure.

---

## Figure 2 — Variants of the neuro-symbolic method

**Title:** *Variants of the neuro-symbolic method*  
**Purpose:** Ablations, governance variants, soft-risk experiments, hard-negative training, and structured neural models relative to the proposed base method.

| Method | Description | Test F1 | Test precision | Test recall |
|--------|-------------|---------|----------------|-------------|
| neuro-symbolic (conservative) | Proposed base: invalid_blocks + conservative profile, threshold 0.66. | 0.452 | 0.380 | 0.556 |
| neuro-symbolic (moderate) | Same pipeline with moderate symbolic profile (threshold 0.50). | 0.372 | 0.335 | 0.418 |
| two threshold governance | Strict valid-or-high-confidence mode with τ_high = 0.80 and threshold 0.66. | 0.457 | 0.411 | 0.516 |
| soft symbolic risk | Continuous symbolic risk penalty on neural scores (λ = 1, threshold 0.66). | 0.438 | 0.342 | 0.608 |
| symbolic risk gate | Soft gate with ρ = 0.1; defers to symbolic veto when risk is high. | 0.452 | 0.380 | 0.556 |
| symbolic hard negatives (iter 3) | Neural scorer retrained after symbolic hard-negative mining; best among iterations ≥ 1 by validation F1. | 0.129 | 0.331 | 0.080 |
| symbolic hard negatives + neuro-symbolic (iter 3) | Iteration-3 scorer with conservative governance; best HN+governed iteration ≥ 1. | 0.124 | 0.456 | 0.072 |
| field attention | Structured field-attention model (validation threshold 0.57). | **0.478** | 0.395 | 0.606 |
| structured transformer | Transformer over structured field tokens (threshold 0.49). | 0.471 | 0.367 | 0.654 |
| field attention + neuro-symbolic | Field-attention scores with conservative symbolic governance. | 0.472 | 0.401 | 0.574 |
| structured transformer + neuro-symbolic | Structured transformer with conservative governance. | 0.468 | 0.377 | 0.618 |

**Takeaway:** Among governance variants, **two threshold governance** slightly edges the conservative base on test F1 (0.457 vs 0.452) with higher precision and lower recall. Soft-risk variants do not beat hard veto on test F1. Hard-negative retraining (iter ≥ 1) severely hurts recall. **Field attention** achieves the highest test F1 on this figure (0.478) without governance; governed structured variants sit near 0.47.

---

## Figure 3 — Why combine neural scoring with symbolic governance?

**Title:** *Why combine neural scoring with symbolic governance?*  
**Purpose:** Motivate the pipeline as a progression from lexical similarity → symbolic constraints → neural scoring → combined neuro-symbolic method.

| Method | Description | Test F1 | Test precision | Test recall |
|--------|-------------|---------|----------------|-------------|
| best lexical baseline | Best validation-F1 lexical baseline (TF-IDF char, threshold 0.35). | 0.404 | 0.294 | 0.648 |
| symbolic-only | Accept pair only if symbolic constraint check returns valid (no learned score). | 0.392 | 0.297 | 0.578 |
| neural-only | Logistic regression threshold 0.66 without symbolic layer. | 0.410 | 0.298 | 0.656 |
| neuro-symbolic (conservative) | Neural score plus conservative invalid_blocks governance (threshold 0.66). | **0.452** | **0.380** | 0.556 |

**Takeaway:** Symbolic-only rules alone underperform the best lexical baseline on test F1. Neural-only adds little over lexical similarity on F1 but keeps high recall. Combining both yields the best test F1 (**0.452**) and the best precision (**0.380**), with fewer false positives than neural-only (453 vs 771 FP at test).

---

## Cross-figure notes

- Metrics are rounded to three decimals in plots; table values match `figure*_selected_rows.csv`.
- Hard-negative rows exclude iteration 0 (untrained baseline).
- Governed structured-model rows in Figure 2 use the model’s validation threshold; validation F1 is not always logged in the governed comparison table.
