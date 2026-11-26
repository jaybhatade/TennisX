# Phase 4 Implementation Log — Model Interpretation

**Module:** 4 — Model Interpretation  
**Script:** `src/04_model_interpretation.py`  
**Status:** ✅ Complete  
**Date:** 2026-11-26

---

## Objective

Load the saved Random Forest model and the engineered test set, compute two
complementary feature-importance measures, produce a side-by-side comparison
chart, and print the top-5 features for each method.

---

## Steps Executed

| # | Step | Result |
|---|------|--------|
| 1 | Load `models/rf_model.joblib` (model + feature_cols) | Model loaded; 15 feature columns confirmed |
| 2 | Load `engineered_test.csv` | 5,152 × 16 |
| 3 | Compute built-in (impurity) importance via `model.feature_importances_` | Completed instantly |
| 4 | Compute permutation importance (`n_repeats=10`, `random_state=42`, `n_jobs=-1`) on test set | Completed (~2 min) |
| 5 | Print top-5 features for both methods | See output below |
| 6 | Plot side-by-side horizontal bar chart and save | `models/feature_importance_comparison.png` |

---

## Run Output

```
Loading model …
Loading engineered test set …
  Test shape: (5152, 16)

Computing built-in (impurity) importance …
Computing permutation importance (n_repeats=10) — this may take a minute …

--- Top-5 Features: Built-in (Impurity) Importance ---
  rank_diff                            0.4950
  pts_diff                             0.4269
  Round_ord                            0.0427
  Court_Outdoor                        0.0058
  Series_International                 0.0043

--- Top-5 Features: Permutation Importance (Test Set) ---
  rank_diff                            0.0626
  pts_diff                             0.0438
  Court_Outdoor                        0.0007
  Round_ord                            0.0005
  Series_Masters Cup                   0.0001

Generating comparison plot …
  Saved → D:\TennisX\models\feature_importance_comparison.png

✓ Phase 4 complete — feature importance comparison saved.
```

---

## Output Files

| File | Notes |
|------|-------|
| `models/feature_importance_comparison.png` | Side-by-side horizontal bar chart (150 dpi) |

---

## Feature Importance Results

### Built-in (Impurity-Based) Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `rank_diff` | 0.4950 |
| 2 | `pts_diff` | 0.4269 |
| 3 | `Round_ord` | 0.0427 |
| 4 | `Court_Outdoor` | 0.0058 |
| 5 | `Series_International` | 0.0043 |

### Permutation Importance (Test Set)

| Rank | Feature | Mean Accuracy Drop |
|------|---------|-------------------|
| 1 | `rank_diff` | 0.0626 |
| 2 | `pts_diff` | 0.0438 |
| 3 | `Court_Outdoor` | 0.0007 |
| 4 | `Round_ord` | 0.0005 |
| 5 | `Series_Masters Cup` | 0.0001 |

---

## Notable Findings

| Finding | Detail |
|---------|--------|
| **Both methods agree on top-2** | `rank_diff` and `pts_diff` are the dominant features under both metrics — consistent with Phase 3's impurity top-5 |
| **Impurity vs. permutation scale differs** | Impurity assigns 92% weight to rank/pts; permutation shows they cause ~6-4% accuracy drop when shuffled — a more calibrated view of real-world impact |
| **Context features contribute little** | `Round_ord`, `Court_Outdoor`, and series features rank 3–5 in both methods but with very small absolute values; they add marginal but measurable signal |
| **No negative permutation importances** | No feature actively hurts model performance; all features contribute ≥ 0, though many near zero |
| **Chart annotation** | Negative permutation importance bars are coloured red (none present here); a dashed zero-line is included for reference |

---

## Deviations from Plan

| # | Plan | Actual | Resolution |
|---|------|--------|------------|
| 1 | "Annotate each chart with a one-line explanation" | Added as subplot titles (second line of each title) | Equivalent; cleaner than text annotations inside the bars |
| 2 | `matplotlib.use("Agg")` not mentioned in plan | Added | Required for headless/non-interactive environments; no functional difference |
