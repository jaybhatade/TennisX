# Phase 3 Implementation Log — Model Building

**Module:** 3 — Model Building  
**Script:** `src/03_model_building.py`  
**Status:** ✅ Complete  
**Date:** 2026-11-26

---

## Objective

Load the engineered train and test CSVs, train a `RandomForestClassifier`,
evaluate it on the test set, and persist the fitted model together with its
feature column list to `models/rf_model.joblib`.

---

## Steps Executed

| # | Step | Result |
|---|------|--------|
| 1 | Load `engineered_train.csv` and `engineered_test.csv` | Train 56,384 × 16, Test 5,152 × 16 |
| 2 | Separate 15 feature columns from target `Player_1_wins` | Feature list confirmed; metadata guard in place |
| 3 | Train `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)` | Completed on 56,384 rows × 15 features |
| 4 | Evaluate on test set (accuracy, log-loss, classification report, top-5 features) | See metrics below |
| 5 | Save `{"model": clf, "feature_cols": feature_cols}` to `models/rf_model.joblib` | 483.9 MB |

---

## Run Output

```
Loading engineered CSVs …
  Train  shape: (56384, 16)
  Test   shape: (5152, 16)

Feature columns (15):
  Best of, rank_diff, pts_diff, Round_ord,
  Surface_Clay, Surface_Grass, Surface_Hard, Court_Outdoor,
  Series_ATP500, Series_Grand Slam, Series_International,
  Series_International Gold, Series_Masters, Series_Masters 1000, Series_Masters Cup

Train samples : 56,384  (target balance: 0.500)
Test  samples :  5,152  (target balance: 0.500)

Training RandomForestClassifier(n_estimators=200, random_state=42) …
  Training complete.

--- Test-Set Evaluation ---

  Accuracy : 0.6095  (60.95%)
  Log-loss : 0.7134

  Classification Report:
                   precision    recall  f1-score   support
  Player_2 wins       0.61      0.62      0.61      2576
  Player_1 wins       0.61      0.60      0.61      2576
       accuracy                           0.61      5152

  Top-5 features (impurity importance):
    rank_diff                       0.4950
    pts_diff                        0.4269
    Round_ord                       0.0427
    Court_Outdoor                   0.0058
    Series_International            0.0043

Saving model to models/rf_model.joblib …
  Saved — file size: 483.9 MB

✓ Phase 3 complete — model trained and saved.
```

---

## Output Files

| File | Size | Notes |
|------|------|-------|
| `models/rf_model.joblib` | 483.9 MB | Dict: `{"model": RFC, "feature_cols": [15 cols]}` |

---

## Model Metrics Summary

| Metric | Value |
|--------|-------|
| Test accuracy | **60.95%** |
| Log-loss | 0.7134 |
| Precision (class 0) | 0.61 |
| Precision (class 1) | 0.61 |
| Recall (class 0) | 0.62 |
| Recall (class 1) | 0.60 |
| F1 (macro avg) | 0.61 |

Target class balance is perfectly even (50/50 in both train and test),
so a naïve baseline would be 50%. The model achieves ~61%, a meaningful
improvement using only pre-match features (ranking differences, surface, round, etc.).

---

## Feature Importance (Impurity-Based, Top 5)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `rank_diff` | 0.4950 |
| 2 | `pts_diff` | 0.4269 |
| 3 | `Round_ord` | 0.0427 |
| 4 | `Court_Outdoor` | 0.0058 |
| 5 | `Series_International` | 0.0043 |

`rank_diff` and `pts_diff` together account for **~92% of impurity-based importance**,
which is expected — relative ranking strength is the dominant signal in tennis.
Categorical context features (surface, court, series) contribute a small but
measurable share; permutation importance in Module 4 may revise this picture.

---

## Deviations from Plan

| # | Plan | Actual | Resolution |
|---|------|--------|------------|
| 1 | Plan did not specify `n_jobs` | Added `n_jobs=-1` | Uses all CPU cores; reduces wall-clock training time with no effect on results |
| 2 | Log-loss described as "optional" | Included | Minimal cost; useful reference for Module 4 calibration discussion |

---

## Notable Findings

| Finding | Detail |
|---------|--------|
| **61% accuracy on balanced classes** | Significantly above 50% baseline; ranking differences are the main driver |
| **Perfectly balanced target** | 50.0% `Player_1_wins` in both train and test — no class imbalance issue |
| **Rank + points dominate** | `rank_diff` + `pts_diff` = 92% of impurity importance; surface/round/series add smaller but real signal |
| **Model file 484 MB** | 200 trees × ~56K training samples; expected size for a full scikit-learn RF |
