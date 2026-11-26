# Phase 1 Implementation Log — Data Preparation

**Module:** 1 — Data Preparation  
**Script:** `src/01_data_preparation.py`  
**Status:** ✅ Complete  
**Date:** 2026-11-26

---

## Objective

Load `atp_tennis.csv`, remove leaky/disallowed columns, handle missing-value
sentinels, derive the binary target label, apply a time-based split, and save
three clean CSVs to `data/`.

---

## Steps Executed

| # | Step | Result |
|---|------|--------|
| 1 | Load raw CSV | 69,100 rows × 17 columns loaded successfully |
| 2 | Parse `Date` as datetime | Range: 2000-01-03 → 2026-11-01 |
| 3 | Derive target label | `Player_1_wins` computed before dropping name columns |
| 4 | Build holdout metadata frame | Preserves `Player_1`, `Player_2`, `Winner`, `Tournament` for Module 5 display |
| 5 | Replace `-1` / `-1.0` sentinels with `NaN` | Applied to `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2` |
| 6 | Drop leaky/disallowed columns | Dropped: `Tournament`, `Winner`, `Score`, `Odd_1`, `Odd_2`, `Player_1`, `Player_2` |
| 7 | Drop rows with missing ranks | 26 rows dropped (< 0.1%) |
| 8 | Time-based split | Train / Test / Holdout as per plan |
| 9 | Save CSVs | 3 files written to `data/` |
| 10 | Sanity checks | All passed: no missing ranks, target column present in all 3 files |

---

## Test Results

```
Raw shape: (69100, 17)
Date range: 2000-01-03 → 2026-11-01
Target distribution — Player_1 wins: 34550 / 69100 (50.0%)

Missing values after sentinel replacement:
  Rank_1:  14 NaN rows (0.0%)
  Rank_2:  12 NaN rows (0.0%)
  Pts_1:  15652 NaN rows (22.7%)
  Pts_2:  15653 NaN rows (22.7%)

Dropped 26 rows with missing Rank_1 or Rank_2 (0.0% of data)
Remaining rows: 69,074

Split summary:
  Train   (< 2022):     56,384 rows  (2000-01-03 → 2021-11-21)
  Test    (2022–2023):   5,152 rows  (2022-01-03 → 2023-12-31)
  Holdout (2024+):       7,538 rows  (2024-01-01 → 2026-11-01)

Sanity Checks:
  Train    rows=56,384  columns=11  missing_ranks=0  target_col=True
  Test     rows= 5,152  columns=11  missing_ranks=0  target_col=True
  Holdout  rows= 7,538  columns=15  missing_ranks=0  target_col=True
```

---

## Output Files

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `data/processed_train.csv` | 56,384 | 11 | Features + target only; no player names |
| `data/processed_test.csv` | 5,152 | 11 | Features + target only; no player names |
| `data/processed_holdout.csv` | 7,538 | 15 | Includes `Player_1`, `Player_2`, `Winner`, `Date`, `Tournament` for Module 5 display |

**Train columns:** `Date, Series, Court, Surface, Round, Best of, Rank_1, Rank_2, Pts_1, Pts_2, Player_1_wins`

---

## Deviations from Plan

| # | Plan Assumption | Actual Result | Resolution |
|---|----------------|---------------|------------|
| 1 | Holdout ~50 rows | Holdout = 7,538 rows (dataset extends to 2026-11-01) | Kept full 2024+ period as holdout. Module 5 will display the 50 most recent as the "live challenge" sample. Plan file updated. |
| 2 | `pandas` `inplace=True` used in draft | Pandas 4.x deprecation warnings | Fixed: replaced all `inplace=True` calls with assignment form (`df = df.drop(...)`) |

---

## Notable Data Findings

| Finding | Detail |
|---------|--------|
| **Perfectly balanced target** | Exactly 50.0% Player_1 wins — dataset is symmetric by design (each match appears once with arbitrary Player_1/Player_2 assignment) |
| **`Pts_1`/`Pts_2` missing 22.7%** | Missing in early years (pre-2003 era). Sentinels replaced with NaN. Module 2 will impute with training median. |
| **Only 26 rows missing both ranks** | Negligible data loss (0.04%). Safe to drop. |
| **Dataset is current** | Extends to 2026-11-01, well past the original planning assumption. Holdout is a genuinely large recent period. |

---

## Environment

- Python venv created: `d:\TennisX\venv\`
- Packages installed: `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `streamlit`
- No pip warnings, no errors on final run
