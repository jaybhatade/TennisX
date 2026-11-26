# Phase 2 Implementation Log — Feature Engineering

**Module:** 2 — Feature Engineering  
**Script:** `src/02_feature_engineering.py`  
**Status:** ✅ Complete  
**Date:** 2026-11-26

---

## Objective

Load the three processed CSVs from `data/`, engineer a numeric feature matrix
ready for the Random Forest, and write three engineered CSVs to `data/`.

---

## Steps Executed

| # | Step | Result |
|---|------|--------|
| 1 | Load processed CSVs | Train 56,384 × 11, Test 5,152 × 11, Holdout 7,538 × 15 |
| 2 | Impute `Pts_1` / `Pts_2` with training medians | Median: Pts_1=876, Pts_2=880; 0 NaN remaining in all splits |
| 3 | Create `rank_diff` and `pts_diff` | Applied to all three splits |
| 4 | Ordinal-encode `Round` → `Round_ord` | All 8 Round values mapped; 0 unmapped values |
| 5 | One-hot encode `Surface`, `Court`, `Series` | Fit on train only; 11 OHE columns created; test/holdout aligned via `reindex` |
| 6 | Drop raw columns (`Date`, `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2`, `Round`) | `Date` retained in holdout for Module 5 display |
| 7 | Save engineered CSVs | 3 files written to `data/` |
| 8 | Sanity checks | 0 feature NaN in all splits; target column present in all 3 files |

---

## Run Output

```
Loading processed CSVs …
  Train   shape: (56384, 11)
  Test    shape: (5152, 11)
  Holdout shape: (7538, 15)

Imputing missing Pts values with training medians …
  Pts_1 median (train): 876
  Pts_2 median (train): 880
  Train      remaining NaN in rank/pts: 0
  Test       remaining NaN in rank/pts: 0
  Holdout    remaining NaN in rank/pts: 0

Creating difference features …
Ordinal-encoding Round …
  All Round values mapped successfully.
One-hot encoding Surface, Court, Series …
  OHE columns created from train: ['Surface_Clay', 'Surface_Grass', 'Surface_Hard',
  'Court_Outdoor', 'Series_ATP500', 'Series_Grand Slam', 'Series_International',
  'Series_International Gold', 'Series_Masters', 'Series_Masters 1000', 'Series_Masters Cup']
Dropping raw numeric and date columns …
  ...

--- Sanity Checks ---

Feature columns (15): Best of, rank_diff, pts_diff, Round_ord, Surface_Clay,
Surface_Grass, Surface_Hard, Court_Outdoor, Series_ATP500, Series_Grand Slam,
Series_International, Series_International Gold, Series_Masters,
Series_Masters 1000, Series_Masters Cup

  Train       rows=56,384  cols=16  feature_NaN=0  target_col=True
  Test        rows= 5,152  cols=16  feature_NaN=0  target_col=True
  Holdout     rows= 7,538  cols=17  feature_NaN=0  target_col=True

✓ Phase 2 complete — feature engineering done.
```

---

## Output Files

| File | Rows | Cols | Notes |
|------|------|------|-------|
| `data/engineered_train.csv` | 56,384 | 16 | 15 features + `Player_1_wins` target |
| `data/engineered_test.csv` | 5,152 | 16 | Same 15 features + target; aligned to train columns |
| `data/engineered_holdout.csv` | 7,538 | 17 | 15 features + target + `Date`, `Tournament`, `Player_1`, `Player_2` for Module 5 display |

**Feature columns (15):**
`Best of`, `rank_diff`, `pts_diff`, `Round_ord`,
`Surface_Clay`, `Surface_Grass`, `Surface_Hard`,
`Court_Outdoor`,
`Series_ATP500`, `Series_Grand Slam`, `Series_International`,
`Series_International Gold`, `Series_Masters`, `Series_Masters 1000`, `Series_Masters Cup`

---

## Deviations from Plan

| # | Plan Assumption | Actual Result | Resolution |
|---|----------------|---------------|------------|
| 1 | `Round Robin` not in plan's ordinal map | Present in real data (ATP Finals format) | Added `Round Robin=5` (same level as Quarterfinals — precedes knockout stage) |
| 2 | `drop_first=True` removes one dummy per category | `Surface_Hard`, `Court_Indoor`, `Series_ATP250` are the reference/dropped categories | Expected — `get_dummies(drop_first=True)` removes alphabetical-first category per group |
| 3 | `inplace=True` used in draft | Pandas 4.x `Pandas4Warning` deprecation | Fixed: replaced with assignment form (`df = df.drop(...)`) |

---

## Notable Findings

| Finding | Detail |
|---------|--------|
| **`Series` naming changed over time** | Pre-~2015 values: `International`, `International Gold`, `Masters`, `Masters Cup`. Post-~2015: `ATP250`, `ATP500`, `Masters 1000`. Both eras co-exist in training data; all values are captured in OHE columns. |
| **`Pts_1`/`Pts_2` missing 22.7% in train** | Early-era rows (pre-2003). Imputed with training-set medians (876 / 880 pts) — well-matched, since the target is balanced at exactly 50%. |
| **15 clean feature columns** | All numeric, no NaN; ready for Random Forest without further preprocessing. |
| **Column alignment verified** | Test and holdout OHE columns exactly match train via `reindex(fill_value=0)`. |
