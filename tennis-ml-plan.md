# Tennis ML — Win Probability Predictor: Plan

## Top-Level Overview

Build a machine learning web application that predicts the win probability of a
match between two ATP players. The dataset (`atp_tennis.csv`) contains historical
ATP match records from 2000 onwards. The app is built for a workshop setting —
beginner-friendly, minimal dependencies, end-to-end runnable in a single Streamlit
session.

**Core constraints:**
- Betting odds (`Odd_1`, `Odd_2`) are excluded from all features.
- Player names (`Player_1`, `Player_2`) are excluded from the model feature set;
  they are only used in the Streamlit UI to look up a player's most recent ranking.
- Features are expressed as *differences* (Player_1 minus Player_2) to avoid
  positional/assignment bias.
- Train/test/holdout split is **time-based**: train < 2022, test 2022–2023,
  holdout 2024+ (~50 most recent matches).

---

## Dataset Summary

**File:** `atp_tennis.csv`  
**Rows inspected:** first 15 (headers + 14 matches)  
**Date range starts:** 2000-01-03

| # | Column | Type | Represents |
|---|--------|------|------------|
| 1 | `Tournament` | string (high-cardinality) | Tournament name — drop in favour of `Series` |
| 2 | `Date` | date string (YYYY-MM-DD) | Match date — used for time-based splitting only |
| 3 | `Series` | categorical (low-cardinality) | Tournament tier: International, Grand Slam, Masters, etc. |
| 4 | `Court` | categorical | Indoor / Outdoor |
| 5 | `Surface` | categorical | Hard / Clay / Grass / Carpet |
| 6 | `Round` | categorical (ordinal) | Match round: 1st Round → 2nd Round → … → Final |
| 7 | `Best of` | integer (3 or 5) | Match format — useful signal (Grand Slams = best of 5) |
| 8 | `Player_1` | string | Player name — UI only, excluded from features |
| 9 | `Player_2` | string | Player name — UI only, excluded from features |
| 10 | `Winner` | string | Match winner — used to derive target label only |
| 11 | `Rank_1` | integer | ATP ranking of Player_1 at match time |
| 12 | `Rank_2` | integer | ATP ranking of Player_2 at match time |
| 13 | `Pts_1` | integer | Ranking points of Player_1 |
| 14 | `Pts_2` | integer | Ranking points of Player_2 |
| 15 | `Odd_1` | float | Betting odds Player_1 — EXCLUDED (leakage / constraint) |
| 16 | `Odd_2` | float | Betting odds Player_2 — EXCLUDED (leakage / constraint) |
| 17 | `Score` | string | Match score — post-match only, EXCLUDED (leakage) |

**Target variable:** Binary column `Player_1_wins` = 1 if `Winner == Player_1`, else 0.

### Leakage Flags

| Column | Reason |
|--------|--------|
| `Winner` | The label itself; never a feature |
| `Score` | Recorded after the match; directly encodes the winner |
| `Odd_1` / `Odd_2` | Market-consensus win probability; excluded per project constraint |

### Data Quality Issues

| Issue | Detail |
|-------|--------|
| Missing-value sentinel `-1` | `Pts_1`, `Pts_2` use `-1` for unknown (early 2000s); replace with NaN then impute or drop |
| Missing-value sentinel `-1.0` | `Odd_1`, `Odd_2` same pattern (moot — columns are dropped) |
| `Rank_1` / `Rank_2` sentinel | Likely also `-1` in some rows; needs verification and replacement |
| `Tournament` high-cardinality | Hundreds of names; drop column, use `Series` instead |
| `Round` is ordinal | String labels need a hand-crafted ordinal mapping before encoding |
| Player name high-cardinality | Thousands of unique player strings; excluded from model features |

---

## Recommended Tech Stack

| Concern | Choice | Why |
|---------|--------|-----|
| Language | Python 3.11 | Stable LTS, widest library support |
| Data handling | `pandas` 2.x | Standard; familiar to beginners |
| ML | `scikit-learn` 1.x | Random Forest, preprocessing, permutation importance all built-in |
| Model persistence | `joblib` | Ships with scikit-learn; one-line save/load |
| Web UI | **Streamlit** | Zero HTML/CSS required; Python-first; ideal for workshop demos |
| Visualisation | `matplotlib` | Built into most Python envs; sufficient for feature importance plots |
| Environment | `venv` + `requirements.txt` | Simplest possible — no Conda or Poetry complexity |
| Testing | None (workshop scope) | Out of scope for a learning workshop |

**Streamlit vs. Gradio:** Both are beginner-friendly Python UI frameworks.
Streamlit is chosen because it has richer layout primitives (columns, tabs, metrics),
a larger community for data-science demos, and direct support for pandas DataFrames
and matplotlib figures without adapters. Gradio is better suited to model-serving APIs.

---

## Project Structure

```
TennisX/
├── atp_tennis.csv               # Raw dataset (do not modify)
├── requirements.txt             # Pinned dependencies
├── tennis-ml-plan.md            # This plan file
│
├── data/
│   ├── processed_train.csv      # Cleaned, feature-engineered training set
│   ├── processed_test.csv       # Test set (2022–2023)
│   └── processed_holdout.csv    # Holdout set (2024+, ~50 matches)
│
├── models/
│   └── rf_model.joblib          # Saved Random Forest model + feature list
│
├── src/
│   ├── 01_data_preparation.py   # Module 1: clean, split, save processed CSVs
│   ├── 02_feature_engineering.py# Module 2: derive difference features, encode categoricals
│   ├── 03_model_building.py     # Module 3: train Random Forest, evaluate, save model
│   ├── 04_model_interpretation.py # Module 4: feature importance plots
│   └── 05_live_prediction.py    # Module 5: held-out challenge, compare to actuals
│
└── app/
    └── streamlit_app.py         # Module 6: Streamlit web UI
```

Each `src/` script is standalone and runnable in order (`01` → `05`).
The Streamlit app (`app/streamlit_app.py`) loads the saved model and processed data at runtime.

---

## Module-by-Module Build Plan

---

### Module 1 — Data Preparation

**Status:** `[x] done`

**Objective:** Load the raw CSV, strip all leaky and disallowed columns, handle
missing-value sentinels, derive the target label, and write three time-based splits
to `data/`.

**Key Steps:**
1. Load `atp_tennis.csv` with `pandas.read_csv`.
2. Parse `Date` as a datetime column.
3. Drop columns: `Tournament`, `Winner`, `Score`, `Odd_1`, `Odd_2`, `Player_1`, `Player_2`.
4. Replace sentinel value `-1` (integer and float) with `NaN` in `Rank_1`, `Rank_2`,
   `Pts_1`, `Pts_2`.
5. Drop rows where both ranks are missing (unresolvable).
6. Derive target: `Player_1_wins = (Winner == Player_1).astype(int)` — compute
   before dropping `Winner` and `Player_1`.
7. Apply time-based split:
   - **Train:** `Date < 2022-01-01`
   - **Test:** `2022-01-01 <= Date < 2024-01-01`
   - **Holdout:** `Date >= 2024-01-01` (full 2024–2026 period; Module 5 displays the 50 most recent as the "live challenge" sample)
8. Save `data/processed_train.csv`, `data/processed_test.csv`, `data/processed_holdout.csv`.
   Retain `Player_1`, `Player_2`, `Date` in the holdout CSV only (for the UI challenge display).

**Expected Output:** Three clean CSV files with consistent columns, no leaky fields,
no `-1` sentinels, and a `Player_1_wins` column.

**Relevant Context:** `atp_tennis.csv` lines 1–15 (confirmed columns and sentinels above).

---

### Module 2 — Feature Engineering

**Status:** `[x] done`

**Objective:** Transform the cleaned data into a numeric feature matrix ready for
the Random Forest. All features must be known *before* a match starts.

**Key Steps:**
1. Load the three processed CSVs from `data/`.
2. Create difference features:
   - `rank_diff = Rank_1 - Rank_2` (lower rank number = better player; negative means Player_1 is higher-ranked)
   - `pts_diff = Pts_1 - Pts_2`
3. Create ordinal encoding for `Round`:
   ```
   1st Round=1, 2nd Round=2, 3rd Round=3, 4th Round=4,
   Quarterfinals=5, Semifinals=6, The Final=7
   ```
4. One-hot encode `Surface`, `Court`, `Series` using `pd.get_dummies` with `drop_first=True`.
   Fit the encoding vocabulary on train only; apply the same columns to test and holdout
   (use `reindex(columns=train_columns, fill_value=0)` to align).
5. Keep `Best of` as-is (already numeric: 3 or 5).
6. Drop `Date`, `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2` after engineering differences.
7. Overwrite the processed CSVs with the engineered versions, or write new files
   (`engineered_train.csv`, etc.).

**Expected Output:** Numeric feature matrix with columns such as:
`rank_diff`, `pts_diff`, `Round_ord`, `Best of`,
`Surface_Clay`, `Surface_Grass`, `Surface_Carpet`,
`Court_Outdoor`, `Series_Grand Slam`, `Series_Masters`, …
plus target `Player_1_wins`.

**Relevant Context:** `Round` values observed: "1st Round"; others inferred from
standard ATP tournament structure. `Surface` values: Hard, Clay, Grass, Carpet.

---

### Module 3 — Model Building

**Status:** `[ ] pending`

**Objective:** Train a Random Forest classifier on the engineered training set,
evaluate it on the test set, and persist the model.

**Key Steps:**
1. Load `engineered_train.csv` and `engineered_test.csv`.
2. Separate features `X` and target `y = Player_1_wins`.
3. Instantiate `RandomForestClassifier(n_estimators=200, random_state=42)` as the baseline.
4. Fit on `X_train`, `y_train`.
5. Evaluate on `X_test`, `y_test`:
   - Accuracy
   - Classification report (precision, recall, F1)
   - `predict_proba` for log-loss (optional for workshop)
6. Save the trained model and the feature column list together using `joblib.dump`
   to `models/rf_model.joblib`.
7. Print a brief summary of results to stdout.

**Expected Output:** `models/rf_model.joblib` with the fitted model; printed
evaluation metrics on the test set.

**Relevant Context:** Feature columns are determined in Module 2.
The model must store the feature column order so the Streamlit app and Module 5
can align inputs correctly.

---

### Module 4 — Model Interpretation

**Status:** `[x] done`

**Objective:** Compare two feature importance methods side-by-side so workshop
participants understand their differences.

**Key Steps:**
1. Load the saved model from `models/rf_model.joblib`.
2. Load `engineered_test.csv` for permutation importance evaluation.
3. Compute **built-in importance:** `model.feature_importances_` (mean decrease in impurity).
4. Compute **permutation importance:** `sklearn.inspection.permutation_importance`
   on the test set (`n_repeats=10`, `random_state=42`).
5. Plot both as horizontal bar charts side-by-side using `matplotlib`.
6. Annotate each chart with a one-line explanation of what the method measures.
7. Save the figure as `models/feature_importance_comparison.png`.
8. Print top-5 features by each method.

**Expected Output:** `models/feature_importance_comparison.png`; console output
of top-5 features per method.

**Relevant Context:** Permutation importance is computed on the test set (not train),
which avoids overestimating importance of features the model memorised.

---

### Module 5 — Live Prediction Challenge

**Status:** `[x] done`

**Objective:** Simulate a "live" prediction by running the model on the held-out
matches, then reveal the actual result. This is the workshop's payoff moment.

**Key Steps:**
1. Load `engineered_holdout.csv` (feature-engineered holdout without `Player_1_wins` label).
   Also load the original holdout metadata CSV (which retains `Player_1`, `Player_2`, `Winner`,
   `Date`, `Tournament`) for display purposes.
2. Load `models/rf_model.joblib`.
3. Run `model.predict_proba(X_holdout)` to get `P(Player_1 wins)` for each match.
4. Display a table: `Player_1 | Player_2 | Date | P(P1 wins) | Predicted Winner | Actual Winner | Correct?`
5. Compute and print overall accuracy on the holdout set.
6. Highlight any high-confidence mispredictions (model > 70% confident but wrong).

**Expected Output:** Printed or saved table of holdout predictions vs. actuals;
holdout accuracy metric; a discussion prompt about model confidence calibration.

**Note on "Live" constraint:** The dataset is historical. The holdout set is the
most recent ~50 matches from 2024+. They were unseen during training and represent
the closest proxy to "live" predictions available without a real-time data feed.

---

### Module 6 — Streamlit Web UI

**Status:** `[ ] pending`

**Objective:** Provide an interactive web dashboard where a user selects two ATP
players by name and immediately sees their predicted win probabilities.

**Key Steps:**
1. At app startup, load:
   - `models/rf_model.joblib` (model + feature column list)
   - `atp_tennis.csv` (to build a player → most-recent-rank lookup table)
2. Build a lookup dictionary: for each unique player name, find their most recent
   `Rank` and `Pts` from the latest date row that features them as Player_1 or Player_2.
3. Render the UI:
   - **Sidebar or top section:** Two `st.selectbox` dropdowns — "Select Player 1" and "Select Player 2"
     populated with all known player names (sorted alphabetically).
   - **Context inputs:** `st.selectbox` for Surface, Court, Series, Round, Best of —
     with sensible defaults (Hard, Outdoor, International, 1st Round, 3).
   - **Predict button:** `st.button("Predict")`
4. On predict:
   - Look up ranks and points for both selected players.
   - Compute `rank_diff`, `pts_diff`.
   - Encode the context inputs (Surface, Court, Series, Round, Best of) using the
     same encoding logic as Module 2 — align to the saved feature column list.
   - Call `model.predict_proba([features])`.
   - Display: `st.metric` for P(Player 1 wins) and P(Player 2 wins), plus a
     colour-coded winner call.
5. Add a brief explanation panel: "How does this model work?" — one paragraph.
6. Add a disclaimer: model is trained on historical data only; betting odds were
   excluded; rankings used are from the player's last recorded match.

**Expected Output:** A running Streamlit app (`streamlit run app/streamlit_app.py`)
that shows win probabilities for any two player matchup in the dataset.

**Relevant Context:** Player names must NOT flow into the model feature vector.
The lookup step converts name → rank/pts, then only the numeric/encoded features
are passed to the model.

---

## Assumptions & Open Questions

### Assumptions

1. **Player ordering is arbitrary.** The dataset does not consistently assign the
   "favourite" or "home player" to `Player_1`. All positional bias is removed by
   using only difference features.
2. **Rankings at match time are accurate.** `Rank_1`/`Rank_2` reflect the ATP ranking
   at the time of the match, making them safe pre-match features.
3. **Most-recent rank is a reasonable proxy.** When a user selects a player in the
   Streamlit UI, their most recent rank in the dataset is used. This may be stale
   for retired or inactive players.
4. **Betting odds are excluded entirely.** Per project constraint — not used in any
   module, including as a comparison baseline.
5. **`Best of` is determined by `Series`/`Round`.** Grand Slam matches past the
   round of 16 are best of 5; others are best of 3. The column already encodes this.
6. **No real-time data feed.** The "live prediction challenge" uses the 2024+ holdout
   as the closest available proxy.
7. **Workshop scope — no hyperparameter search.** A fixed `RandomForestClassifier`
   with `n_estimators=200` is the baseline. Grid search is out of scope.

### Open Questions / Decisions for the User

| # | Question | Default Assumption |
|---|----------|--------------------|
| Q1 | Should rows with missing ranks (`-1` sentinel) be dropped entirely or imputed with the median rank? | Drop rows where either rank is missing |
| Q2 | Should `Pts_1`/`Pts_2` be dropped if too many are missing in early years, or imputed? | Impute median; fall back to `pts_diff=0` if both missing |
| Q3 | Should the Streamlit app allow users to override the rank manually (slider) rather than relying on the lookup? | No — use lookup only for simplicity |
| Q4 | Should `Tournament` (high-cardinality) be used at all, or always dropped in favour of `Series`? | Always drop `Tournament` |
| Q5 | Is Carpet surface still present in modern ATP data? (ATP retired Carpet in 2009) | Treat as a valid category for historical rows; it won't appear for new predictions |

---

## Risks & Trade-Offs

### Risk 1 — Data Leakage via Time (Most Critical)

**Risk:** A random train/test split would allow the model to train on future matches
and test on past ones, inflating accuracy because the model implicitly learns
player-level trends from future data.

**Recommended approach:** Time-based split (train < 2022, test 2022–2023, holdout
2024+) as confirmed. Never shuffle before splitting.

---

### Risk 2 — Rank Symmetry / Positional Bias

**Risk:** If raw `Rank_1` and `Rank_2` are used as separate features, the model
may learn a spurious association between the "Player_1 slot" and winning, because
the dataset assignment is not random across all eras.

**Recommended approach:** Use only `rank_diff = Rank_1 - Rank_2` and
`pts_diff = Pts_1 - Pts_2`. The model only sees relative strength, not slot identity.

---

### Risk 3 — High-Cardinality Encoding and Train/Test Column Mismatch

**Risk:** `pd.get_dummies` on training data creates different columns than on test
data if some categories only appear in one split (e.g., a surface type that only
appears in recent years). This causes a shape mismatch when scoring.

**Recommended approach:** Fit the encoding schema on the training set only. After
encoding test/holdout, call `reindex(columns=train_feature_columns, fill_value=0)`
to guarantee identical column sets. Store the canonical feature column list in
`rf_model.joblib` alongside the model object.

---

## Implementation Order

```
Module 1 (data_preparation.py)
  → Module 2 (feature_engineering.py)
    → Module 3 (model_building.py)
      → Module 4 (model_interpretation.py)
      → Module 5 (live_prediction.py)
      → Module 6 (streamlit_app.py)
```

Modules 4, 5, and 6 all depend on the saved model from Module 3 but are
otherwise independent of each other.
