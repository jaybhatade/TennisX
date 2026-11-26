# Phase 6 Implementation Log — Streamlit Web UI

**Module:** 6 — Streamlit Web UI  
**Script:** `app/streamlit_app.py`  
**Status:** ✅ Complete  
**Date:** 2026-11-26

---

## Objective

Build an interactive Streamlit web application where a user selects two ATP
players by name and immediately sees their modelled win probabilities, along
with match context controls and explanatory copy.

---

## Steps Executed

| # | Step | Result |
|---|------|--------|
| 1 | Read `models/rf_model.joblib` to confirm 15 feature columns | Confirmed; feature_cols cached via `@st.cache_resource` |
| 2 | Build `load_player_lookup()` from `atp_tennis.csv` | 1,822 unique players; most-recent-row-wins strategy using a combined sorted frame |
| 3 | Player selector — two `st.selectbox` dropdowns, sorted alphabetically | Defaults to Djokovic N. (P1) and Alcaraz C. (P2) if present |
| 4 | Display last-known rank for each player as `st.metric` | Shows rank immediately below the player dropdowns |
| 5 | Match context inputs — Surface, Court, Series, Round, Best of | 3-column layout; sensible defaults (Hard, Outdoor, International, 1st Round, 3) |
| 6 | `build_feature_row()` — encodes inputs into the 15-column feature vector | Replicates Module 2 OHE logic exactly (baseline = Carpet / Indoor / ATP250) |
| 7 | Predict button calls `model.predict_proba(feature_row)` | Returns `[P(P2 wins), P(P1 wins)]`; both displayed as `st.metric` |
| 8 | Confidence label (High / Moderate / Low) added below metrics | Threshold: ≥75% High, ≥60% Moderate, else Low |
| 9 | "Show feature inputs" expander for transparency | Displays the raw 15-column feature row sent to the model |
| 10 | "How does this model work?" section with feature table | One-paragraph explanation + metric table + accuracy figures |
| 11 | Disclaimer panel | Historical data only; betting odds excluded; workshop-only warning |
| 12 | Smoke-test script verifies all logic without Streamlit | Djokovic (rank 12) vs Alcaraz (rank 3), Grand Slam SF: P(Alcaraz) = 84% ✓ |

---

## Smoke-Test Results

```
Model loaded — 15 features
Player lookup built — 1,822 unique players
Djokovic N.: rank=12, pts=3090.0
Alcaraz C.: rank=3,  pts=5610.0
P(Djokovic N. wins): 16.0%
P(Alcaraz C. wins):  84.0%

All logic checks passed!
```

Alcaraz (#3, 5610 pts) is correctly predicted as the heavy favourite over
Djokovic (#12, 3090 pts) in a Grand Slam semifinal — consistent with rankings.

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| `@st.cache_resource` for model | Model is a large sklearn object; load once per session |
| `@st.cache_data` for player lookup | DataFrame-derived dict; deterministic; cache until app restarts |
| Combined-frame approach for lookup | Sort all P1 + P2 appearances by date, iterate once → last row wins automatically; avoids two-pass logic |
| OHE baseline in `build_feature_row` | Baseline categories (Carpet / Indoor / ATP250) produce all-zero OHE columns, matching `drop_first=True` from Module 2 |
| Player names never enter feature vector | Lookup converts name → rank/pts; only numeric/encoded features passed to `predict_proba` |
| `pd.notna()` guard on `pts` | Some players may have NaN pts (early-era records); safe fallback to 0.0 |

---

## UI Structure

```
TennisX — ATP Win Probability Predictor
│
├── Players
│   ├── [Player 1 selectbox]      [Player 2 selectbox]
│   └── [P1 Last Known Rank]      [P2 Last Known Rank]
│
├── Match Context
│   ├── Surface / Court     Series           Round / Best of
│
├── [⚡ Predict button]
│
├── Prediction (shown after Predict)
│   ├── P(P1 wins) metric    P(P2 wins) metric
│   ├── Predicted winner + confidence label
│   └── [Show feature inputs expander]
│
├── How does this model work?
│   └── Feature table + accuracy figures
│
└── Disclaimer
```

---

## Deviations from Plan

| # | Plan | Actual | Resolution |
|---|------|--------|------------|
| 1 | "Sidebar or top section" for dropdowns | Used main-area columns (not sidebar) | Cleaner for `layout="centered"`; sidebar would be wasted space |
| 2 | Plan implied separate P1 / P2 probability display | Used `st.metric` with ±delta from 50% | Delta from 50% conveys edge more clearly than raw probability alone |
| 3 | Plan did not mention rank display | Added `st.metric` for last-known rank of each player | Useful for the user to sanity-check the lookup before predicting |
| 4 | Plan said "colour-coded winner call" | Used `st.info` with confidence label (green/yellow/red emoji) | Equivalent effect within Streamlit's theming constraints |

---

## Launch Instructions

```bash
# From the TennisX root directory:
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501` in the default browser.

---

## Output Files

| File | Notes |
|------|-------|
| `app/streamlit_app.py` | Module 6 — full Streamlit application |
