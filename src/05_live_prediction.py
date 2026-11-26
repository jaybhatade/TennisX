"""
Module 5 — Live Prediction Challenge
======================================
Simulates a "live" prediction by running the model on the held-out 2024+ matches,
then reveals the actual results. This is the workshop's payoff moment.

Steps:
  1. Load engineered_holdout.csv (feature-engineered rows + Player_1_wins label).
  2. Load processed_holdout.csv for display metadata (Player_1, Player_2, Tournament, Date).
  3. Load models/rf_model.joblib.
  4. Take the 50 most-recent matches from the holdout as the "live challenge" sample.
  5. Run model.predict_proba to get P(Player_1 wins).
  6. Build and print the prediction vs. actual table.
  7. Compute and print overall holdout accuracy.
  8. Highlight high-confidence mispredictions (model > 70% confident but wrong).

Run:
  python src/05_live_prediction.py
"""

import os
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

ENGINEERED_HOLDOUT = os.path.join(DATA_DIR, "engineered_holdout.csv")
PROCESSED_HOLDOUT  = os.path.join(DATA_DIR, "processed_holdout.csv")
MODEL_PATH         = os.path.join(MODELS_DIR, "rf_model.joblib")

# Number of most-recent matches to use as the "live challenge" sample
CHALLENGE_SIZE = 50

# Confidence threshold above which a wrong prediction is "high-confidence"
CONFIDENCE_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print("Loading model …")
payload      = joblib.load(MODEL_PATH)
model        = payload["model"]
feature_cols = payload["feature_cols"]
print(f"  Model loaded — {len(feature_cols)} feature columns")

# ---------------------------------------------------------------------------
# 2. Load engineered holdout (features + target)
# ---------------------------------------------------------------------------
print("\nLoading engineered holdout …")
eng = pd.read_csv(ENGINEERED_HOLDOUT, parse_dates=["Date"])
print(f"  Engineered holdout shape: {eng.shape}")

# Cast any bool columns to int to avoid dtype mismatch with the trained model
for col in feature_cols:
    if col in eng.columns and eng[col].dtype == bool:
        eng[col] = eng[col].astype(int)

# ---------------------------------------------------------------------------
# 3. Load processed holdout for display metadata
# ---------------------------------------------------------------------------
print("Loading processed holdout (metadata) …")
meta = pd.read_csv(PROCESSED_HOLDOUT, parse_dates=["Date"])
print(f"  Processed holdout shape: {meta.shape}")

# ---------------------------------------------------------------------------
# 4. Merge metadata columns onto the engineered frame
#    Both CSVs share the same row order (produced by the same pipeline),
#    so a positional join on Date + index is safe.
#    We merge on a shared row-index after sorting both by Date to be sure.
# ---------------------------------------------------------------------------
# Sort both by Date consistently and reset index
eng  = eng.sort_values("Date", kind="stable").reset_index(drop=True)
meta = meta.sort_values("Date", kind="stable").reset_index(drop=True)

# Attach display columns from meta (avoid column name conflicts)
eng["Player_1"]  = meta["Player_1"]
eng["Player_2"]  = meta["Player_2"]
eng["Tournament"] = meta["Tournament"]

# Take the 50 most-recent matches (tail after sort)
challenge = eng.tail(CHALLENGE_SIZE).copy()
print(f"\nChallenge sample: {len(challenge)} most-recent matches")
print(f"  Date range: {challenge['Date'].min().date()} → {challenge['Date'].max().date()}")

# ---------------------------------------------------------------------------
# 5. Run predict_proba on the challenge set
# ---------------------------------------------------------------------------
X_challenge = challenge[feature_cols]
proba = model.predict_proba(X_challenge)   # shape (n, 2): [P(0), P(1)]
p_p1_wins = proba[:, 1]                    # P(Player_1 wins)

predicted_winner_idx = (p_p1_wins >= 0.5).astype(int)   # 1 = Player_1, 0 = Player_2
actual_winner_idx    = challenge["Player_1_wins"].values

correct = (predicted_winner_idx == actual_winner_idx)

# ---------------------------------------------------------------------------
# 6. Build results table
# ---------------------------------------------------------------------------
results = pd.DataFrame({
    "Date":             challenge["Date"].dt.date.values,
    "Tournament":       challenge["Tournament"].values,
    "Player_1":         challenge["Player_1"].values,
    "Player_2":         challenge["Player_2"].values,
    "P(P1 wins)":       np.round(p_p1_wins, 3),
    "Predicted Winner": [
        row["Player_1"] if pred == 1 else row["Player_2"]
        for pred, (_, row) in zip(predicted_winner_idx, challenge.iterrows())
    ],
    "Actual Winner": [
        row["Player_1"] if actual == 1 else row["Player_2"]
        for actual, (_, row) in zip(actual_winner_idx, challenge.iterrows())
    ],
    "Correct?": ["✓" if c else "✗" for c in correct],
})

# ---------------------------------------------------------------------------
# 7. Print the table
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("  LIVE PREDICTION CHALLENGE — 50 Most-Recent Holdout Matches (2024+)")
print("=" * 100)
print(results.to_string(index=True))
print("=" * 100)

# ---------------------------------------------------------------------------
# 8. Overall holdout accuracy
# ---------------------------------------------------------------------------
holdout_acc = correct.mean()
n_correct   = correct.sum()
print(f"\n  Holdout accuracy : {n_correct}/{CHALLENGE_SIZE}  ({holdout_acc*100:.1f}%)")
print(f"  (Test-set accuracy for reference: 60.95%)")

# ---------------------------------------------------------------------------
# 9. High-confidence mispredictions
# ---------------------------------------------------------------------------
high_conf_wrong = results[
    (~correct) & (np.maximum(p_p1_wins, 1 - p_p1_wins) > CONFIDENCE_THRESHOLD)
]

print(f"\n--- High-Confidence Mispredictions (confidence > {CONFIDENCE_THRESHOLD*100:.0f}%) ---")
if len(high_conf_wrong) == 0:
    print("  None — no high-confidence wrong predictions in this sample.")
else:
    print(f"  {len(high_conf_wrong)} match(es) where model was >70% confident but wrong:\n")
    print(high_conf_wrong[["Date", "Player_1", "Player_2",
                             "P(P1 wins)", "Predicted Winner", "Actual Winner"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 10. Discussion prompt
# ---------------------------------------------------------------------------
print("""
--- Workshop Discussion Prompt ---
  • The model achieved {acc:.0f}% on the 50-match live challenge.
  • High-confidence wrong predictions are especially interesting: they reveal
    matches where ranking/points suggested a clear favourite — yet the lower-ranked
    player won. What factors might explain this? (Surface? Momentum? Injury?)
  • The model only uses pre-match features (ranking difference, surface, round).
    Real upsets often involve form, fitness, or head-to-head history — information
    this model does not have.
  • How does {acc:.0f}% compare to betting-market accuracy (~68%) and expert pundits (~65%)?
    The gap highlights how much signal the excluded betting odds contain.
""".format(acc=holdout_acc * 100))

print("✓ Phase 5 complete — live prediction challenge done.")
