"""
Module 2 — Feature Engineering
================================
Loads the three processed CSVs from data/, engineers numeric features,
and writes three engineered CSVs:

  data/engineered_train.csv
  data/engineered_test.csv
  data/engineered_holdout.csv

Feature engineering steps:
  1. Impute missing Pts_1 / Pts_2 with training-set medians.
  2. Create difference features: rank_diff, pts_diff.
  3. Ordinal-encode Round.
  4. One-hot encode Surface, Court, Series (fit on train; align test/holdout).
  5. Keep Best of as-is.
  6. Drop Date, Rank_1, Rank_2, Pts_1, Pts_2 after engineering.

The holdout output keeps Player_1, Player_2, Tournament, Date columns (for
Module 5 display) alongside the engineered feature columns.

Run:
  python src/02_feature_engineering.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_IN   = os.path.join(DATA_DIR, "processed_train.csv")
TEST_IN    = os.path.join(DATA_DIR, "processed_test.csv")
HOLDOUT_IN = os.path.join(DATA_DIR, "processed_holdout.csv")

TRAIN_OUT   = os.path.join(DATA_DIR, "engineered_train.csv")
TEST_OUT    = os.path.join(DATA_DIR, "engineered_test.csv")
HOLDOUT_OUT = os.path.join(DATA_DIR, "engineered_holdout.csv")

# ---------------------------------------------------------------------------
# Ordinal mapping for Round
# Round Robin is an ATP Finals format; slot it between Quarterfinals and
# Semifinals (it precedes the knockout stage).
# ---------------------------------------------------------------------------
ROUND_ORDINAL = {
    "1st Round":     1,
    "2nd Round":     2,
    "3rd Round":     3,
    "4th Round":     4,
    "Quarterfinals": 5,
    "Round Robin":   5,  # treat as equivalent importance to Quarterfinal stage
    "Semifinals":    6,
    "The Final":     7,
}

# Categorical columns to one-hot encode (fit on train, align test/holdout)
OHE_COLS = ["Surface", "Court", "Series"]

# Columns to drop after engineering (raw numerics replaced by diffs)
DROP_AFTER = ["Date", "Rank_1", "Rank_2", "Pts_1", "Pts_2"]

# Extra display columns present only in holdout (kept in output for Module 5)
HOLDOUT_DISPLAY_COLS = ["Date", "Tournament", "Player_1", "Player_2"]

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("Loading processed CSVs …")
train   = pd.read_csv(TRAIN_IN,   parse_dates=["Date"])
test    = pd.read_csv(TEST_IN,    parse_dates=["Date"])
holdout = pd.read_csv(HOLDOUT_IN, parse_dates=["Date"])

print(f"  Train   shape: {train.shape}")
print(f"  Test    shape: {test.shape}")
print(f"  Holdout shape: {holdout.shape}")

# ---------------------------------------------------------------------------
# 2. Impute missing Pts_1 / Pts_2 using TRAINING medians only
#    (avoids data leakage into test/holdout)
# ---------------------------------------------------------------------------
print("\nImputing missing Pts values with training medians …")
pts1_median = train["Pts_1"].median()
pts2_median = train["Pts_2"].median()
print(f"  Pts_1 median (train): {pts1_median:.0f}")
print(f"  Pts_2 median (train): {pts2_median:.0f}")

for df in [train, test, holdout]:
    df["Pts_1"] = df["Pts_1"].fillna(pts1_median)
    df["Pts_2"] = df["Pts_2"].fillna(pts2_median)

# Confirm no missing values remain in rank/pts columns
for name, df in [("Train", train), ("Test", test), ("Holdout", holdout)]:
    missing = df[["Rank_1", "Rank_2", "Pts_1", "Pts_2"]].isna().sum().sum()
    print(f"  {name:10s} remaining NaN in rank/pts: {missing}")

# ---------------------------------------------------------------------------
# 3. Difference features
# ---------------------------------------------------------------------------
print("\nCreating difference features …")
for df in [train, test, holdout]:
    df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
    df["pts_diff"]  = df["Pts_1"]  - df["Pts_2"]

# ---------------------------------------------------------------------------
# 4. Ordinal encoding for Round
# ---------------------------------------------------------------------------
print("Ordinal-encoding Round …")
unknown_rounds = set()
for df in [train, test, holdout]:
    df["Round_ord"] = df["Round"].map(ROUND_ORDINAL)
    unknown = df[df["Round_ord"].isna()]["Round"].unique().tolist()
    unknown_rounds.update(unknown)

if unknown_rounds:
    print(f"  WARNING: unmapped Round values → {unknown_rounds} (will be NaN)")
else:
    print("  All Round values mapped successfully.")

# ---------------------------------------------------------------------------
# 5. One-hot encode Surface, Court, Series
#    Fit get_dummies on TRAIN only; align test/holdout to the same columns.
# ---------------------------------------------------------------------------
print("One-hot encoding Surface, Court, Series …")

# Encode train
train_encoded = pd.get_dummies(train, columns=OHE_COLS, drop_first=True)

# Identify the OHE columns that were created from the training set
ohe_feature_cols = [c for c in train_encoded.columns
                    if any(c.startswith(prefix + "_") for prefix in OHE_COLS)]
print(f"  OHE columns created from train: {ohe_feature_cols}")

# Encode test/holdout and align to train's column schema
def encode_and_align(df: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    """One-hot encode and reindex to match training columns exactly."""
    encoded = pd.get_dummies(df, columns=OHE_COLS, drop_first=True)
    # Reindex to match train; missing OHE cols become 0, extra cols are dropped
    encoded = encoded.reindex(columns=train_columns, fill_value=0)
    return encoded

test_encoded    = encode_and_align(test,    train_encoded.columns.tolist())
holdout_encoded = encode_and_align(holdout, train_encoded.columns.tolist())

# ---------------------------------------------------------------------------
# 6. Drop raw columns no longer needed
#    (Date is kept in holdout via the display-columns path)
# ---------------------------------------------------------------------------
print("Dropping raw numeric and date columns …")

def drop_raw(df: pd.DataFrame, drop_cols: list, keep_cols: list = None) -> pd.DataFrame:
    """Drop raw columns except those in keep_cols."""
    to_drop = [c for c in drop_cols if c in df.columns
               and (keep_cols is None or c not in keep_cols)]
    return df.drop(columns=to_drop)

train_final   = drop_raw(train_encoded,   DROP_AFTER)
test_final    = drop_raw(test_encoded,    DROP_AFTER)
holdout_final = drop_raw(holdout_encoded, DROP_AFTER, keep_cols=HOLDOUT_DISPLAY_COLS)

# Drop Round (raw string) — now replaced by Round_ord — and Winner column
# in holdout (not in train/test, was kept for metadata only)
for col in ["Round", "Winner"]:
    for name_df in [(train_final, "train"), (test_final, "test"), (holdout_final, "holdout")]:
        df, _lbl = name_df
        if col in df.columns:
            # Assign back to drop without inplace (Pandas 4.x compat)
            if _lbl == "train":
                train_final = train_final.drop(columns=[col])
            elif _lbl == "test":
                test_final = test_final.drop(columns=[col])
            else:
                holdout_final = holdout_final.drop(columns=[col])

print(f"\n  Train   final columns: {list(train_final.columns)}")
print(f"  Test    final columns: {list(test_final.columns)}")

# ---------------------------------------------------------------------------
# 7. Save engineered CSVs
# ---------------------------------------------------------------------------
print("\nSaving engineered CSVs …")
train_final.to_csv(TRAIN_OUT,   index=False)
test_final.to_csv(TEST_OUT,    index=False)
holdout_final.to_csv(HOLDOUT_OUT, index=False)
print(f"  {TRAIN_OUT}")
print(f"  {TEST_OUT}")
print(f"  {HOLDOUT_OUT}")

# ---------------------------------------------------------------------------
# 8. Sanity checks
# ---------------------------------------------------------------------------
print("\n--- Sanity Checks ---")
feature_cols = [c for c in train_final.columns if c != "Player_1_wins"]
print(f"\nFeature columns ({len(feature_cols)}):")
for c in feature_cols:
    print(f"  {c}")

for name, path in [("Train", TRAIN_OUT), ("Test", TEST_OUT), ("Holdout", HOLDOUT_OUT)]:
    df_check = pd.read_csv(path)
    missing  = df_check[feature_cols].isna().sum().sum()
    has_target = "Player_1_wins" in df_check.columns
    print(f"\n  {name:10s}  rows={len(df_check):6,}  cols={len(df_check.columns)}"
          f"  feature_NaN={missing}  target_col={has_target}")

print("\n✓ Phase 2 complete — feature engineering done.")
