"""
Module 1 — Data Preparation
============================
Loads raw atp_tennis.csv, removes leaky/disallowed columns, handles
missing-value sentinels, derives the binary target label, applies a
time-based train/test/holdout split, and saves three clean CSVs to data/.

Output files:
  data/processed_train.csv    — matches before 2022-01-01
  data/processed_test.csv     — matches 2022-01-01 to 2023-12-31
  data/processed_holdout.csv  — matches 2024-01-01+ (includes player names
                                 and date for the live-challenge display)

Run:
  python src/01_data_preparation.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "atp_tennis.csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("Loading raw data …")
df = pd.read_csv(RAW_CSV)
print(f"  Raw shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 2. Parse Date
# ---------------------------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# ---------------------------------------------------------------------------
# 3. Derive target label BEFORE dropping Player_1 / Winner
# ---------------------------------------------------------------------------
df["Player_1_wins"] = (df["Winner"].str.strip() == df["Player_1"].str.strip()).astype(int)
print(f"  Target distribution — Player_1 wins: {df['Player_1_wins'].sum()} / {len(df)}"
      f"  ({df['Player_1_wins'].mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# 4. Build holdout metadata (player names + date kept for display)
#    Must be captured before we drop name columns from the main frame.
# ---------------------------------------------------------------------------
holdout_meta_cols = ["Date", "Tournament", "Player_1", "Player_2", "Winner",
                     "Series", "Court", "Surface", "Round", "Best of",
                     "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Player_1_wins"]
holdout_meta = df[holdout_meta_cols].copy()

# ---------------------------------------------------------------------------
# 5. Replace missing-value sentinels (-1 / -1.0) with NaN
# ---------------------------------------------------------------------------
sentinel_cols = ["Rank_1", "Rank_2", "Pts_1", "Pts_2"]
for col in sentinel_cols:
    df[col] = df[col].replace(-1, pd.NA)
    holdout_meta[col] = holdout_meta[col].replace(-1, pd.NA)

sentinel_count = df[sentinel_cols].isna().sum()
print("\nMissing values after sentinel replacement:")
for col, n in sentinel_count.items():
    print(f"  {col}: {n} NaN rows ({n/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 6. Drop columns that are leaky, disallowed, or not useful as features
#    - Leaky:      Score, Odd_1, Odd_2, Winner
#    - Disallowed: Player_1, Player_2 (names excluded from model features)
#    - Redundant:  Tournament (replaced by Series)
# ---------------------------------------------------------------------------
drop_cols = ["Tournament", "Winner", "Score", "Odd_1", "Odd_2",
             "Player_1", "Player_2"]
df = df.drop(columns=drop_cols)
print(f"\nAfter dropping leaky/disallowed columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 7. Drop rows where EITHER rank is missing (unresolvable for prediction)
# ---------------------------------------------------------------------------
before = len(df)
df = df.dropna(subset=["Rank_1", "Rank_2"])
dropped = before - len(df)
print(f"\nDropped {dropped} rows with missing Rank_1 or Rank_2 "
      f"({dropped/before*100:.1f}% of data)")
print(f"Remaining rows: {len(df)}")

# ---------------------------------------------------------------------------
# 8. Time-based split
# ---------------------------------------------------------------------------
train_cutoff = pd.Timestamp("2022-01-01")
holdout_cutoff = pd.Timestamp("2024-01-01")

train = df[df["Date"] < train_cutoff].copy()
test  = df[(df["Date"] >= train_cutoff) & (df["Date"] < holdout_cutoff)].copy()

# Holdout: use the metadata frame (retains player names for display)
holdout_meta_clean = holdout_meta.copy()
# Apply same sentinel and rank-drop logic to metadata frame
for col in ["Rank_1", "Rank_2", "Pts_1", "Pts_2"]:
    holdout_meta_clean[col] = holdout_meta_clean[col].replace(-1, pd.NA)
holdout_meta_clean = holdout_meta_clean.dropna(subset=["Rank_1", "Rank_2"])
holdout = holdout_meta_clean[holdout_meta_clean["Date"] >= holdout_cutoff].copy()

print(f"\nSplit summary:")
print(f"  Train   (< 2022):        {len(train):>6,} rows  "
      f"({train['Date'].min().date()} → {train['Date'].max().date()})")
print(f"  Test    (2022–2023):     {len(test):>6,} rows  "
      f"({test['Date'].min().date()} → {test['Date'].max().date()})")
print(f"  Holdout (2024+):         {len(holdout):>6,} rows  "
      f"({holdout['Date'].min().date()} → {holdout['Date'].max().date()})")

# ---------------------------------------------------------------------------
# 9. Save CSVs
# ---------------------------------------------------------------------------
train_path   = os.path.join(DATA_DIR, "processed_train.csv")
test_path    = os.path.join(DATA_DIR, "processed_test.csv")
holdout_path = os.path.join(DATA_DIR, "processed_holdout.csv")

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)
holdout.to_csv(holdout_path, index=False)

print(f"\nSaved:")
print(f"  {train_path}")
print(f"  {test_path}")
print(f"  {holdout_path}")

# ---------------------------------------------------------------------------
# 10. Sanity checks
# ---------------------------------------------------------------------------
print("\n--- Sanity Checks ---")
for name, path in [("Train", train_path), ("Test", test_path), ("Holdout", holdout_path)]:
    df_check = pd.read_csv(path)
    missing_ranks = df_check[["Rank_1", "Rank_2"]].isna().sum().sum()
    print(f"  {name:10s}  rows={len(df_check):6,}  columns={len(df_check.columns)}  "
          f"missing_ranks={missing_ranks}  "
          f"target_col={'Player_1_wins' in df_check.columns}")

print("\n✓ Phase 1 complete — data preparation done.")
