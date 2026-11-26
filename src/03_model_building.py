"""
Module 3 — Model Building
==========================
Loads the engineered train and test CSVs, trains a Random Forest classifier,
evaluates it on the test set, and saves the fitted model together with its
feature column list to models/rf_model.joblib.

Outputs
-------
  models/rf_model.joblib   – dict: {"model": <RFC>, "feature_cols": [...]}

Run
---
  python src/03_model_building.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "engineered_train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "engineered_test.csv")
MODEL_OUT = os.path.join(MODELS_DIR, "rf_model.joblib")

# ---------------------------------------------------------------------------
# 1. Load engineered data
# ---------------------------------------------------------------------------
print("Loading engineered CSVs …")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
print(f"  Train  shape: {train.shape}")
print(f"  Test   shape: {test.shape}")

# ---------------------------------------------------------------------------
# 2. Separate features and target
#    Exclude any residual metadata columns that may have survived into these
#    files (they should not exist in train/test, but guard defensively).
# ---------------------------------------------------------------------------
METADATA_COLS = {"Player_1_wins", "Date", "Tournament", "Player_1", "Player_2", "Winner"}

feature_cols = [c for c in train.columns if c not in METADATA_COLS]

X_train = train[feature_cols]
y_train = train["Player_1_wins"]

X_test  = test[feature_cols]
y_test  = test["Player_1_wins"]

print(f"\nFeature columns ({len(feature_cols)}):")
for col in feature_cols:
    print(f"  {col}")

print(f"\nTrain samples : {len(X_train):,}  (target balance: {y_train.mean():.3f})")
print(f"Test  samples : {len(X_test):,}  (target balance: {y_test.mean():.3f})")

# ---------------------------------------------------------------------------
# 3. Train Random Forest
# ---------------------------------------------------------------------------
print("\nTraining RandomForestClassifier(n_estimators=200, random_state=42) …")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("  Training complete.")

# ---------------------------------------------------------------------------
# 4. Evaluate on test set
# ---------------------------------------------------------------------------
print("\n--- Test-Set Evaluation ---")

y_pred      = clf.predict(X_test)
y_proba     = clf.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
ll  = log_loss(y_test, y_proba)

print(f"\n  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Log-loss : {ll:.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["Player_2 wins (0)", "Player_1 wins (1)"]))

# Top-5 features by built-in impurity importance
importances = pd.Series(clf.feature_importances_, index=feature_cols)
top5 = importances.nlargest(5)
print("  Top-5 features (impurity importance):")
for feat, imp in top5.items():
    print(f"    {feat:<30s}  {imp:.4f}")

# ---------------------------------------------------------------------------
# 5. Save model + feature column list
# ---------------------------------------------------------------------------
print(f"\nSaving model to {MODEL_OUT} …")
payload = {
    "model":        clf,
    "feature_cols": feature_cols,
}
joblib.dump(payload, MODEL_OUT)
print(f"  Saved — file size: {os.path.getsize(MODEL_OUT) / 1024 / 1024:.1f} MB")

print("\n✓ Phase 3 complete — model trained and saved.")
