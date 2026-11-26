"""
Module 4 — Model Interpretation
================================
Loads the saved Random Forest model and the engineered test set, then computes
and plots two feature-importance measures side-by-side:

  1. Built-in (impurity-based) importance  — model.feature_importances_
     Measures mean decrease in Gini impurity across all trees.
     Fast to compute; can overestimate high-cardinality or correlated features.

  2. Permutation importance                — sklearn.inspection.permutation_importance
     Measures the drop in model accuracy when each feature's values are randomly
     shuffled (breaking its relationship with the target). Computed on the test set
     so it reflects generalisation, not memorisation.

Outputs
-------
  models/feature_importance_comparison.png  – side-by-side horizontal bar chart

Run
---
  python src/04_model_interpretation.py
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe for headless runs
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TEST_CSV   = os.path.join(DATA_DIR,   "engineered_test.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
FIG_OUT    = os.path.join(MODELS_DIR, "feature_importance_comparison.png")

# ---------------------------------------------------------------------------
# 1. Load model and test set
# ---------------------------------------------------------------------------
print("Loading model …")
payload      = joblib.load(MODEL_PATH)
model        = payload["model"]
feature_cols = payload["feature_cols"]

print("Loading engineered test set …")
test   = pd.read_csv(TEST_CSV)
X_test = test[feature_cols]
y_test = test["Player_1_wins"]
print(f"  Test shape: {test.shape}")

# ---------------------------------------------------------------------------
# 2. Built-in (impurity-based) importance
# ---------------------------------------------------------------------------
print("\nComputing built-in (impurity) importance …")
builtin_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()

# ---------------------------------------------------------------------------
# 3. Permutation importance  (evaluated on test set)
# ---------------------------------------------------------------------------
print("Computing permutation importance (n_repeats=10) — this may take a minute …")
perm_result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)
perm_imp = pd.Series(perm_result.importances_mean, index=feature_cols).sort_values()

# ---------------------------------------------------------------------------
# 4. Print top-5 for each method
# ---------------------------------------------------------------------------
print("\n--- Top-5 Features: Built-in (Impurity) Importance ---")
for feat, val in builtin_imp.nlargest(5).items():
    print(f"  {feat:<35s}  {val:.4f}")

print("\n--- Top-5 Features: Permutation Importance (Test Set) ---")
for feat, val in perm_imp.nlargest(5).items():
    print(f"  {feat:<35s}  {val:.4f}")

# ---------------------------------------------------------------------------
# 5. Side-by-side horizontal bar chart
# ---------------------------------------------------------------------------
print("\nGenerating comparison plot …")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Feature Importance — Random Forest (Tennis Win Predictor)",
             fontsize=13, fontweight="bold")

# --- Left panel: built-in importance ---
ax_left = axes[0]
ax_left.barh(builtin_imp.index, builtin_imp.values, color="#3b82d4", edgecolor="white")
ax_left.set_title("Built-in Importance\n(Mean Decrease in Impurity)", fontsize=11)
ax_left.set_xlabel("Importance Score")
ax_left.tick_params(axis="y", labelsize=9)
for i, v in enumerate(builtin_imp.values):
    ax_left.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8, color="#444")

# --- Right panel: permutation importance ---
ax_right = axes[1]
colors = ["#c0392b" if v < 0 else "#3b82d4" for v in perm_imp.values]
ax_right.barh(perm_imp.index, perm_imp.values, color=colors, edgecolor="white")
ax_right.set_title("Permutation Importance\n(Mean Accuracy Drop on Test Set)", fontsize=11)
ax_right.set_xlabel("Mean Accuracy Decrease")
ax_right.tick_params(axis="y", labelsize=9)
ax_right.axvline(0, color="black", linewidth=0.8, linestyle="--")
for i, v in enumerate(perm_imp.values):
    offset = 0.001 if v >= 0 else -0.001
    ha = "left" if v >= 0 else "right"
    ax_right.text(v + offset, i, f"{v:.4f}", va="center", ha=ha, fontsize=8, color="#444")

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIG_OUT}")

print("\n✓ Phase 4 complete — feature importance comparison saved.")
