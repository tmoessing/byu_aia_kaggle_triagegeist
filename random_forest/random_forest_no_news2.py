import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# STEP 1: LOAD DATA
# ──────────────────────────────────────────────

train = pd.read_csv("data/clean_train.csv")
test = pd.read_csv("data/clean_test.csv")


# ──────────────────────────────────────────────
# STEP 2: IDENTIFY COLUMNS
# ──────────────────────────────────────────────

TARGET = "triage_acuity"

ID_COLS = ["patient_id"]

TEXT_COLS = ["chief_complaint_raw"]

# disposition and ed_los_hours are outcome columns (known only after the ED visit),
# so they are leakage and must be dropped.  They also don't exist in the test set.
LEAKAGE_COLS = ["disposition", "ed_los_hours"]

# Ablation: drop NEWS2 to measure how much the rest of the features carry alone.
EXCLUDE_FEATURES = ["news2_score"]

DROP_COLS = ID_COLS + TEXT_COLS + LEAKAGE_COLS + EXCLUDE_FEATURES

CATEGORICAL_COLS = []
for col in train.columns:
    if col in DROP_COLS or col == TARGET:
        continue
    if not pd.api.types.is_numeric_dtype(train[col]):
        CATEGORICAL_COLS.append(col)

NUMERIC_COLS = []
for col in train.columns:
    if col in DROP_COLS or col == TARGET or col in CATEGORICAL_COLS:
        continue
    NUMERIC_COLS.append(col)

print(f"Target:       {TARGET}")
print(f"Drop:         {DROP_COLS}")
print(f"Categorical:  {CATEGORICAL_COLS}")
print(f"Numeric:      {NUMERIC_COLS}")


# ──────────────────────────────────────────────
# STEP 3: CHECK CLASS IMBALANCE
# ──────────────────────────────────────────────

print("\n=== CLASS DISTRIBUTION ===")
class_counts = train[TARGET].value_counts().sort_index()
print(class_counts)
print(f"\nPercentages:")
print((class_counts / len(train) * 100).round(1))


# ──────────────────────────────────────────────
# STEP 4: ENCODE CATEGORICALS + HANDLE NaN
# ──────────────────────────────────────────────

train_encoded = pd.get_dummies(train, columns=CATEGORICAL_COLS, drop_first=True)
test_encoded = pd.get_dummies(test, columns=CATEGORICAL_COLS, drop_first=True)

train_cols_after = set(train_encoded.columns) - set(DROP_COLS) - {TARGET}
test_cols_after = set(test_encoded.columns) - set(DROP_COLS)

only_in_train = train_cols_after - test_cols_after
only_in_test = test_cols_after - train_cols_after

for col in only_in_train:
    test_encoded[col] = 0
for col in only_in_test:
    train_encoded[col] = 0

FEATURE_COLS = sorted(list(train_cols_after | test_cols_after))

X_train = train_encoded[FEATURE_COLS].copy()
y_train = train_encoded[TARGET].copy()
X_test = test_encoded[FEATURE_COLS].copy()

X_train = X_train.fillna(-1)
X_test = X_test.fillna(-1)

print(f"\nFeature matrix shape: {X_train.shape}")
print(f"Any NaN remaining?   {X_train.isnull().any().any()}")


# ──────────────────────────────────────────────
# STEP 5: BASELINE RANDOM FOREST (NO TUNING)
# ──────────────────────────────────────────────
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV


print("\n=== PHASE 1: FAST TUNING ON SUBSAMPLE ===")

SUBSAMPLE_SIZE = 80_000
TUNING_TREES = 150

if len(X_train) > SUBSAMPLE_SIZE:
    X_tune = X_train.sample(n=SUBSAMPLE_SIZE, random_state=42)
    y_tune = y_train.loc[X_tune.index]
    print(f"Tuning on {SUBSAMPLE_SIZE:,} row stratified-ish sample "
          f"(from {len(X_train):,} total).")
else:
    X_tune, y_tune = X_train, y_train
    print(f"Dataset small enough — tuning on all {len(X_train):,} rows.")

param_distributions = {
    "max_depth":         [12, 16, 20, 25],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 5, 10],
    "max_features":      ["sqrt", "log2", 0.3, 0.5],
}

cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

halving_search = HalvingRandomSearchCV(
    estimator=RandomForestClassifier(
        n_estimators=TUNING_TREES,
        random_state=42,
        n_jobs=-1,
    ),
    param_distributions=param_distributions,
    n_candidates=40,
    factor=3,
    resource="n_samples",
    min_resources=8_000,
    cv=cv_tune,
    scoring="accuracy",
    n_jobs=1,
    verbose=1,
    random_state=42,
)
halving_search.fit(X_tune, y_tune)

print(f"\nBest parameters:   {halving_search.best_params_}")
print(f"Best tuning score: {halving_search.best_score_:.4f}")


print("\n=== PHASE 2: FINAL MODEL ON FULL DATA ===")

FINAL_TREES = 600

best_params = halving_search.best_params_
best_rf = RandomForestClassifier(
    n_estimators=FINAL_TREES,
    **best_params,
    random_state=42,
    n_jobs=-1,
    oob_score=True,
)

print(f"Training {FINAL_TREES} trees on full {len(X_train):,} rows...")
best_rf.fit(X_train, y_train)

print(f"OOB accuracy:      {best_rf.oob_score_:.4f}")
print(f"Training accuracy: {best_rf.score(X_train, y_train):.4f}")

tuned_cv_scores = np.array([best_rf.oob_score_])
tuned_train_acc = best_rf.score(X_train, y_train)

# ──────────────────────────────────────────────
# STEP 8: FEATURE IMPORTANCE
# ──────────────────────────────────────────────

print("\n=== FEATURE IMPORTANCE ===")

importances = best_rf.feature_importances_
importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": importances
}).sort_values("importance", ascending=False)

print("Top 15 features:")
print(importance_df.head(15).to_string(index=False))

top_n = 20
top_features = importance_df.head(top_n)

plt.figure(figsize=(10, 8))
plt.barh(
    range(len(top_features)),
    top_features["importance"].values,
    color="#388e3c"
)
plt.yticks(range(len(top_features)), top_features["feature"].values)
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance (Mean Decrease in Impurity)")
plt.title(f"Top {top_n} Feature Importances — Random Forest (no news2_score)")
plt.tight_layout()
plt.savefig("random_forest/feature_importance_no_news2.png", dpi=150)
plt.close()
print(f"Saved: random_forest/feature_importance_no_news2.png")


# ──────────────────────────────────────────────
# STEP 9: COMPARISON WITH DECISION TREE
# ──────────────────────────────────────────────

print("\n=== RANDOM FOREST (no news2_score) vs DECISION TREE COMPARISON ===")
print(f"{'Metric':<25} {'Decision Tree':>15} {'Random Forest':>15}")
print("-" * 57)

# Quick decision tree baseline for comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
dt_cv = cross_val_score(dt, X_train, y_train, cv=cv, scoring="accuracy")
rf_cv = tuned_cv_scores

print(f"{'CV Accuracy (mean)':<25} {dt_cv.mean():>15.4f} {rf_cv.mean():>15.4f}")
print(f"{'CV Accuracy (std)':<25} {dt_cv.std():>15.4f} {rf_cv.std():>15.4f}")
print(f"{'Training Accuracy':<25} {'—':>15} {tuned_train_acc:>15.4f}")

improvement = rf_cv.mean() - dt_cv.mean()
print(f"\nRandom Forest improvement over Decision Tree: {improvement:+.4f}")


# ──────────────────────────────────────────────
# STEP 10: GENERATE KAGGLE SUBMISSION
# ──────────────────────────────────────────────

print("\n=== GENERATING SUBMISSION ===")

test_preds = best_rf.predict(X_test)

submission = pd.DataFrame({
    "patient_id": test["patient_id"],
    "acuity": test_preds,
})
submission.to_csv("random_forest/submission_no_news2.csv", index=False)
print(f"Saved: random_forest/submission_no_news2.csv")
print(f"Submission shape: {submission.shape}")
print(f"Prediction distribution:\n{pd.Series(test_preds).value_counts().sort_index()}")

print("\n✅ Done! Upload random_forest/submission_no_news2.csv to Kaggle.")
