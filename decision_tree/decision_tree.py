import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
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

DROP_COLS = ID_COLS + TEXT_COLS + LEAKAGE_COLS

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

plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar", color=["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1976d2"])
plt.title("Class Distribution: Acuity Levels")
plt.xlabel("Acuity Level")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("decision_tree/class_distribution.png", dpi=150)
plt.close()
print("Saved: decision_tree/class_distribution.png")


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
# STEP 5: LEAKAGE CHECK
# ──────────────────────────────────────────────

print("\n=== LEAKAGE CHECK ===")
print("Features that perfectly separate the target are suspicious.")
print("Checking individual feature accuracy with a depth-1 tree...\n")

leakage_results = []
for col in FEATURE_COLS:
    mini_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
    mini_tree.fit(X_train[[col]], y_train)
    acc = mini_tree.score(X_train[[col]], y_train)
    leakage_results.append((col, acc))

leakage_results.sort(key=lambda x: x[1], reverse=True)
print("Top 10 single-feature accuracies:")
for name, acc in leakage_results[:10]:
    flag = " ⚠️  SUSPICIOUS" if acc > 0.9 else ""
    print(f"  {name:40s} {acc:.4f}{flag}")

print("\nIf any feature has >90% accuracy alone, investigate whether")
print("it was derived FROM the target (that would be leakage).")


# ──────────────────────────────────────────────
# STEP 6: BASELINE DECISION TREE (NO TUNING)
# ──────────────────────────────────────────────

print("\n=== BASELINE DECISION TREE ===")
baseline_tree = DecisionTreeClassifier(random_state=42)
baseline_tree.fit(X_train, y_train)

baseline_train_acc = baseline_tree.score(X_train, y_train)
print(f"Training accuracy: {baseline_train_acc:.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_cv_scores = cross_val_score(baseline_tree, X_train, y_train, cv=cv, scoring="accuracy")
print(f"CV accuracy:       {baseline_cv_scores.mean():.4f} ± {baseline_cv_scores.std():.4f}")
print(f"CV fold scores:    {[round(s, 4) for s in baseline_cv_scores]}")

if baseline_train_acc > baseline_cv_scores.mean() + 0.10:
    print("\n⚠️  Large gap between train and CV → the tree is overfitting.")
    print("   This is expected for an unrestricted tree. We'll fix it with tuning.\n")


# ──────────────────────────────────────────────
# STEP 7: HYPERPARAMETER TUNING
# ──────────────────────────────────────────────

print("=== HYPERPARAMETER TUNING (GridSearchCV) ===")
print("This may take a few minutes...\n")

param_grid = {
    "max_depth":        [3, 5, 7, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf":  [1, 2, 5, 10, 20],
    "criterion":         ["gini", "entropy"],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

best_tree = grid_search.best_estimator_


# ──────────────────────────────────────────────
# STEP 8: EVALUATE THE TUNED TREE
# ──────────────────────────────────────────────

print("\n=== TUNED TREE EVALUATION ===")

tuned_train_acc = best_tree.score(X_train, y_train)
tuned_cv_scores = cross_val_score(best_tree, X_train, y_train, cv=cv, scoring="accuracy")

print(f"Training accuracy: {tuned_train_acc:.4f}")
print(f"CV accuracy:       {tuned_cv_scores.mean():.4f} ± {tuned_cv_scores.std():.4f}")

y_train_pred = best_tree.predict(X_train)
print(f"\nClassification Report (on training data):")
print(classification_report(y_train, y_train_pred, digits=3))

cm = confusion_matrix(y_train, y_train_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_train.unique()))
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix — Tuned Decision Tree (Train)")
plt.tight_layout()
plt.savefig("decision_tree/confusion_matrix.png", dpi=150)
plt.close()
print("Saved: decision_tree/confusion_matrix.png")


# ──────────────────────────────────────────────
# STEP 9: FEATURE IMPORTANCE
# ──────────────────────────────────────────────

print("\n=== FEATURE IMPORTANCE ===")

importances = best_tree.feature_importances_
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
    color="#1976d2"
)
plt.yticks(range(len(top_features)), top_features["feature"].values)
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance (Gini / Entropy)")
plt.title(f"Top {top_n} Feature Importances — Decision Tree")
plt.tight_layout()
plt.savefig("decision_tree/feature_importance.png", dpi=150)
plt.close()
print(f"Saved: decision_tree/feature_importance.png")


# ──────────────────────────────────────────────
# STEP 10: VISUALIZE THE TREE (TOP LEVELS)
# ──────────────────────────────────────────────

print("\n=== TREE VISUALIZATION (top 4 levels) ===")

plt.figure(figsize=(24, 12))
plot_tree(
    best_tree,
    feature_names=FEATURE_COLS,
    class_names=[str(c) for c in sorted(y_train.unique())],
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=8,
)
plt.title("Decision Tree — Top 4 Levels")
plt.tight_layout()
plt.savefig("decision_tree/tree_visualization.png", dpi=150)
plt.close()
print("Saved: decision_tree/tree_visualization.png")

text_tree = export_text(best_tree, feature_names=FEATURE_COLS, max_depth=4)
with open("decision_tree/tree_rules.txt", "w") as f:
    f.write(text_tree)
print("Saved: decision_tree/tree_rules.txt")


# ──────────────────────────────────────────────
# STEP 11: GENERATE KAGGLE SUBMISSION
# ──────────────────────────────────────────────

print("\n=== GENERATING SUBMISSION ===")

test_preds = best_tree.predict(X_test)

submission = pd.DataFrame({
    "patient_id": test["patient_id"],
    "acuity": test_preds,
})
submission.to_csv("decision_tree/submission.csv", index=False)
print(f"Saved: decision_tree/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Prediction distribution:\n{pd.Series(test_preds).value_counts().sort_index()}")

print("\n✅ Done! Upload decision_tree/submission.csv to Kaggle.")
