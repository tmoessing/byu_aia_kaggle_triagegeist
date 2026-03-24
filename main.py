import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("data/clean_train.csv")

# Drop non-predictive columns and target leakage
drop_cols = ["patient_id", "site_id", "triage_nurse_id", "disposition", "ed_los_hours",
             "chief_complaint_raw", "triage_acuity"]
X = data.drop(columns=drop_cols)
y = data["triage_acuity"]

# Encode categorical columns
cat_cols = X.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_val)
print(f"Accuracy: {accuracy_score(y_val, preds):.4f}\n")
print(classification_report(y_val, preds))

# Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\n--- Feature Importance (top 20) ---")
print(importance.head(20).to_string())
