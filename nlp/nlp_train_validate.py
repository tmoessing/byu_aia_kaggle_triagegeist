import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

print("Loading Data and Labels...")
base_dir = Path(__file__).resolve().parent
runs_dir = base_dir / "model_runs"
registry_path = runs_dir / "model_registry.json"
TEST_SIZE = 0.20
CV_FOLDS = 3
REVEAL_TEST_METRICS = False
USE_STRUCTURED_FEATURES = True

STRUCTURED_EXCLUDE_COLUMNS = {
    "triage_acuity",
    "chief_complaint_raw",
    "chief_complaint_system",
    "disposition",
    "ed_los_hours",
}

def load_registry(path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"latest_run": None, "runs": []}

def next_run_id(registry):
    latest = registry.get("latest_run")
    if latest and latest.startswith("run_"):
        try:
            return f"run_{int(latest.split('_')[1]) + 1:02d}"
        except (IndexError, ValueError):
            pass
    return f"run_{len(registry.get('runs', [])) + 1:02d}"

def metric_from_report(report, metric_name):
    if metric_name == "accuracy":
        return float(report["accuracy"])
    return float(report["weighted avg"][metric_name])

def build_comparison_plot(metric_rows, output_path, title_suffix):
    sns.set_theme(style="whitegrid", context="talk")
    metrics = ["accuracy", "precision", "recall", "f1-score"]
    metric_titles = {
        "accuracy": "Accuracy",
        "precision": "Precision (Weighted)",
        "recall": "Recall (Weighted)",
        "f1-score": "F1-Score (Weighted)",
    }

    pipeline_labels = {
        "pipeline_a": "A",
        "pipeline_b": "B",
        "pipeline_a_2_phrase": "A-2P",
        "pipeline_b_2_phrase": "B-2P",
    }

    # Distinct colors for models
    model_colors = {
        "RF": "#1f77b4",
        "XGB": "#ff7f0e",
        "LR": "#2ca02c",
        "MLP": "#d62728",
    }

    labels = []
    bar_colors = []
    for row in metric_rows:
        pipeline_key = row["pipeline"]
        model_type = row["model_type"].upper()
        labels.append(f"{pipeline_labels.get(pipeline_key, pipeline_key)}-{model_type}")
        bar_colors.append(model_colors.get(model_type, "#4c4c4c"))

    all_scores = [row[metric] for row in metric_rows for metric in metrics]
    min_score = min(all_scores) if all_scores else 0.0
    y_min = max(0, np.floor(min_score * 10) / 10 - 0.1)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [row[metric] for row in metric_rows]
        axes[idx].bar(labels, values, color=bar_colors, edgecolor="#222222", linewidth=0.5)
        axes[idx].set_title(metric_titles[metric])
        axes[idx].set_ylim(y_min, 1.0)
        axes[idx].set_ylabel("Score")
        axes[idx].tick_params(axis="x", rotation=30, labelsize=10)
        for x_pos, value in enumerate(values):
            axes[idx].text(x_pos, min(value + 0.01, 0.99), f"{value:.3f}", ha="center", fontsize=8)

    fig.suptitle(f"Model Comparison {title_suffix} (CV Mean Scores)", fontsize=18, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def cross_validate_model(df, pipeline_name, model_name, model_factory, scale_features=False):
    print(f"\nTRAINING: {pipeline_name} | {model_name} (Cross-Validation)")
    
    # Pre-split test set
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=42, stratify=df["triage_acuity"]
    )
    
    X = train_val_df.drop(columns=["patient_id", "triage_acuity"])
    y = train_val_df["triage_acuity"]
    X_test_final = test_df.drop(columns=["patient_id", "triage_acuity"])
    y_test_final = test_df["triage_acuity"]

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    
    # XGBoost and others work best with 0..N-1 labels
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), index=y.index)

    fold_reports = []
    all_y_true = []
    all_y_pred = []
    best_model = None
    best_f1 = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

        if scale_features:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

        model = model_factory()
        model.fit(X_train, y_train)
        
        preds_enc = np.asarray(model.predict(X_val), dtype=int)
        preds = le.inverse_transform(preds_enc)
        y_val_orig = le.inverse_transform(y_val)

        report = classification_report(y_val_orig, preds, output_dict=True, zero_division=0)
        fold_reports.append(report)
        all_y_true.extend(y_val_orig)
        all_y_pred.extend(preds)

        val_f1 = report["weighted avg"]["f1-score"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model

    # Aggregated metrics
    avg_metrics = {
        "accuracy": np.mean([r["accuracy"] for r in fold_reports]),
        "precision": np.mean([r["weighted avg"]["precision"] for r in fold_reports]),
        "recall": np.mean([r["weighted avg"]["recall"] for r in fold_reports]),
        "f1-score": np.mean([r["weighted avg"]["f1-score"] for r in fold_reports]),
    }

    print(f"CV Mean F1: {avg_metrics['f1-score']:.4f}")

    return {
        "model": best_model,
        "avg_metrics": avg_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "feature_names": list(X.columns),
    }

# --- DATA PREP ---
df_train_master = pd.read_csv(base_dir / "clean_train.csv")
df_labels = df_train_master[["patient_id", "triage_acuity"]]

if USE_STRUCTURED_FEATURES:
    structured_columns = [col for col in df_train_master.columns if col not in STRUCTURED_EXCLUDE_COLUMNS]
    df_structured = df_train_master[structured_columns].copy()
    if "patient_id" not in df_structured.columns: df_structured.insert(0, "patient_id", df_train_master["patient_id"])
    numeric_cols = df_structured.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in df_structured.columns if col not in numeric_cols and col != "patient_id"]
    df_structured[numeric_cols] = df_structured[numeric_cols].fillna(0)
    if categorical_cols:
        df_structured[categorical_cols] = df_structured[categorical_cols].fillna("missing")
        df_structured = pd.get_dummies(df_structured, columns=categorical_cols)
else:
    df_structured = df_labels[["patient_id"]].copy()

feature_files = {
    "pipeline_a": ("1-Phrase Full", "features_pipeline_A_full.csv"),
    "pipeline_b": ("1-Phrase Objective", "features_pipeline_B_objective.csv"),
    "pipeline_a_2_phrase": ("2-Phrase Full", "features_pipeline_A_full_2_phrase.csv"),
    "pipeline_b_2_phrase": ("2-Phrase Objective", "features_pipeline_B_objective_2_phrase.csv"),
}

experiments = []
for pk, (pn, fn) in feature_files.items():
    fdf = pd.read_csv(base_dir / fn)
    merged = pd.merge(df_structured, fdf, on="patient_id", how="inner")
    merged = pd.merge(merged, df_labels, on="patient_id", how="inner")
    experiments.append((pk, pn, merged))

# --- RUN CONFIG ---
runs_dir.mkdir(parents=True, exist_ok=True)
registry = load_registry(registry_path)
run_id = next_run_id(registry)
run_dir = runs_dir / run_id
run_dir.mkdir(parents=True, exist_ok=True)

run_record = {
    "run_id": run_id,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "cv_folds": CV_FOLDS,
    "models": [],
}

comparison_rows = []
best_by_pipeline = {}

# --- MODEL LOOP ---
for pk, pn, data in experiments:
    models_to_run = [
        ("RF", lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
        ("LR", lambda: LogisticRegression(max_iter=1000, random_state=42), True),
        ("MLP", lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42), True),
    ]
    if HAS_XGBOOST:
        models_to_run.append(("XGB", lambda: XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1), False))

    for m_short, m_factory, scale in models_to_run:
        res = cross_validate_model(data, pn, m_short, m_factory, scale_features=scale)
        
        comparison_rows.append({
            "pipeline": pk,
            "model_type": m_short.lower(),
            "accuracy": res["avg_metrics"]["accuracy"],
            "precision": res["avg_metrics"]["precision"],
            "recall": res["avg_metrics"]["recall"],
            "f1-score": res["avg_metrics"]["f1-score"],
        })

        if pk not in best_by_pipeline or res["avg_metrics"]["f1-score"] > best_by_pipeline[pk]["val_f1"]:
            best_by_pipeline[pk] = {
                "val_f1": res["avg_metrics"]["f1-score"],
                "model_name": m_short,
                "y_true": res["y_true"],
                "y_pred": res["y_pred"],
            }

# --- PLOTTING & SAVE ---
rows_1p = [r for r in comparison_rows if "2_phrase" not in r["pipeline"]]
rows_2p = [r for r in comparison_rows if "2_phrase" in r["pipeline"]]

img_1p = run_dir / "comparison_1_phrase.png"
img_2p = run_dir / "comparison_2_phrase.png"

build_comparison_plot(rows_1p, img_1p, "(1-Phrase)")
build_comparison_plot(rows_2p, img_2p, "(2-Phrase)")

registry["latest_run"] = run_id
registry.setdefault("runs", []).append(run_record)
with registry_path.open("w") as f: json.dump(registry, f, indent=2)

print(f"\nRun Complete: {run_id}. Plots saved in {run_dir}")
