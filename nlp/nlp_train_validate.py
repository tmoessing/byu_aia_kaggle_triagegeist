import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
VAL_SIZE_WITHIN_REMAINING = 0.20  # 20% of 80% gives 16% overall validation
REVEAL_TEST_METRICS = False  # Keep False to preserve the final test-set vault
USE_STRUCTURED_FEATURES = True

# Columns excluded from structured features to avoid target leakage or duplicate text pathways.
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


def top_feature_weights(model, feature_names, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return []
    weights = list(model.feature_importances_)
    ranked = sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True)
    return [
        {"feature": feature, "weight": float(weight)}
        for feature, weight in ranked[:top_n]
    ]


def metric_from_report(report, metric_name):
    if metric_name == "accuracy":
        return float(report["accuracy"])
    return float(report["weighted avg"][metric_name])


def build_comparison_plot(metric_rows, output_path):
    sns.set_theme(style="whitegrid", context="talk")

    metrics = ["accuracy", "precision", "recall", "f1-score"]
    metric_titles = {
        "accuracy": "Validation Accuracy",
        "precision": "Validation Precision (Weighted)",
        "recall": "Validation Recall (Weighted)",
        "f1-score": "Validation F1-Score (Weighted)",
    }

    pipeline_colors = {
        "pipeline_a": "#1f77b4",
        "pipeline_b": "#ff7f0e",
        "pipeline_a_2_phrase": "#2ca02c",
        "pipeline_b_2_phrase": "#d62728",
    }
    pipeline_labels = {
        "pipeline_a": "A",
        "pipeline_b": "B",
        "pipeline_a_2_phrase": "A-2P",
        "pipeline_b_2_phrase": "B-2P",
    }

    labels = []
    bar_colors = []
    for row in metric_rows:
        pipeline_key = row["pipeline"]
        model_short = "RF" if row["model_type"] == "rf" else "XGB"
        labels.append(f"{pipeline_labels.get(pipeline_key, pipeline_key)}-{model_short}")
        bar_colors.append(pipeline_colors.get(pipeline_key, "#4c4c4c"))

    all_scores = [row[metric] for row in metric_rows for metric in metrics]
    min_score = min(all_scores) if all_scores else 0.0
    if min_score >= 0.8:
        y_min = 0.8
    elif min_score >= 0.6:
        y_min = 0.6
    else:
        y_min = 0.0

    y_ticks = np.round(np.arange(y_min, 1.01, 0.05), 2)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [row[metric] for row in metric_rows]
        axes[idx].bar(labels, values, color=bar_colors, edgecolor="#222222", linewidth=0.5)
        axes[idx].set_title(metric_titles[metric])
        axes[idx].set_ylim(y_min, 1.0)
        axes[idx].set_yticks(y_ticks)
        axes[idx].set_ylabel("Score")
        axes[idx].tick_params(axis="x", rotation=30, labelsize=11)
        axes[idx].tick_params(axis="y", labelsize=11)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.35)
        for x_pos, value in enumerate(values):
            axes[idx].text(x_pos, min(value + 0.02, 0.99), f"{value:.3f}", ha="center", fontsize=8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in pipeline_colors.values()
    ]
    fig.legend(
        legend_handles,
        ["Pipeline A", "Pipeline B", "Pipeline A 2-Phrase", "Pipeline B 2-Phrase"],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Model Comparison Across Pipelines (Validation Metrics)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_confusion_matrix_grid(best_by_pipeline, labels, output_path):
    sns.set_theme(style="whitegrid", context="talk")
    pipeline_order = ["pipeline_a", "pipeline_b", "pipeline_a_2_phrase", "pipeline_b_2_phrase"]
    pipeline_titles = {
        "pipeline_a": "Pipeline A",
        "pipeline_b": "Pipeline B",
        "pipeline_a_2_phrase": "Pipeline A 2-Phrase",
        "pipeline_b_2_phrase": "Pipeline B 2-Phrase",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    axes = axes.flatten()

    for idx, pipeline_key in enumerate(pipeline_order):
        ax = axes[idx]
        best_entry = best_by_pipeline.get(pipeline_key)
        if best_entry is None:
            ax.axis("off")
            ax.set_title(f"{pipeline_titles[pipeline_key]}\nNo model data")
            continue

        cm = confusion_matrix(best_entry["y_true"], best_entry["y_pred"], labels=labels)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(
            f"{pipeline_titles[pipeline_key]} | Best: {best_entry['model_name']}\n"
            f"Val weighted F1={best_entry['val_f1']:.3f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Validation Confusion Matrices (Best Model Per Pipeline)", fontsize=17, fontweight="bold")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_split_patient_ids(labels_df):
    temp_df, test_df = train_test_split(
        labels_df,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=labels_df["triage_acuity"],
    )

    train_df, val_df = train_test_split(
        temp_df,
        test_size=VAL_SIZE_WITHIN_REMAINING,
        random_state=42,
        stratify=temp_df["triage_acuity"],
    )

    return {
        "train_ids": set(train_df["patient_id"]),
        "val_ids": set(val_df["patient_id"]),
        "test_ids": set(test_df["patient_id"]),
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "total": int(len(labels_df)),
        },
    }


def subset_by_ids(df, id_set):
    subset = df[df["patient_id"].isin(id_set)].copy()
    return subset.sort_values("patient_id").reset_index(drop=True)


def train_model(df, split_ids, pipeline_name, model_name, model):
    print(f"\n{'=' * 50}")
    print(f"TRAINING MODEL: {pipeline_name} | {model_name}")
    print(f"{'=' * 50}")

    train_df = subset_by_ids(df, split_ids["train_ids"])
    val_df = subset_by_ids(df, split_ids["val_ids"])
    test_df = subset_by_ids(df, split_ids["test_ids"])

    X_train = train_df.drop(columns=["patient_id", "triage_acuity"])
    y_train = train_df["triage_acuity"]
    X_val = val_df.drop(columns=["patient_id", "triage_acuity"])
    y_val = val_df["triage_acuity"]
    X_test = test_df.drop(columns=["patient_id", "triage_acuity"])
    y_test = test_df["triage_acuity"]

    if model_name == "XGBoost":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)

        model.fit(X_train, y_train_encoded)

        train_preds = label_encoder.inverse_transform(np.asarray(model.predict(X_train), dtype=int))
        val_preds = label_encoder.inverse_transform(np.asarray(model.predict(X_val), dtype=int))
        test_preds = label_encoder.inverse_transform(np.asarray(model.predict(X_test), dtype=int))
    else:
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

    train_report = classification_report(y_train, train_preds, output_dict=True, zero_division=0)
    val_report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    test_report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)

    print("\n--- TRAINING SCORES ---")
    print(classification_report(y_train, train_preds, zero_division=0))

    print("\n--- VALIDATION SCORES ---")
    print(classification_report(y_val, val_preds, zero_division=0))

    if REVEAL_TEST_METRICS:
        print("\n--- FINAL TEST SCORES (Vault Opened) ---")
        print(classification_report(y_test, test_preds, zero_division=0))
    else:
        print("\nFinal test set kept hidden (vault mode). Set REVEAL_TEST_METRICS=True when ready.")

    return {
        "model": model,
        "feature_names": list(X_train.columns),
        "train_report": train_report,
        "val_report": val_report,
        "test_report": test_report,
        "y_val_true": y_val,
        "y_val_pred": val_preds,
    }


# 1. Load labels + structured data source
df_train_master = pd.read_csv(base_dir / "clean_train.csv")
df_labels = df_train_master[["patient_id", "triage_acuity"]]

if USE_STRUCTURED_FEATURES:
    structured_columns = [
        col for col in df_train_master.columns
        if col not in STRUCTURED_EXCLUDE_COLUMNS
    ]
    df_structured = df_train_master[structured_columns].copy()
    if "patient_id" not in df_structured.columns:
        df_structured.insert(0, "patient_id", df_train_master["patient_id"])

    numeric_cols = df_structured.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [
        col for col in df_structured.columns
        if col not in numeric_cols and col != "patient_id"
    ]

    if numeric_cols:
        df_structured[numeric_cols] = df_structured[numeric_cols].fillna(0)
    if categorical_cols:
        df_structured[categorical_cols] = df_structured[categorical_cols].fillna("missing")
        df_structured = pd.get_dummies(
            df_structured,
            columns=categorical_cols,
            drop_first=False,
        )
else:
    df_structured = df_labels[["patient_id"]].copy()

feature_files = {
    "pipeline_a": ("PIPELINE A (Full Context with Subjective Words)", "features_pipeline_A_full.csv"),
    "pipeline_b": ("PIPELINE B (Objective Words Only)", "features_pipeline_B_objective.csv"),
    "pipeline_a_2_phrase": ("PIPELINE A 2-PHRASE (Full Context)", "features_pipeline_A_full_2_phrase.csv"),
    "pipeline_b_2_phrase": ("PIPELINE B 2-PHRASE (Objective Words Only)", "features_pipeline_B_objective_2_phrase.csv"),
}

experiments = []
for pipeline_key, (pipeline_name, file_name) in feature_files.items():
    feature_df = pd.read_csv(base_dir / file_name)
    if USE_STRUCTURED_FEATURES:
        merged_df = pd.merge(df_structured, feature_df, on="patient_id", how="inner")
        merged_df = pd.merge(merged_df, df_labels, on="patient_id", how="inner")
    else:
        merged_df = pd.merge(feature_df, df_labels, on="patient_id", how="inner")
    experiments.append((pipeline_key, pipeline_name, merged_df))

# Build one shared split from IDs present in every experiment so all model comparisons are fair.
common_ids = set(df_labels["patient_id"])
for _, _, data in experiments:
    common_ids &= set(data["patient_id"])

df_labels_for_split = df_labels[df_labels["patient_id"].isin(common_ids)].copy()
split_ids = build_split_patient_ids(df_labels_for_split)

# 4. Set up run folder + registry
runs_dir.mkdir(parents=True, exist_ok=True)
registry = load_registry(registry_path)
run_id = next_run_id(registry)
run_dir = runs_dir / run_id
run_dir.mkdir(parents=True, exist_ok=True)

run_record = {
    "run_id": run_id,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "uses_structured_features": USE_STRUCTURED_FEATURES,
    "split_config": {
        "train_fraction": 0.64,
        "validation_fraction": 0.16,
        "test_fraction": 0.20,
        "test_metrics_revealed": REVEAL_TEST_METRICS,
        "counts": split_ids["counts"],
    },
    "models": [],
}

print(f"\nSaving artifacts under: {run_dir}")
if USE_STRUCTURED_FEATURES:
    print(f"Structured features enabled: {df_structured.shape[1] - 1} columns (excluding patient_id)")
print(
    "Dataset split counts -> "
    f"Train: {split_ids['counts']['train']}, "
    f"Validation: {split_ids['counts']['val']}, "
    f"Test: {split_ids['counts']['test']}"
)

comparison_rows = []
best_by_pipeline = {}

for pipeline_key, pipeline_name, data in experiments:
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_result = train_model(data, split_ids, pipeline_name, "RandomForest", rf_model)

    rf_path = run_dir / f"{pipeline_key}_random_forest.pkl"
    with rf_path.open("wb") as f:
        pickle.dump(rf_result["model"], f)

    run_record["models"].append(
        {
            "pipeline": pipeline_key,
            "model_name": "random_forest",
            "artifact_path": rf_path.relative_to(base_dir).as_posix(),
            "validation_accuracy": metric_from_report(rf_result["val_report"], "accuracy"),
            "validation_weighted_precision": metric_from_report(rf_result["val_report"], "precision"),
            "validation_weighted_recall": metric_from_report(rf_result["val_report"], "recall"),
            "validation_weighted_f1": float(rf_result["val_report"]["weighted avg"]["f1-score"]),
            "test_weighted_f1": float(rf_result["test_report"]["weighted avg"]["f1-score"]) if REVEAL_TEST_METRICS else None,
            "top_feature_weights": top_feature_weights(rf_result["model"], rf_result["feature_names"]),
        }
    )
    comparison_rows.append(
        {
            "model_key": f"{pipeline_key}_rf",
            "pipeline": pipeline_key,
            "model_type": "rf",
            "accuracy": metric_from_report(rf_result["val_report"], "accuracy"),
            "precision": metric_from_report(rf_result["val_report"], "precision"),
            "recall": metric_from_report(rf_result["val_report"], "recall"),
            "f1-score": metric_from_report(rf_result["val_report"], "f1-score"),
        }
    )

    rf_val_f1 = metric_from_report(rf_result["val_report"], "f1-score")
    current_best = best_by_pipeline.get(pipeline_key)
    if (current_best is None) or (rf_val_f1 > current_best["val_f1"]):
        best_by_pipeline[pipeline_key] = {
            "val_f1": rf_val_f1,
            "pipeline": pipeline_key,
            "model_name": "RandomForest",
            "y_true": rf_result["y_val_true"],
            "y_pred": rf_result["y_val_pred"],
        }

    if HAS_XGBOOST:
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_result = train_model(data, split_ids, pipeline_name, "XGBoost", xgb_model)

        xgb_path = run_dir / f"{pipeline_key}_xgboost.pkl"
        with xgb_path.open("wb") as f:
            pickle.dump(xgb_result["model"], f)

        run_record["models"].append(
            {
                "pipeline": pipeline_key,
                "model_name": "xgboost",
                "artifact_path": xgb_path.relative_to(base_dir).as_posix(),
                "validation_accuracy": metric_from_report(xgb_result["val_report"], "accuracy"),
                "validation_weighted_precision": metric_from_report(xgb_result["val_report"], "precision"),
                "validation_weighted_recall": metric_from_report(xgb_result["val_report"], "recall"),
                "validation_weighted_f1": float(xgb_result["val_report"]["weighted avg"]["f1-score"]),
                "test_weighted_f1": float(xgb_result["test_report"]["weighted avg"]["f1-score"]) if REVEAL_TEST_METRICS else None,
                "top_feature_weights": top_feature_weights(xgb_result["model"], xgb_result["feature_names"]),
            }
        )
        comparison_rows.append(
            {
                "model_key": f"{pipeline_key}_xgb",
                "pipeline": pipeline_key,
                "model_type": "xgb",
                "accuracy": metric_from_report(xgb_result["val_report"], "accuracy"),
                "precision": metric_from_report(xgb_result["val_report"], "precision"),
                "recall": metric_from_report(xgb_result["val_report"], "recall"),
                "f1-score": metric_from_report(xgb_result["val_report"], "f1-score"),
            }
        )

        xgb_val_f1 = metric_from_report(xgb_result["val_report"], "f1-score")
        current_best = best_by_pipeline.get(pipeline_key)
        if (current_best is None) or (xgb_val_f1 > current_best["val_f1"]):
            best_by_pipeline[pipeline_key] = {
                "val_f1": xgb_val_f1,
                "pipeline": pipeline_key,
                "model_name": "XGBoost",
                "y_true": xgb_result["y_val_true"],
                "y_pred": xgb_result["y_val_pred"],
            }
    else:
        print("XGBoost is not installed. Skipping XGBoost models.")

comparison_image = run_dir / "model_comparison_metrics.png"
build_comparison_plot(comparison_rows, comparison_image)
run_record["comparison_image_path"] = comparison_image.relative_to(base_dir).as_posix()

if best_by_pipeline:
    class_labels = sorted(df_labels_for_split["triage_acuity"].unique().tolist())
    confusion_image = run_dir / "best_models_confusion_matrix_grid.png"
    build_confusion_matrix_grid(best_by_pipeline, class_labels, confusion_image)
    run_record["confusion_matrix_image_path"] = confusion_image.relative_to(base_dir).as_posix()
    run_record["best_validation_models_by_pipeline"] = [
        {
            "pipeline": pipeline_key,
            "model_name": best_entry["model_name"],
            "validation_weighted_f1": float(best_entry["val_f1"]),
        }
        for pipeline_key, best_entry in best_by_pipeline.items()
    ]

registry["latest_run"] = run_id
registry.setdefault("runs", []).append(run_record)

with registry_path.open("w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

print(f"\nRun complete: {run_id}")
print(f"Registry updated: {registry_path}")
print(f"Comparison plot saved: {comparison_image}")
if best_by_pipeline:
    print(f"Confusion matrix saved: {confusion_image}")