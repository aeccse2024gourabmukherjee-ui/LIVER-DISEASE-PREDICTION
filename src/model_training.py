"""
src/model_training.py
Phase 3 & 4: Model Training, Evaluation, Comparison, and Saving.

Models trained and compared:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Evaluation metrics:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC, Specificity
- Confusion Matrix & Classification Report

The optimal model is selected based on clinical evaluation metrics
(prioritizing Recall, F1-score, and ROC-AUC for medical disease detection).
The complete fitted pipeline and metadata are saved using Joblib.
"""

import os
import sys
import json
import joblib
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data_preprocessing import (
    load_and_preprocess_data,
    build_preprocessor,
    split_data,
    FEATURE_NAMES
)

FIGURES_DIR = os.path.join("models", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


def get_candidate_models() -> dict:
    """Returns the dictionary of candidate models with tuned hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    use_smote: bool = True
) -> Tuple[dict, dict, dict]:
    """
    Trains each candidate model inside an imbalanced-learn Pipeline.
    Evaluates on the unseen test set, calculates all metrics,
    and returns models_dict, pipelines_dict, and results_dict.
    """
    models = get_candidate_models()
    trained_pipelines = {}
    metrics_summary = {}
    detailed_reports = {}

    print("\n" + "="*70)
    print("PHASE 3: TRAINING AND COMPARING MACHINE LEARNING MODELS")
    print(f"SMOTE applied on training data: {use_smote}")
    print("="*70)

    for name, model in models.items():
        print(f"\n---> Training: {name}...")
        
        # Build individual pipeline: Preprocessor -> (SMOTE) -> Model
        preprocessor = build_preprocessor()
        steps = [("preprocessor", preprocessor)]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("classifier", model))
        
        pipeline = ImbPipeline(steps)
        pipeline.fit(X_train, y_train)
        
        # Test predictions
        y_pred = pipeline.predict(X_test)
        y_proba = None
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
        else:
            roc_auc = 0.0

        # Metrics calculation
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        report = classification_report(
            y_test, y_pred,
            target_names=["Healthy (0)", "Liver Disease (1)"],
            output_dict=True
        )

        metrics_summary[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall (Sensitivity)": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Specificity": round(specificity, 4),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Confusion_Matrix": cm
        }
        detailed_reports[name] = report
        trained_pipelines[name] = pipeline

        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f} (Crucial for clinical screening)")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  ROC-AUC:     {roc_auc:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return trained_pipelines, metrics_summary, detailed_reports


def plot_model_comparisons(metrics_summary: dict, output_path: str):
    """Plots a side-by-side bar chart of Accuracy, Precision, Recall, F1, and ROC-AUC."""
    df_metrics = pd.DataFrame(metrics_summary).T[
        ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC"]
    ]
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 6))
    
    df_metrics.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
    ax.set_title("Machine Learning Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Comparison chart saved to: {output_path}")


def plot_confusion_matrices(metrics_summary: dict, output_path: str):
    """Plots confusion matrices for all models side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, (name, metrics) in zip(axes, metrics_summary.items()):
        cm = np.array(metrics["Confusion_Matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Healthy (0)", "Liver Disease (1)"],
            yticklabels=["Healthy (0)", "Liver Disease (1)"]
        )
        ax.set_title(f"{name}\nRecall: {metrics['Recall (Sensitivity)']:.2f} | F1: {metrics['F1-Score']:.2f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Confusion matrices saved to: {output_path}")


def plot_roc_curves(trained_pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series, output_path: str):
    """Plots ROC curves for all candidate models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, pipeline in trained_pipelines.items():
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"ROC curves saved to: {output_path}")


def select_best_model(metrics_summary: dict) -> str:
    """
    Selects the best model.
    In medical screening:
    A weighted composite score is used:
    Composite Score = 0.40 * Recall + 0.35 * F1-Score + 0.25 * ROC-AUC.
    This guarantees we pick a model that catches liver disease (high recall)
    without excessive false alarms (balanced F1 and ROC-AUC).
    """
    best_name = None
    best_score = -1.0
    
    print("\n" + "="*70)
    print("MODEL SELECTION BASED ON CLINICAL EVALUATION METRICS:")
    print("Composite Score = 0.40 * Recall + 0.35 * F1-Score + 0.25 * ROC-AUC")
    print("="*70)
    
    for name, m in metrics_summary.items():
        score = (
            0.40 * m["Recall (Sensitivity)"] +
            0.35 * m["F1-Score"] +
            0.25 * m["ROC-AUC"]
        )
        print(f"  {name:20s}: Composite Score = {score:.4f} "
              f"(Recall: {m['Recall (Sensitivity)']:.4f}, F1: {m['F1-Score']:.4f}, ROC-AUC: {m['ROC-AUC']:.4f})")
        if score > best_score:
            best_score = score
            best_name = name

    print(f"\n>>> Selected Best Model: {best_name} (Composite Score: {best_score:.4f})")
    return best_name


def save_best_model(
    best_name: str,
    trained_pipelines: dict,
    metrics_summary: dict,
    model_path: str = "models/best_liver_model.joblib",
    meta_path: str = "models/model_metadata.json"
):
    """
    Saves the complete fitted preprocessing and prediction pipeline using Joblib,
    along with metadata in JSON format.
    """
    best_pipeline = trained_pipelines[best_name]
    
    # Save the fitted pipeline
    joblib.dump(best_pipeline, model_path)
    print(f"\n[OK] Complete fitted pipeline successfully saved to: {model_path}")

    # Prepare metadata
    metadata = {
        "best_model_name": best_name,
        "features": FEATURE_NAMES,
        "metrics_summary": metrics_summary,
        "best_model_metrics": metrics_summary[best_name],
        "class_mapping": {
            "0": "Healthy / No Liver Disease Detected",
            "1": "Liver Disease Detected"
        }
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Model metadata saved to: {meta_path}")
    
    # Save a CSV summary table for quick viewing
    df_metrics = pd.DataFrame(metrics_summary).T
    df_metrics.to_csv("models/all_models_summary.csv", index_label="Model")
    print("[OK] Metrics table saved to: models/all_models_summary.csv")


def main():
    # 1. Load and split data
    X, y, df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 2. Train and evaluate models
    trained_pipelines, metrics_summary, reports = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, use_smote=True
    )
    
    # 3. Generate Visualizations
    comp_plot = os.path.join(FIGURES_DIR, "models_comparison.png")
    cm_plot = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    roc_plot = os.path.join(FIGURES_DIR, "roc_curves.png")
    
    plot_model_comparisons(metrics_summary, comp_plot)
    plot_confusion_matrices(metrics_summary, cm_plot)
    plot_roc_curves(trained_pipelines, X_test, y_test, roc_plot)
    
    # 4. Select best model
    best_model_name = select_best_model(metrics_summary)
    
    # 5. Save best model and metadata
    save_best_model(best_model_name, trained_pipelines, metrics_summary)
    
    print("\n" + "="*70)
    print("PHASES 2, 3, AND 4 COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
