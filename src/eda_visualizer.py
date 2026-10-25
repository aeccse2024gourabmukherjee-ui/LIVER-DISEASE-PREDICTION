"""
src/eda_visualizer.py
Phase 5: Exploratory Data Analysis (EDA) Visualizations for Liver Disease Dataset.

Generates high-resolution visualizations:
1. Target class distribution (pie chart & bar chart).
2. Numerical feature distributions (histograms with KDE).
3. Correlation heatmap for numerical features.
4. Age vs Bilirubin scatter by diagnosis.
5. Best model dedicated confusion matrix.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import (
    load_and_preprocess_data,
    NUMERICAL_FEATURES,
    TARGET_COLUMN
)

FIGURES_DIR = os.path.join("models", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_target_distribution_plot(df: pd.DataFrame, output_path: str):
    """Generates a combined bar and donut chart for target distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df[TARGET_COLUMN].value_counts()
    labels = ["Liver Disease (1)", "Healthy (0)"]
    colors = ["#e74c3c", "#2ecc71"]
    
    # Bar Chart
    bars = ax1.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_title("Target Class Counts", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Number of Patients", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax1.annotate(f"{h:,}\n({pct:.1f}%)",
                     (bar.get_x() + bar.get_width() / 2, h / 2),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    # Donut Chart
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2)
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    ax2.set_title("Class Proportion Breakdown", fontsize=13, fontweight="bold", pad=10)

    plt.suptitle("Liver Disease Dataset - Target Class Distribution", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Target distribution plot saved to: {output_path}")


def generate_correlation_heatmap(df: pd.DataFrame, output_path: str):
    """Generates a correlation heatmap for numerical features and target."""
    num_cols_with_target = NUMERICAL_FEATURES + [TARGET_COLUMN]
    corr = df[num_cols_with_target].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Spearman/Pearson Correlation Matrix of Hepatic Biomarkers", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Correlation heatmap saved to: {output_path}")


def generate_feature_distributions(df: pd.DataFrame, output_path: str):
    """Generates distribution plots for numerical features by class."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(NUMERICAL_FEATURES):
        ax = axes[i]
        sns.kdeplot(
            data=df,
            x=col,
            hue=TARGET_COLUMN,
            common_norm=False,
            palette={1: "#e74c3c", 0: "#2ecc71"},
            ax=ax,
            fill=True,
            alpha=0.3
        )
        ax.set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        # Log scale for skewed enzymes if max is large
        if df[col].max() > 100:
            ax.set_xscale("log")
            ax.set_xlabel(f"{col} (log scale)", fontsize=9)

    plt.suptitle("Feature Distributions Conditioned on Diagnosis (0: Healthy, 1: Diseased)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Feature distributions plot saved to: {output_path}")


def generate_best_model_cm(output_path: str):
    """Creates a dedicated high-resolution confusion matrix for the chosen best model."""
    import json
    meta_path = "models/model_metadata.json"
    if not os.path.exists(meta_path):
        return
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    best_name = meta.get("best_model_name", "Random Forest")
    metrics = meta["best_model_metrics"]
    cm = np.array(metrics["Confusion_Matrix"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=["Healthy (0)", "Liver Disease (1)"],
        yticklabels=["Healthy (0)", "Liver Disease (1)"],
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(
        f"Selected Best Model: {best_name}\n"
        f"Accuracy: {metrics['Accuracy']*100:.1f}% | Recall: {metrics['Recall (Sensitivity)']*100:.1f}% | F1: {metrics['F1-Score']:.3f}",
        fontsize=11,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Predicted Diagnosis", fontsize=10, fontweight="bold")
    ax.set_ylabel("Actual Diagnosis", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Dedicated best model CM saved to: {output_path}")


def main():
    print("\n" + "="*70)
    print("PHASE 5: GENERATING EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("="*70)
    
    _, _, df = load_and_preprocess_data()
    
    target_plot = os.path.join(FIGURES_DIR, "target_distribution.png")
    corr_plot = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    dist_plot = os.path.join(FIGURES_DIR, "feature_distributions.png")
    best_cm_plot = os.path.join(FIGURES_DIR, "best_model_cm.png")
    
    generate_target_distribution_plot(df, target_plot)
    generate_correlation_heatmap(df, corr_plot)
    generate_feature_distributions(df, dist_plot)
    generate_best_model_cm(best_cm_plot)
    
    print("\n[SUCCESS] All EDA visualizations generated!")


if __name__ == "__main__":
    main()
