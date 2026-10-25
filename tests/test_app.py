"""
tests/test_app.py
Smoke tests for the Streamlit application structure.
"""

import os
import pytest


def test_app_file_exists():
    """Verify app.py exists."""
    assert os.path.exists("app.py"), "app.py does not exist."


def test_required_artifacts_exist():
    """Verify all saved model artifacts and figures exist."""
    required_files = [
        os.path.join("models", "best_liver_model.joblib"),
        os.path.join("models", "model_metadata.json"),
        os.path.join("models", "all_models_summary.csv"),
        os.path.join("models", "figures", "models_comparison.png"),
        os.path.join("models", "figures", "confusion_matrices.png"),
        os.path.join("models", "figures", "roc_curves.png"),
        os.path.join("models", "figures", "target_distribution.png"),
        os.path.join("models", "figures", "correlation_heatmap.png"),
        os.path.join("models", "figures", "feature_distributions.png"),
        os.path.join("models", "figures", "best_model_cm.png"),
    ]
    for rf in required_files:
        assert os.path.exists(rf), f"Expected artifact '{rf}' not found."
