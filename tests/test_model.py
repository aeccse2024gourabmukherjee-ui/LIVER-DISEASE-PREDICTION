"""
tests/test_model.py
Unit tests for model persistence, prediction functions, and boundary validations.
"""

import os
import pytest
import pandas as pd
from src.predict import (
    load_model,
    predict_single,
    predict_batch,
    validate_patient_input,
    MODEL_PATH_DEFAULT
)


def test_saved_model_file_exists():
    """Verify best_liver_model.joblib exists and is loadable."""
    assert os.path.exists(MODEL_PATH_DEFAULT), f"Model file not found at {MODEL_PATH_DEFAULT}"
    model = load_model(MODEL_PATH_DEFAULT)
    assert model is not None
    assert hasattr(model, "predict")


def test_predict_single_valid_high_risk():
    """Verify single prediction on clearly elevated biomarker patient."""
    patient = {
        "age": 62,
        "gender": "Male",
        "total_bilirubin": 10.9,
        "direct_bilirubin": 5.5,
        "alkaline_phosphotase": 699,
        "alamine_aminotransferase": 64,
        "aspartate_aminotransferase": 100,
        "total_proteins": 7.5,
        "albumin": 3.2,
        "albumin_globulin_ratio": 0.74
    }
    result = predict_single(patient)
    assert result["success"] is True
    assert result["prediction"] in [0, 1]
    assert result["prediction"] == 1  # Should flag high risk
    assert result["label"] == "Liver Disease Detected"
    assert 0.0 <= result["probability"] <= 1.0
    assert result["probability"] > 0.8
    assert "disclaimer" in result
    assert len(result["disclaimer"]) > 20


def test_predict_single_valid_healthy():
    """Verify single prediction on healthy patient profile."""
    patient = {
        "age": 28,
        "gender": "Female",
        "total_bilirubin": 0.7,
        "direct_bilirubin": 0.1,
        "alkaline_phosphotase": 130,
        "alamine_aminotransferase": 18,
        "aspartate_aminotransferase": 20,
        "total_proteins": 7.1,
        "albumin": 4.2,
        "albumin_globulin_ratio": 1.35
    }
    result = predict_single(patient)
    assert result["success"] is True
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_single_invalid_inputs():
    """Verify graceful failure and detailed errors on invalid inputs."""
    # 1. Missing fields
    res_missing = predict_single({"age": 30})
    assert res_missing["success"] is False
    assert len(res_missing["errors"]) > 0

    # 2. Out of range values
    res_out_of_bounds = predict_single({
        "age": 250,  # Impossible age
        "gender": "Male",
        "total_bilirubin": 1.0,
        "direct_bilirubin": 0.2,
        "alkaline_phosphotase": 150,
        "alamine_aminotransferase": 25,
        "aspartate_aminotransferase": 30,
        "total_proteins": 7.0,
        "albumin": 3.8,
        "albumin_globulin_ratio": 1.0
    })
    assert res_out_of_bounds["success"] is False
    assert any("Age" in err for err in res_out_of_bounds["errors"])

    # 3. Direct Bilirubin greater than Total Bilirubin
    res_inconsistent_bilirubin = predict_single({
        "age": 45,
        "gender": "Male",
        "total_bilirubin": 1.0,
        "direct_bilirubin": 5.0,  # Inconsistent
        "alkaline_phosphotase": 150,
        "alamine_aminotransferase": 25,
        "aspartate_aminotransferase": 30,
        "total_proteins": 7.0,
        "albumin": 3.8,
        "albumin_globulin_ratio": 1.0
    })
    assert res_inconsistent_bilirubin["success"] is False
    assert any("Direct Bilirubin" in err for err in res_inconsistent_bilirubin["errors"])


def test_predict_batch():
    """Verify batch prediction returns DataFrame with prediction columns."""
    sample_df = pd.DataFrame([
        {
            "Age of the patient": 62,
            "Gender of the patient": "Male",
            "Total Bilirubin": 10.9,
            "Direct Bilirubin": 5.5,
            "\xa0Alkphos Alkaline Phosphotase": 699,
            "\xa0Sgpt Alamine Aminotransferase": 64,
            "Sgot Aspartate Aminotransferase": 100,
            "Total Protiens": 7.5,
            "\xa0ALB Albumin": 3.2,
            "A/G Ratio Albumin and Globulin Ratio": 0.74
        },
        {
            "Age of the patient": 28,
            "Gender of the patient": "Female",
            "Total Bilirubin": 0.7,
            "Direct Bilirubin": 0.1,
            "\xa0Alkphos Alkaline Phosphotase": 130,
            "\xa0Sgpt Alamine Aminotransferase": 18,
            "Sgot Aspartate Aminotransferase": 20,
            "Total Protiens": 7.1,
            "\xa0ALB Albumin": 4.2,
            "A/G Ratio Albumin and Globulin Ratio": 1.35
        }
    ])
    
    batch_res = predict_batch(sample_df)
    assert len(batch_res) == 2
    assert "predicted_class" in batch_res.columns
    assert "predicted_label" in batch_res.columns
    assert "disease_probability" in batch_res.columns
    assert batch_res["predicted_class"].iloc[0] == 1
