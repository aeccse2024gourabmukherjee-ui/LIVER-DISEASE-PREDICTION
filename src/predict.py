"""
src/predict.py
Phase 4: Reusable Inference and Prediction Functions for Liver Disease Prediction.

Provides:
- Model loader with caching.
- Input validation against plausible physiological medical ranges.
- Single-instance prediction returning class, label, probability, risk level, and medical disclaimer.
- Batch prediction on Pandas DataFrames.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, Optional, List

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import (
    FEATURE_NAMES,
    clean_column_name,
    standardize_dataframe
)

MODEL_PATH_DEFAULT = os.path.join("models", "best_liver_model.joblib")

# Reference physiological ranges for clinical sanity checks
PHYSIOLOGICAL_RANGES = {
    "age": {"min": 1, "max": 120, "label": "Age (years)", "unit": "years"},
    "gender": {"allowed": ["Male", "Female"], "label": "Biological Sex"},
    "total_bilirubin": {"min": 0.1, "max": 80.0, "label": "Total Bilirubin", "unit": "mg/dL"},
    "direct_bilirubin": {"min": 0.05, "max": 30.0, "label": "Direct Bilirubin", "unit": "mg/dL"},
    "alkaline_phosphotase": {"min": 20.0, "max": 3000.0, "label": "Alkaline Phosphatase (ALP)", "unit": "IU/L"},
    "alamine_aminotransferase": {"min": 5.0, "max": 3000.0, "label": "Alamine Aminotransferase (ALT/SGPT)", "unit": "IU/L"},
    "aspartate_aminotransferase": {"min": 5.0, "max": 5000.0, "label": "Aspartate Aminotransferase (AST/SGOT)", "unit": "IU/L"},
    "total_proteins": {"min": 1.0, "max": 15.0, "label": "Total Proteins", "unit": "g/dL"},
    "albumin": {"min": 0.5, "max": 8.0, "label": "Albumin", "unit": "g/dL"},
    "albumin_globulin_ratio": {"min": 0.1, "max": 5.0, "label": "A/G Ratio", "unit": "ratio"}
}

_CACHED_MODEL = None


def load_model(model_path: Optional[str] = None):
    """Loads and caches the fitted pipeline from disk."""
    global _CACHED_MODEL
    path = model_path or MODEL_PATH_DEFAULT
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'. Please train the model first.")
    
    if _CACHED_MODEL is None or model_path is not None:
        _CACHED_MODEL = joblib.load(path)
    return _CACHED_MODEL


def validate_patient_input(patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates patient input against physiological bounds and schema.
    Returns (is_valid, list_of_warning_or_error_messages).
    """
    errors = []
    
    # Standardize input keys
    standardized = {}
    for k, v in patient_data.items():
        standardized[clean_column_name(k)] = v
        
    # Check missing required fields
    for field in FEATURE_NAMES:
        if field not in standardized or standardized[field] is None:
            errors.append(f"Missing required field: '{field}'")
            
    if errors:
        return False, errors
        
    # Validate gender
    gender_val = str(standardized["gender"]).strip().title()
    if gender_val not in ["Male", "Female"]:
        errors.append(f"Invalid gender '{standardized['gender']}'. Must be 'Male' or 'Female'.")
        
    # Validate numerical bounds
    for feat, limits in PHYSIOLOGICAL_RANGES.items():
        if feat == "gender":
            continue
        val = standardized.get(feat)
        try:
            val_float = float(val)
            if np.isnan(val_float):
                errors.append(f"Value for '{limits['label']}' cannot be NaN.")
            elif val_float < limits["min"] or val_float > limits["max"]:
                errors.append(
                    f"'{limits['label']}' value {val_float} is outside plausible physiological range "
                    f"[{limits['min']}, {limits['max']} {limits.get('unit', '')}]."
                )
        except (ValueError, TypeError):
            errors.append(f"'{limits['label']}' must be a valid number, got '{val}'.")

    # Domain clinical logic check: Direct Bilirubin cannot strictly exceed Total Bilirubin
    try:
        tb = float(standardized.get("total_bilirubin", 0))
        db = float(standardized.get("direct_bilirubin", 0))
        if db > tb + 0.1:  # slight float tolerance
            errors.append(f"Direct Bilirubin ({db} mg/dL) cannot exceed Total Bilirubin ({tb} mg/dL).")
    except Exception:
        pass

    return len(errors) == 0, errors


def get_risk_assessment(probability: float) -> Tuple[str, str]:
    """Translates predicted probability into clinical risk level and guidance."""
    if probability < 0.25:
        return "Low Risk", "Biomarkers are largely within normal liver function parameters."
    elif probability < 0.50:
        return "Borderline / Mild Risk", "Biomarkers suggest mild elevation; routine monitoring advised."
    elif probability < 0.75:
        return "Moderate Risk", "Significant biomarker deviation suggestive of hepatic stress."
    else:
        return "High Risk", "Strong clinical biomarker profile indicative of liver pathology."


def predict_single(
    patient_dict: Dict[str, Any],
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs inference for a single patient record.
    Returns structured results including prediction, probability,
    risk level, and mandatory disclaimer.
    """
    model = load_model(model_path)
    
    # Standardize dictionary keys
    clean_dict = {clean_column_name(k): v for k, v in patient_dict.items()}
    
    # Validate
    is_valid, validation_messages = validate_patient_input(clean_dict)
    if not is_valid:
        return {
            "success": False,
            "errors": validation_messages,
            "prediction": None,
            "label": None,
            "probability": None,
            "disclaimer": "This prediction tool is experimental and for educational use only."
        }
        
    # Build single-row DataFrame with canonical ordering
    df_single = pd.DataFrame([{f: clean_dict[f] for f in FEATURE_NAMES}])
    df_single["gender"] = df_single["gender"].astype(str).str.strip().str.title()
    for col in FEATURE_NAMES:
        if col != "gender":
            df_single[col] = pd.to_numeric(df_single[col], errors="coerce")
            
    # Predict
    pred = int(model.predict(df_single)[0])
    
    # Calculate probability if model supports predict_proba
    prob = None
    if hasattr(model, "predict_proba"):
        prob_array = model.predict_proba(df_single)[0]
        # Class 1 is positive (Liver Disease)
        prob = float(prob_array[1])
        risk_level, risk_desc = get_risk_assessment(prob)
    else:
        risk_level = "High Risk" if pred == 1 else "Low Risk"
        risk_desc = "Classification prediction without probability estimates."
        
    label = "Liver Disease Detected" if pred == 1 else "Healthy / No Liver Disease"
    
    return {
        "success": True,
        "errors": [],
        "prediction": pred,
        "label": label,
        "probability": round(prob, 4) if prob is not None else None,
        "risk_level": risk_level,
        "risk_description": risk_desc,
        "disclaimer": (
            "NOTICE: This prediction is generated by an experimental machine learning model "
            "for educational/research purposes. It does NOT constitute a clinical diagnosis. "
            "Always consult a qualified healthcare professional for medical interpretation."
        ),
        "input_features": clean_dict
    }


def predict_batch(
    df: pd.DataFrame,
    model_path: Optional[str] = None
) -> pd.DataFrame:
    """Performs batch inference on a DataFrame."""
    model = load_model(model_path)
    df_clean = standardize_dataframe(df)
    
    for f in FEATURE_NAMES:
        if f not in df_clean.columns:
            raise ValueError(f"Batch DataFrame missing required feature: '{f}'")
            
    df_features = df_clean[FEATURE_NAMES].copy()
    preds = model.predict(df_features)
    df_results = df_features.copy()
    df_results["predicted_class"] = preds
    df_results["predicted_label"] = np.where(preds == 1, "Liver Disease Detected", "Healthy")
    
    if hasattr(model, "predict_proba"):
        df_results["disease_probability"] = model.predict_proba(df_features)[:, 1].round(4)
        
    return df_results


if __name__ == "__main__":
    print("Testing src/predict.py...")
    # Sample patient 1: Likely diseased (high bilirubin and enzymes)
    sample_sick = {
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
    
    res1 = predict_single(sample_sick)
    print("\nResult for High Risk Patient Sample:")
    print(f"  Predicted Label: {res1['label']}")
    print(f"  Probability:     {res1['probability'] * 100:.2f}%")
    print(f"  Risk Level:      {res1['risk_level']}")
    print(f"  Disclaimer:      {res1['disclaimer']}")
    
    # Sample patient 2: Healthy patient profile
    sample_healthy = {
        "age": 28,
        "gender": "Female",
        "total_bilirubin": 0.8,
        "direct_bilirubin": 0.2,
        "alkaline_phosphotase": 160,
        "alamine_aminotransferase": 20,
        "aspartate_aminotransferase": 22,
        "total_proteins": 7.0,
        "albumin": 4.1,
        "albumin_globulin_ratio": 1.25
    }
    res2 = predict_single(sample_healthy)
    print("\nResult for Healthy Patient Sample:")
    print(f"  Predicted Label: {res2['label']}")
    print(f"  Probability:     {res2['probability'] * 100:.2f}%")
    print(f"  Risk Level:      {res2['risk_level']}")
    
    # Test Invalid input
    invalid_sample = {
        "age": 250,  # Impossible age
        "gender": "Alien",  # Invalid gender
        "total_bilirubin": 5.0,
        "direct_bilirubin": 10.0  # DB > TB
    }
    res_inv = predict_single(invalid_sample)
    print("\nResult for Invalid Sample:")
    print(f"  Success: {res_inv['success']}")
    print(f"  Errors:  {res_inv['errors']}")
