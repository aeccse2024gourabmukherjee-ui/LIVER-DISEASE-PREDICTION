"""
src/data_preprocessing.py
Phase 2: Data Preprocessing and Pipeline Construction for Liver Disease Prediction.

Handles:
- Dataset discovery and loading with robust encoding.
- Column name standardization and sanitization.
- Deduplication to prevent train-test data leakage.
- Target transformation (1: Liver Disease -> 1, 2: Healthy -> 0).
- Scikit-learn ColumnTransformer for numerical (impute median + scale)
  and categorical (impute mode + one-hot encode) features.
- Train/Test splitting with stratification.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Canonical feature names and clean aliases
FEATURE_NAMES = [
    "age",
    "gender",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphotase",
    "alamine_aminotransferase",
    "aspartate_aminotransferase",
    "total_proteins",
    "albumin",
    "albumin_globulin_ratio"
]

NUMERICAL_FEATURES = [
    "age",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphotase",
    "alamine_aminotransferase",
    "aspartate_aminotransferase",
    "total_proteins",
    "albumin",
    "albumin_globulin_ratio"
]

CATEGORICAL_FEATURES = ["gender"]

TARGET_COLUMN = "result"

# Raw-to-standard column name mapping
RAW_COLUMN_MAPPING = {
    "age of the patient": "age",
    "gender of the patient": "gender",
    "total bilirubin": "total_bilirubin",
    "direct bilirubin": "direct_bilirubin",
    "alkphos alkaline phosphotase": "alkaline_phosphotase",
    "sgpt alamine aminotransferase": "alamine_aminotransferase",
    "sgot aspartate aminotransferase": "aspartate_aminotransferase",
    "total protiens": "total_proteins",
    "total proteins": "total_proteins",
    "alb albumin": "albumin",
    "a/g ratio albumin and globulin ratio": "albumin_globulin_ratio",
    "result": "result"
}


def find_dataset(custom_path: str = None) -> str:
    """Finds the dataset path automatically."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    candidates = [
        os.path.join("dataset", "liver_disease_data.csv"),
        os.path.join("dataset", "Liver Patient Dataset (LPD)_train.csv"),
        "Liver Patient Dataset (LPD)_train.csv",
        "dataset/liver_disease_data.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find the liver disease dataset in standard paths.")


def clean_column_name(col: str) -> str:
    """Sanitizes raw column names by removing non-breaking spaces and normalizing."""
    # Replace non-breaking spaces, strip, and lowercase
    cleaned = col.replace("\xa0", " ").strip().lower()
    # Remove extra spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    
    # Check if exact match in mapping
    if cleaned in RAW_COLUMN_MAPPING:
        return RAW_COLUMN_MAPPING[cleaned]
    
    # Partial matching rules for robustness
    if "age" in cleaned:
        return "age"
    if "gender" in cleaned or "sex" in cleaned:
        return "gender"
    if "direct" in cleaned and "bilirubin" in cleaned:
        return "direct_bilirubin"
    if "total" in cleaned and "bilirubin" in cleaned:
        return "total_bilirubin"
    if "alkphos" in cleaned or "alkaline" in cleaned:
        return "alkaline_phosphotase"
    if "sgpt" in cleaned or "alamine" in cleaned:
        return "alamine_aminotransferase"
    if "sgot" in cleaned or "aspartate" in cleaned:
        return "aspartate_aminotransferase"
    if "protien" in cleaned or "protein" in cleaned:
        return "total_proteins"
    if "alb" in cleaned and "ratio" not in cleaned:
        return "albumin"
    if "ratio" in cleaned or "a/g" in cleaned:
        return "albumin_globulin_ratio"
    if "result" in cleaned or "target" in cleaned or "class" in cleaned:
        return "result"
        
    return cleaned


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names of an input DataFrame."""
    df_clean = df.copy()
    new_cols = [clean_column_name(c) for c in df_clean.columns]
    df_clean.columns = new_cols
    return df_clean


def load_and_preprocess_data(
    file_path: str = None,
    drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads dataset, standardizes columns, handles duplicates,
    and returns features X, target y, and the cleaned DataFrame.
    
    Target mapping:
      1 (Liver Disease) -> 1 (Positive)
      2 (Healthy / Non-disease) -> 0 (Negative)
    """
    path = find_dataset(file_path)
    # Attempt loading with latin1 or utf-8
    try:
        df = pd.read_csv(path, encoding="latin1")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8")
        
    df = standardize_dataframe(df)
    
    # Clean string values in gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.title()
        df["gender"] = df["gender"].replace({"Nan": np.nan, "None": np.nan, "": np.nan})
        
    # Deduplicate if requested to prevent train-test data leakage
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        
    # Ensure target column is present
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {list(df.columns)}")
        
    # Map target: 1 -> 1 (Liver Disease), 2 -> 0 (Healthy)
    # Filter any invalid targets
    df = df[df[TARGET_COLUMN].isin([1, 2])].reset_index(drop=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({1: 1, 2: 0}).astype(int)
    y = df[TARGET_COLUMN].copy()
    
    # Select only known features
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[available_features].copy()
    
    return X, y, df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    1. Imputes numerical columns with median and scales them with StandardScaler.
    2. Imputes categorical columns with most frequent value and encodes with OneHotEncoder.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into stratified train and test sets to prevent leakage.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


if __name__ == "__main__":
    X, y, df = load_and_preprocess_data()
    print(f"Preprocessed DataFrame shape: {df.shape}")
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Target class distribution:\n{y.value_counts(normalize=True).round(4)}")
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    prep = build_preprocessor()
    X_train_trans = prep.fit_transform(X_train)
    print(f"Transformed train features shape: {X_train_trans.shape}")
    print("Preprocessing pipeline built and validated successfully!")
