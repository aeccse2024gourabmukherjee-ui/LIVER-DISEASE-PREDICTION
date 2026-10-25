"""
tests/test_preprocessing.py
Unit tests for data preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import (
    clean_column_name,
    standardize_dataframe,
    load_and_preprocess_data,
    build_preprocessor,
    split_data,
    FEATURE_NAMES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


def test_clean_column_name_non_breaking_spaces():
    """Verify non-breaking space \\xa0 is stripped properly."""
    raw = "\xa0Alkphos Alkaline Phosphotase"
    cleaned = clean_column_name(raw)
    assert cleaned == "alkaline_phosphotase"


def test_clean_column_name_various_variations():
    """Verify different column name casings and slight variants."""
    assert clean_column_name("Age of the patient") == "age"
    assert clean_column_name("Gender of the patient") == "gender"
    assert clean_column_name("Total Bilirubin") == "total_bilirubin"
    assert clean_column_name("Direct Bilirubin") == "direct_bilirubin"
    assert clean_column_name("\xa0Sgpt Alamine Aminotransferase") == "alamine_aminotransferase"
    assert clean_column_name("Sgot Aspartate Aminotransferase") == "aspartate_aminotransferase"
    assert clean_column_name("Total Protiens") == "total_proteins"
    assert clean_column_name("\xa0ALB Albumin") == "albumin"
    assert clean_column_name("A/G Ratio Albumin and Globulin Ratio") == "albumin_globulin_ratio"
    assert clean_column_name("Result") == "result"


def test_standardize_dataframe():
    """Verify standardize_dataframe updates columns correctly."""
    df_raw = pd.DataFrame({
        "Age of the patient": [45],
        "\xa0Total Bilirubin": [1.2],
        "Result": [1]
    })
    df_clean = standardize_dataframe(df_raw)
    assert "age" in df_clean.columns
    assert "total_bilirubin" in df_clean.columns
    assert "result" in df_clean.columns


def test_load_and_preprocess_data():
    """Verify data loading, deduplication, and target mapping."""
    X, y, df = load_and_preprocess_data()
    assert len(X) == len(y) == len(df)
    assert len(df) > 1000
    # Verify all expected features are present in X
    for feat in FEATURE_NAMES:
        assert feat in X.columns
    # Verify target values are mapped to 0 and 1
    assert set(y.unique()).issubset({0, 1})
    # Verify no duplicates in cleaned df
    assert df.duplicated().sum() == 0


def test_build_preprocessor_transformation():
    """Verify ColumnTransformer transforms features without NaNs."""
    X, y, _ = load_and_preprocess_data()
    prep = build_preprocessor()
    X_trans = prep.fit_transform(X)
    
    assert X_trans.shape[0] == len(X)
    # 9 numerical scaled + 1 one-hot encoded gender column = 10 columns
    assert X_trans.shape[1] == 10
    assert not np.isnan(X_trans).any()


def test_split_data_stratification():
    """Verify stratified train-test split maintains exact target proportions."""
    X, y, _ = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.01
