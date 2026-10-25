"""
dataset_inspector.py
Phase 1: Automated Dataset Analysis for Liver Disease Prediction Dataset.
"""

import os
import pandas as pd
import numpy as np

def find_dataset_path():
    candidates = [
        os.path.join("dataset", "liver_disease_data.csv"),
        os.path.join("dataset", "Liver Patient Dataset (LPD)_train.csv"),
        "Liver Patient Dataset (LPD)_train.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate liver disease dataset file.")

def inspect_dataset():
    path = find_dataset_path()
    print(f"Loading dataset from: {path}")
    
    # Try reading with utf-8 or latin1 if special characters exist
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")
        
    print("\n" + "="*60)
    print("1. DATASET SHAPE & BASIC INFO")
    print("="*60)
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    
    print("\nRaw Column Names:")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] '{col}' (dtype: {df[col].dtype})")
        
    print("\nSample 5 rows:")
    print(df.head())
    
    print("\n" + "="*60)
    print("2. TARGET VARIABLE ANALYSIS")
    print("="*60)
    # The last column is typically the target or named 'Result'
    target_col = df.columns[-1]
    print(f"Identified Target Column: '{target_col}'")
    value_counts = df[target_col].value_counts(dropna=False)
    print(f"Target Value Counts:\n{value_counts}")
    print(f"Target Percentage Distribution:\n{(value_counts / len(df) * 100).round(2)}%")
    
    print("\n" + "="*60)
    print("3. MISSING VALUES ANALYSIS")
    print("="*60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(3)
    missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
    print(missing_df[missing_df["Missing Count"] > 0])
    if (missing == 0).all():
        print("No missing values found.")

    print("\n" + "="*60)
    print("4. DUPLICATES ANALYSIS")
    print("="*60)
    dup_count = df.duplicated().sum()
    print(f"Duplicate Rows Count: {dup_count} ({(dup_count / len(df) * 100):.2f}%)")

    print("\n" + "="*60)
    print("5. FEATURE TYPES")
    print("="*60)
    features = [c for c in df.columns if c != target_col]
    num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[features].select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")
    for c in cat_cols:
        print(f"  Categorical '{c}' unique values: {df[c].unique()[:10]}")
        
    print("\n" + "="*60)
    print("6. STATISTICAL SUMMARY (NUMERICAL)")
    print("="*60)
    print(df.describe().T)

    print("\n" + "="*60)
    print("7. OUTLIER SUMMARY (IQR METHOD)")
    print("="*60)
    for col in num_cols:
        series = df[col].dropna()
        q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(series)*100:.2f}%) [Range: {series.min():.2f} to {series.max():.2f}]")

if __name__ == "__main__":
    inspect_dataset()
