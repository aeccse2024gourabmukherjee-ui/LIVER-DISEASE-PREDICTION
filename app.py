"""
app.py
Phase 6: Comprehensive Liver Disease Prediction Web Application.
Built with Streamlit and Scikit-Learn.

Sections:
1. Home Dashboard
2. Dataset Overview
3. Exploratory Data Analysis (EDA)
4. Model Performance Comparison
5. Liver Disease Prediction Form
6. About the Project
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src module is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing import (
    load_and_preprocess_data,
    FEATURE_NAMES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN
)
from src.predict import (
    predict_single,
    load_model,
    PHYSIOLOGICAL_RANGES,
    validate_patient_input
)

# Page configuration
st.set_page_config(
    page_title="HepatoScan | Liver Disease Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a polished, medical-grade UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 14px;
        border-radius: 6px;
        color: #856404;
        font-size: 0.92rem;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .result-box-positive {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 20px;
        border-radius: 8px;
        color: #721c24;
        margin-top: 15px;
    }
    .result-box-negative {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 20px;
        border-radius: 8px;
        color: #155724;
        margin-top: 15px;
    }
    .clinical-hint {
        font-size: 0.82rem;
        color: #6c757d;
        margin-top: -10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cached_data():
    """Loads and caches the cleaned dataset for fast app rendering."""
    X, y, df = load_and_preprocess_data()
    return X, y, df


@st.cache_data
def load_cached_metadata():
    """Loads model training metadata."""
    meta_path = os.path.join("models", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def show_home(df, meta):
    st.markdown('<div class="main-header">🩺 HepatoScan: AI-Powered Liver Disease Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An intelligent clinical decision-support system analyzing liver function biomarkers.</div>', unsafe_allow_html=True)

    # Key Statistics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Patients Analyzed", value="30,691", delta="Full Cohort")
    with col2:
        st.metric(label="Selected Model", value="Random Forest", delta="Composite Score: 0.977")
    with col3:
        st.metric(label="Screening Recall", value="96.13%", delta="+40.3% vs Baseline")
    with col4:
        st.metric(label="Model ROC-AUC", value="0.9988", delta="Excellent Discrimination")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📌 Project Motivation & Clinical Need")
        st.write("""
        Liver disease represents one of the leading global causes of morbidity and mortality, often remaining 
        asymptomatic until reaching advanced stages such as cirrhosis or hepatic failure. Early detection 
        through routine biochemical liver function tests (LFTs) is crucial for timely therapeutic intervention.

        **HepatoScan** leverages machine learning trained on tens of thousands of patient biochemical records 
        to identify subtle patterns across standard hepatic biomarkers, including:
        - **Bilirubin levels** (Total and Direct)
        - **Liver transaminases** (ALT/SGPT and AST/SGOT)
        - **Alkaline Phosphatase (ALP)**
        - **Serum Proteins and Albumin-to-Globulin Ratio**
        """)

        st.subheader("🚀 Quick Workflow Guide")
        st.markdown("""
        1. **Dataset Overview**: Inspect patient cohort statistics, distributions, and missing data handling.
        2. **EDA**: Explore hepatic biomarker correlations and diagnostic distributions.
        3. **Model Performance**: Compare Logistic Regression, Decision Tree, and Random Forest pipelines.
        4. **Prediction Tool**: Enter patient laboratory values to obtain real-time risk assessment.
        5. **About**: Review the machine learning pipeline architecture, SMOTE balancing, and methodology.
        """)

    with col_right:
        st.subheader("🧬 Liver Function Biomarkers")
        biomarker_info = pd.DataFrame({
            "Biomarker": ["Total Bilirubin", "Direct Bilirubin", "Alkaline Phosphatase (ALP)", "ALT (SGPT)", "AST (SGOT)", "Total Proteins", "Albumin", "A/G Ratio"],
            "Normal Range": ["0.1 - 1.2 mg/dL", "0.0 - 0.3 mg/dL", "44 - 147 IU/L", "7 - 56 IU/L", "10 - 40 IU/L", "6.0 - 8.3 g/dL", "3.5 - 5.5 g/dL", "0.8 - 2.0"],
            "Clinical Significance": [
                "Bile pigment; elevated in jaundice & hemolysis",
                "Conjugated fraction; indicates biliary obstruction",
                "Biliary canalicular enzyme; elevated in cholestasis",
                "Liver-specific enzyme; hallmarks hepatocellular injury",
                "Mitochondrial enzyme; elevated in acute hepatic necrosis",
                "Synthesized by liver; reflects hepatic synthetic reserve",
                "Major plasma protein; drops in chronic liver failure",
                "Proportion of albumin to globulins in serum"
            ]
        })
        st.dataframe(biomarker_info, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical & Research Disclaimer:</strong> This web application was created for machine learning 
        educational and research purposes. Predictions generated by this system are experimental risk estimates 
        and do not constitute medical diagnoses.
    </div>
    """, unsafe_allow_html=True)


def show_dataset_overview(df):
    st.header("📊 Dataset Overview & Clinical Characteristics")
    st.write("""
    This section provides an automated audit of the patient dataset. In accordance with strict machine learning 
    practices, duplicates were removed to prevent train-test data leakage, and missing values were handled 
    inside the preprocessing pipeline using robust median and mode imputation.
    """)

    tab1, tab2, tab3 = st.tabs(["📋 Data Sample & Schema", "📈 Statistical Summary", "🔍 Data Quality & Missingness"])

    with tab1:
        st.subheader("Cleaned Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**Total Records (Deduplicated):** {len(df):,}")
        with c2:
            st.info(f"**Total Features:** {df.shape[1] - 1} (9 Numerical, 1 Categorical)")
        with c3:
            st.info(f"**Target Classes:** 1 (Liver Disease) vs 0 (Healthy)")

    with tab2:
        st.subheader("Descriptive Statistics for Numerical Biomarkers")
        st.dataframe(df[NUMERICAL_FEATURES].describe().T.round(3), use_container_width=True)

    with tab3:
        st.subheader("Data Cleaning & Preprocessing Protocol")
        st.markdown("""
        - **Deduplication:** Kaggle LPD dataset contained duplicated observations. Removing them yielded **19,368 unique patients**, completely safeguarding validation integrity.
        - **Numerical Features (9):** Imputed with the column **Median** to protect against distortion from extreme clinical outliers (e.g. ALT > 2,000 IU/L), followed by **StandardScaler**.
        - **Categorical Feature (`gender`):** Imputed with the **Most Frequent** value and transformed via **OneHotEncoder(drop='first')**.
        - **Target Mapping:** Standardized to binary format (`1 = Liver Disease`, `0 = Healthy`).
        - **SMOTE Pipeline:** Applied Synthetic Minority Over-sampling Technique **strictly on training folds** via `imblearn.pipeline.Pipeline`, preventing synthetic leakage into test evaluation.
        """)


def show_eda(df):
    st.header("🔬 Exploratory Data Analysis (EDA)")
    st.write("Visual exploration of biomarker distributions, target class proportions, and inter-feature correlations.")

    tab1, tab2, tab3 = st.tabs(["🎯 Target Distribution", "🔥 Correlation Matrix", "📊 Feature Distributions"])

    with tab1:
        st.subheader("Distribution of Liver Disease in Cohort")
        col1, col2 = st.columns([3, 2])
        with col1:
            target_img = os.path.join("models", "figures", "target_distribution.png")
            if os.path.exists(target_img):
                st.image(target_img, use_container_width=True)
            else:
                st.warning("Target distribution figure not found.")
        with col2:
            st.markdown("""
            **Class Imbalance Analysis:**
            - **Liver Disease Patients (1):** ~71.3% of cohort
            - **Healthy Patients (0):** ~28.7% of cohort
            
            **Machine Learning Implication:**
            Because healthy patients represent the minority class in this screening dataset, evaluating solely 
            by raw Accuracy can be deceptive. A naive classifier guessing class 1 for every patient would yield 
            71.3% accuracy but fail to identify any healthy individual. 
            Hence, we incorporate **SMOTE**, **Precision**, **Recall**, and **ROC-AUC** into model evaluation.
            """)

    with tab2:
        st.subheader("Biomarker Correlation Matrix")
        corr_img = os.path.join("models", "figures", "correlation_heatmap.png")
        if os.path.exists(corr_img):
            st.image(corr_img, use_container_width=True)
        else:
            st.warning("Correlation figure not found.")
        st.info("""
        **Clinical Key Insights:**
        - **Total Bilirubin & Direct Bilirubin** exhibit very strong positive correlation (> 0.85), matching physiological expectations.
        - **ALT (SGPT) and AST (SGOT)** are strongly correlated liver transaminases that rise together during acute hepatic injury.
        - **Albumin and Total Proteins** display moderate positive correlation, as albumin constitutes over 50% of total circulating serum protein.
        """)

    with tab3:
        st.subheader("Feature Distributions Conditioned on Diagnosis")
        dist_img = os.path.join("models", "figures", "feature_distributions.png")
        if os.path.exists(dist_img):
            st.image(dist_img, use_container_width=True)
        else:
            st.warning("Distribution figure not found.")


def show_model_comparison(meta):
    st.header("🏆 Model Training & Performance Comparison")
    st.write("""
    Three candidate algorithms were trained inside identical end-to-end pipelines with SMOTE oversampling:
    **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**.
    """)

    if meta is None or "metrics_summary" not in meta:
        st.warning("Model metadata not found. Please train models first.")
        return

    metrics = meta["metrics_summary"]
    df_metrics = pd.DataFrame(metrics).T[
        ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC", "Specificity"]
    ]

    st.subheader("📊 Test Set Benchmark Table")
    st.dataframe(
        df_metrics.style.highlight_max(axis=0, color="#d4edda").format("{:.4f}"),
        use_container_width=True
    )

    st.subheader("📈 Visual Comparison Charts")
    c1, c2 = st.columns(2)
    with c1:
        comp_img = os.path.join("models", "figures", "models_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Multi-Metric Model Comparison", use_container_width=True)
    with c2:
        roc_img = os.path.join("models", "figures", "roc_curves.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="ROC Curves & AUC Scores", use_container_width=True)

    st.subheader("🔍 Confusion Matrices (Test Cohort = 3,874 Patients)")
    cm_img = os.path.join("models", "figures", "confusion_matrices.png")
    if os.path.exists(cm_img):
        st.image(cm_img, caption="Confusion Matrices across All Models", use_container_width=True)

    st.subheader("💡 Why Random Forest Was Selected")
    st.markdown("""
    In a medical screening task:
    1. **High Recall (96.13%)**: The Random Forest caught 2,655 out of 2,762 actual disease cases on the unseen test set, reducing False Negatives to just 107.
    2. **Near-Zero False Positives (6)**: Demonstrates exceptional specificity (99.46%), meaning healthy individuals are not unduly alarmed.
    3. **Superior ROC-AUC (0.9988)**: Indicates near-perfect diagnostic discrimination between healthy and diseased patient distributions across all classification thresholds.
    """)


def show_prediction_form():
    st.header("🧪 Patient Liver Disease Risk Prediction Form")
    st.write("Enter patient biochemical laboratory parameters below to generate an immediate, AI-assisted risk prediction.")

    # Preset Sample buttons for quick testing
    st.markdown("#### ⚡ Quick-Fill Sample Profiles")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    preset = None
    with col_btn1:
        if st.button("📋 Load Healthy Patient Profile", use_container_width=True):
            preset = "healthy"
    with col_btn2:
        if st.button("🚨 Load High Risk Profile", use_container_width=True):
            preset = "high_risk"
    with col_btn3:
        if st.button("🔄 Reset to Default Values", use_container_width=True):
            preset = "default"

    # Default values based on preset
    if preset == "healthy":
        init_age, init_gender = 32, "Female"
        init_tb, init_db = 0.8, 0.2
        init_alp, init_alt, init_ast = 120.0, 22.0, 24.0
        init_tp, init_alb, init_agr = 7.2, 4.2, 1.40
    elif preset == "high_risk":
        init_age, init_gender = 58, "Male"
        init_tb, init_db = 8.5, 4.2
        init_alp, init_alt, init_ast = 520.0, 85.0, 110.0
        init_tp, init_alb, init_agr = 6.4, 2.7, 0.72
    else:
        init_age, init_gender = 45, "Male"
        init_tb, init_db = 1.2, 0.4
        init_alp, init_alt, init_ast = 190.0, 35.0, 42.0
        init_tp, init_alb, init_agr = 6.8, 3.4, 1.00

    # Prediction Form
    with st.form("liver_prediction_form"):
        st.subheader("1. Patient Demographics")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            age = st.number_input("Age of the Patient (years)", min_value=1, max_value=110, value=init_age, step=1)
            st.markdown('<div class="clinical-hint">Plausible range: 1 - 100 years</div>', unsafe_allow_html=True)
        with f_col2:
            gender_index = 0 if init_gender == "Male" else 1
            gender = st.selectbox("Biological Sex", options=["Male", "Female"], index=gender_index)
            st.markdown('<div class="clinical-hint">Used for hormone-influenced baseline reference</div>', unsafe_allow_html=True)

        st.subheader("2. Serum Bilirubin Panel")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            total_bilirubin = st.number_input(
                "Total Bilirubin (mg/dL)",
                min_value=0.1, max_value=80.0, value=float(init_tb), step=0.1, format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal reference: 0.1 - 1.2 mg/dL</div>', unsafe_allow_html=True)
        with b_col2:
            direct_bilirubin = st.number_input(
                "Direct Bilirubin (mg/dL)",
                min_value=0.05, max_value=30.0, value=float(init_db), step=0.1, format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal reference: 0.0 - 0.3 mg/dL (Must not exceed Total Bilirubin)</div>', unsafe_allow_html=True)

        st.subheader("3. Hepatic Enzymes (Transaminases & Phosphatase)")
        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            alkaline_phosphotase = st.number_input(
                "Alkaline Phosphatase - ALP (IU/L)",
                min_value=20.0, max_value=3000.0, value=float(init_alp), step=5.0
            )
            st.markdown('<div class="clinical-hint">Normal: 44 - 147 IU/L</div>', unsafe_allow_html=True)
        with e_col2:
            alamine_aminotransferase = st.number_input(
                "Alamine Aminotransferase - ALT / SGPT (IU/L)",
                min_value=5.0, max_value=3000.0, value=float(init_alt), step=1.0
            )
            st.markdown('<div class="clinical-hint">Normal: 7 - 56 IU/L</div>', unsafe_allow_html=True)
        with e_col3:
            aspartate_aminotransferase = st.number_input(
                "Aspartate Aminotransferase - AST / SGOT (IU/L)",
                min_value=5.0, max_value=5000.0, value=float(init_ast), step=1.0
            )
            st.markdown('<div class="clinical-hint">Normal: 10 - 40 IU/L</div>', unsafe_allow_html=True)

        st.subheader("4. Serum Proteins Panel")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            total_proteins = st.number_input(
                "Total Proteins (g/dL)",
                min_value=1.0, max_value=15.0, value=float(init_tp), step=0.1, format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal: 6.0 - 8.3 g/dL</div>', unsafe_allow_html=True)
        with p_col2:
            albumin = st.number_input(
                "Albumin (g/dL)",
                min_value=0.5, max_value=8.0, value=float(init_alb), step=0.1, format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal: 3.5 - 5.5 g/dL</div>', unsafe_allow_html=True)
        with p_col3:
            albumin_globulin_ratio = st.number_input(
                "A/G Ratio (Albumin/Globulin)",
                min_value=0.1, max_value=5.0, value=float(init_agr), step=0.05, format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal: 0.8 - 2.0</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Run Liver Disease Assessment", use_container_width=True)

    if submitted:
        patient_dict = {
            "age": age,
            "gender": gender,
            "total_bilirubin": total_bilirubin,
            "direct_bilirubin": direct_bilirubin,
            "alkaline_phosphotase": alkaline_phosphotase,
            "alamine_aminotransferase": alamine_aminotransferase,
            "aspartate_aminotransferase": aspartate_aminotransferase,
            "total_proteins": total_proteins,
            "albumin": albumin,
            "albumin_globulin_ratio": albumin_globulin_ratio
        }

        with st.spinner("Processing laboratory biomarkers through the fitted ML pipeline..."):
            result = predict_single(patient_dict)

        if not result["success"]:
            st.error("Validation Error(s) Detected:")
            for err in result["errors"]:
                st.write(f"- {err}")
        else:
            st.markdown("### 📋 Clinical Assessment Results")
            pred = result["prediction"]
            prob = result["probability"]
            risk_level = result["risk_level"]
            risk_desc = result["risk_description"]

            # Visual Display Box
            if pred == 1:
                st.markdown(f"""
                <div class="result-box-positive">
                    <h3 style="margin: 0; color: #721c24;">⚠️ Prediction: High Probability of Liver Disease</h3>
                    <p style="margin-top: 8px; font-size: 1.05rem;"><strong>Assessment:</strong> {risk_desc}</p>
                    <p style="margin: 0;"><strong>Estimated Disease Probability:</strong> {prob * 100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-negative">
                    <h3 style="margin: 0; color: #155724;">✅ Prediction: Low Probability / Healthy Liver Profile</h3>
                    <p style="margin-top: 8px; font-size: 1.05rem;"><strong>Assessment:</strong> {risk_desc}</p>
                    <p style="margin: 0;"><strong>Estimated Disease Probability:</strong> {prob * 100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # Probability Gauge Progress Bar
            st.markdown("#### Estimated Hepatic Pathology Probability")
            st.progress(float(prob))
            
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.metric("Predicted Class", f"{result['label']}")
            with p_col2:
                st.metric("Risk Category", risk_level)
            with p_col3:
                st.metric("Model Confidence", f"{max(prob, 1 - prob) * 100:.1f}%")

            # Elevated Biomarker Highlights
            st.markdown("#### 🔬 Biomarker Flag Analysis")
            flags = []
            if total_bilirubin > 1.2:
                flags.append(f"**Total Bilirubin elevated** ({total_bilirubin} mg/dL vs normal <= 1.2)")
            if direct_bilirubin > 0.3:
                flags.append(f"**Direct Bilirubin elevated** ({direct_bilirubin} mg/dL vs normal <= 0.3)")
            if alkaline_phosphotase > 147:
                flags.append(f"**Alkaline Phosphatase elevated** ({alkaline_phosphotase} IU/L vs normal <= 147)")
            if alamine_aminotransferase > 56:
                flags.append(f"**ALT / SGPT elevated** ({alamine_aminotransferase} IU/L vs normal <= 56)")
            if aspartate_aminotransferase > 40:
                flags.append(f"**AST / SGOT elevated** ({aspartate_aminotransferase} IU/L vs normal <= 40)")
            if albumin < 3.5:
                flags.append(f"**Albumin depressed** ({albumin} g/dL vs normal >= 3.5)")
            if albumin_globulin_ratio < 0.8:
                flags.append(f"**A/G Ratio depressed** ({albumin_globulin_ratio} vs normal >= 0.8)")

            if flags:
                st.warning("Potentially Elevated / Abnormal Biomarkers Detected:\n" + "\n".join([f"- {f}" for f in flags]))
            else:
                st.success("All entered biomarkers are within conventional physiological reference boundaries.")

            # Medical Disclaimer
            st.markdown(f"""
            <div class="disclaimer-box">
                <strong>⚠️ Clinical Advisory:</strong> {result['disclaimer']}
            </div>
            """, unsafe_allow_html=True)


def show_about():
    st.header("📖 About the Project")
    st.write("""
    ### Student Machine Learning Project: Liver Disease Risk Prediction
    This project was conceived and implemented as a comprehensive Machine Learning exploration 
    into clinical decision support, medical biomarker analytics, and robust end-to-end ML engineering.
    """)

    st.subheader("🛠️ Technology Stack")
    st.markdown("""
    - **Language:** Python 3.12
    - **Data Manipulation & Analysis:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn
    - **Machine Learning & Pipeline:** Scikit-Learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`)
    - **Imbalance Handling:** Imbalanced-learn (`SMOTE`, `imblearn.pipeline.Pipeline`)
    - **Model Persistence:** Joblib
    - **Web Framework:** Streamlit
    - **Testing:** Pytest
    """)

    st.subheader("⚙️ Machine Learning Pipeline Architecture")
    st.markdown("""
    ```
    Raw Patient Input (10 Features)
           │
           ▼
    Column Sanitizer (Normalize non-breaking spaces & naming)
           │
           ▼
    ColumnTransformer:
      ├── Numerical (9): Median Imputer -> StandardScaler
      └── Categorical (1): Mode Imputer -> OneHotEncoder(drop='first')
           │
           ▼
    SMOTE (Synthetic Minority Over-sampling, applied only during training)
           │
           ▼
    Random Forest Classifier (150 Estimators, max_depth=12)
           │
           ▼
    Prediction Output (Class Label, Risk Tier, Calibrated Probability)
    ```
    """)

    st.subheader("🛡️ Mitigations for Data Leakage & Overfitting")
    st.markdown("""
    1. **Strict Deduplication:** kaggle LPD contains synthetic duplicates. Deduplication removed 11,323 identical samples prior to splitting.
    2. **Stratified Split:** 80% train / 20% test partitioning preserves the target class balance across both folds.
    3. **SMOTE inside Pipeline:** SMOTE resamples only within the training fold. Test data remains 100% pristine and unadulterated.
    4. **Hyperparameter Regularization:** `max_depth` and `min_samples_leaf` constraints were enforced to avoid memorization.
    """)

    st.subheader("⚠️ Limitations & Future Work")
    st.markdown("""
    - **Single Cohort:** The dataset originated from regional hospital records; validation across diverse international cohorts is required.
    - **Absence of Viral Serology:** Viral hepatitis markers (HBsAg, Anti-HCV) and ultrasound imaging are not included.
    - **Future Scope:** Adding SHAP (SHapley Additive exPlanations) for local explainability and integrating longitudinal tracking.
    """)


def main():
    # Load dataset and metadata
    try:
        X, y, df = load_cached_data()
        meta = load_cached_metadata()
    except Exception as e:
        st.error(f"Error loading project data: {e}")
        return

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/liver.png", width=70)
    st.sidebar.title("HepatoScan AI")
    st.sidebar.markdown("**Liver Disease Risk Prediction**")
    st.sidebar.markdown("---")

    nav_options = [
        "🏠 Home Dashboard",
        "📊 Dataset Overview",
        "🔬 Exploratory Data Analysis",
        "🏆 Model Comparison",
        "🧪 Prediction Tool",
        "📖 About Project"
    ]
    choice = st.sidebar.radio("Navigation", nav_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Model Status")
    if meta:
        st.sidebar.success(f"**Active Model:** {meta.get('best_model_name', 'Random Forest')}")
        st.sidebar.info(f"**Recall:** {meta['best_model_metrics']['Recall (Sensitivity)']*100:.1f}%")
        st.sidebar.info(f"**ROC-AUC:** {meta['best_model_metrics']['ROC-AUC']:.4f}")
    else:
        st.sidebar.warning("Model metadata unavailable.")

    # Render selected section
    if choice == "🏠 Home Dashboard":
        show_home(df, meta)
    elif choice == "📊 Dataset Overview":
        show_dataset_overview(df)
    elif choice == "🔬 Exploratory Data Analysis":
        show_eda(df)
    elif choice == "🏆 Model Comparison":
        show_model_comparison(meta)
    elif choice == "🧪 Prediction Tool":
        show_prediction_form()
    elif choice == "📖 About Project":
        show_about()


if __name__ == "__main__":
    main()
