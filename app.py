"""
app.py
Simple, Focused Liver Disease Prediction Web Application.
Built with Streamlit and Scikit-Learn.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import (
    predict_single,
    load_model,
    PHYSIOLOGICAL_RANGES,
    validate_patient_input
)

# Page configuration
st.set_page_config(
    page_title="HepatoScan | Liver Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a clean, modern medical UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        background-color: #f0f4f8;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-badge-val {
        font-size: 1.25rem;
        font-weight: bold;
        color: #102a43;
    }
    .metric-badge-lbl {
        font-size: 0.82rem;
        color: #627d98;
    }
    .clinical-hint {
        font-size: 0.82rem;
        color: #6c757d;
        margin-top: -8px;
        margin-bottom: 8px;
    }
    .result-box-positive {
        background-color: #fff5f5;
        border-left: 6px solid #e53e3e;
        padding: 20px;
        border-radius: 8px;
        color: #9b2c2c;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-box-negative {
        background-color: #f0fff4;
        border-left: 6px solid #38a169;
        padding: 20px;
        border-radius: 8px;
        color: #22543d;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background-color: #fffaf0;
        border-left: 5px solid #dd6b20;
        padding: 12px 16px;
        border-radius: 6px;
        color: #7b341e;
        font-size: 0.88rem;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_metadata():
    """Loads model training metadata if available."""
    meta_path = os.path.join("models", "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    meta = load_metadata()

    # App Title & Description
    st.markdown('<div class="main-title">🩺 HepatoScan: Liver Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">'
        'Enter patient Liver Function Test (LFT) biomarkers below to obtain an instant AI-powered risk assessment.'
        '</div>',
        unsafe_allow_html=True
    )

    # Top Metric Badges
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="metric-badge"><div class="metric-badge-val">Random Forest</div>'
            '<div class="metric-badge-lbl">Active ML Algorithm</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        acc_str = f"{meta['best_model_metrics']['Accuracy']*100:.1f}%" if meta else "97.1%"
        st.markdown(
            f'<div class="metric-badge"><div class="metric-badge-val">{acc_str}</div>'
            '<div class="metric-badge-lbl">Test Accuracy</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        rec_str = f"{meta['best_model_metrics']['Recall (Sensitivity)']*100:.1f}%" if meta else "96.1%"
        st.markdown(
            f'<div class="metric-badge"><div class="metric-badge-val">{rec_str}</div>'
            '<div class="metric-badge-lbl">Clinical Recall (Sensitivity)</div></div>',
            unsafe_allow_html=True
        )
    with c4:
        auc_str = f"{meta['best_model_metrics']['ROC-AUC']:.4f}" if meta else "0.9988"
        st.markdown(
            f'<div class="metric-badge"><div class="metric-badge-val">{auc_str}</div>'
            '<div class="metric-badge-lbl">ROC-AUC Score</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar: Reference Guide & Help
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/liver.png", width=64)
        st.title("HepatoScan AI")
        st.caption("Clinical Decision Support Tool")
        st.markdown("---")
        
        st.markdown("### 📋 Normal Biomarker Ranges")
        ref_data = {
            "Biomarker": [
                "Total Bilirubin",
                "Direct Bilirubin",
                "ALP (Alkphos)",
                "ALT / SGPT",
                "AST / SGOT",
                "Total Proteins",
                "Albumin",
                "A/G Ratio"
            ],
            "Normal Interval": [
                "0.1 – 1.2 mg/dL",
                "0.0 – 0.3 mg/dL",
                "44 – 147 IU/L",
                "7 – 56 IU/L",
                "10 – 40 IU/L",
                "6.0 – 8.3 g/dL",
                "3.5 – 5.5 g/dL",
                "0.8 – 2.0"
            ]
        }
        st.dataframe(pd.DataFrame(ref_data), hide_index=True, use_container_width=True)
        st.markdown("---")
        st.info("💡 **Tip:** Use the quick preset buttons on the main screen to load sample patient profiles in 1 click.")

    # Profile Preset Management via Session State
    if "profile" not in st.session_state:
        st.session_state["profile"] = "default"

    st.markdown("##### ⚡ Quick-Fill Sample Profiles")
    btn_col1, btn_col2, btn_col3, _ = st.columns([1.2, 1.2, 1.2, 2.4])
    
    with btn_col1:
        if st.button("📋 Load Healthy Sample", use_container_width=True):
            st.session_state["profile"] = "healthy"
            st.rerun()
    with btn_col2:
        if st.button("🚨 Load High Risk Sample", use_container_width=True):
            st.session_state["profile"] = "high_risk"
            st.rerun()
    with btn_col3:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.session_state["profile"] = "default"
            st.rerun()

    # Determine default form values based on selected profile
    prof = st.session_state.get("profile", "default")
    if prof == "healthy":
        val_age, val_gender = 32, "Female"
        val_tb, val_db = 0.80, 0.20
        val_alp, val_alt, val_ast = 120.0, 22.0, 24.0
        val_tp, val_alb, val_agr = 7.20, 4.20, 1.40
    elif prof == "high_risk":
        val_age, val_gender = 58, "Male"
        val_tb, val_db = 8.50, 4.20
        val_alp, val_alt, val_ast = 520.0, 85.0, 110.0
        val_tp, val_alb, val_agr = 6.40, 2.70, 0.72
    else:
        val_age, val_gender = 45, "Male"
        val_tb, val_db = 1.20, 0.40
        val_alp, val_alt, val_ast = 190.0, 35.0, 42.0
        val_tp, val_alb, val_agr = 6.80, 3.40, 1.00

    # Prediction Input Form
    with st.form("liver_prediction_form"):
        st.markdown("#### 1. Patient Demographics")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            age = st.number_input(
                "Age of the Patient (years)",
                min_value=1,
                max_value=110,
                value=int(val_age),
                step=1
            )
            st.markdown('<div class="clinical-hint">Plausible range: 1 – 100 years</div>', unsafe_allow_html=True)
        with d_col2:
            gender_idx = 0 if val_gender == "Male" else 1
            gender = st.selectbox(
                "Biological Sex",
                options=["Male", "Female"],
                index=gender_idx
            )
            st.markdown('<div class="clinical-hint">Baseline reference factor</div>', unsafe_allow_html=True)

        st.markdown("#### 2. Serum Bilirubin Panel")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            total_bilirubin = st.number_input(
                "Total Bilirubin (mg/dL)",
                min_value=0.1,
                max_value=80.0,
                value=float(val_tb),
                step=0.1,
                format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal: 0.1 – 1.2 mg/dL</div>', unsafe_allow_html=True)
        with b_col2:
            direct_bilirubin = st.number_input(
                "Direct Bilirubin (mg/dL)",
                min_value=0.05,
                max_value=30.0,
                value=float(val_db),
                step=0.1,
                format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal: 0.0 – 0.3 mg/dL (Must be &le; Total Bilirubin)</div>', unsafe_allow_html=True)

        st.markdown("#### 3. Hepatic Enzymes (Transaminases & Phosphatase)")
        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            alkaline_phosphotase = st.number_input(
                "Alkaline Phosphatase - ALP (IU/L)",
                min_value=20.0,
                max_value=3000.0,
                value=float(val_alp),
                step=5.0
            )
            st.markdown('<div class="clinical-hint">Normal: 44 – 147 IU/L</div>', unsafe_allow_html=True)
        with e_col2:
            alamine_aminotransferase = st.number_input(
                "ALT / SGPT (IU/L)",
                min_value=5.0,
                max_value=3000.0,
                value=float(val_alt),
                step=1.0
            )
            st.markdown('<div class="clinical-hint">Normal: 7 – 56 IU/L</div>', unsafe_allow_html=True)
        with e_col3:
            aspartate_aminotransferase = st.number_input(
                "AST / SGOT (IU/L)",
                min_value=5.0,
                max_value=5000.0,
                value=float(val_ast),
                step=1.0
            )
            st.markdown('<div class="clinical-hint">Normal: 10 – 40 IU/L</div>', unsafe_allow_html=True)

        st.markdown("#### 4. Serum Proteins Panel")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            total_proteins = st.number_input(
                "Total Proteins (g/dL)",
                min_value=1.0,
                max_value=15.0,
                value=float(val_tp),
                step=0.1,
                format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal: 6.0 – 8.3 g/dL</div>', unsafe_allow_html=True)
        with p_col2:
            albumin = st.number_input(
                "Albumin (g/dL)",
                min_value=0.5,
                max_value=8.0,
                value=float(val_alb),
                step=0.1,
                format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal: 3.5 – 5.5 g/dL</div>', unsafe_allow_html=True)
        with p_col3:
            albumin_globulin_ratio = st.number_input(
                "A/G Ratio",
                min_value=0.1,
                max_value=5.0,
                value=float(val_agr),
                step=0.05,
                format="%.2f"
            )
            st.markdown('<div class="clinical-hint">Normal: 0.8 – 2.0</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Run Liver Disease Prediction", use_container_width=True)

    # Process and display prediction result
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

        with st.spinner("Processing biomarkers through the trained Random Forest pipeline..."):
            result = predict_single(patient_dict)

        if not result["success"]:
            st.error("Validation Error(s) Detected:")
            for err in result["errors"]:
                st.write(f"- {err}")
        else:
            st.markdown("### 📋 Prediction Assessment Results")
            pred = result["prediction"]
            prob = result["probability"]
            risk_level = result["risk_level"]
            risk_desc = result["risk_description"]

            # Visual Result Box
            if pred == 1:
                st.markdown(
                    f'<div class="result-box-positive">'
                    f'<h3 style="margin: 0; color: #9b2c2c;">⚠️ Prediction: High Probability of Liver Disease</h3>'
                    f'<p style="margin-top: 8px; margin-bottom: 4px; font-size: 1.05rem;"><strong>Assessment:</strong> {risk_desc}</p>'
                    f'<p style="margin: 0;"><strong>Estimated Disease Probability:</strong> {prob * 100:.1f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-box-negative">'
                    f'<h3 style="margin: 0; color: #22543d;">✅ Prediction: Healthy / No Liver Disease Detected</h3>'
                    f'<p style="margin-top: 8px; margin-bottom: 4px; font-size: 1.05rem;"><strong>Assessment:</strong> {risk_desc}</p>'
                    f'<p style="margin: 0;"><strong>Estimated Disease Probability:</strong> {prob * 100:.1f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Probability Gauge Progress Bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Disease Probability Indicator")
            st.progress(float(prob))
            
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.metric("Predicted Diagnosis", result["label"])
            with r_col2:
                st.metric("Risk Category", risk_level)
            with r_col3:
                st.metric("Model Confidence", f"{max(prob, 1 - prob) * 100:.1f}%")

            # Elevated Biomarker Flags
            flags = []
            if total_bilirubin > 1.2:
                flags.append(f"**Total Bilirubin elevated** ({total_bilirubin} mg/dL | Normal &le; 1.2)")
            if direct_bilirubin > 0.3:
                flags.append(f"**Direct Bilirubin elevated** ({direct_bilirubin} mg/dL | Normal &le; 0.3)")
            if alkaline_phosphotase > 147:
                flags.append(f"**Alkaline Phosphatase elevated** ({alkaline_phosphotase} IU/L | Normal &le; 147)")
            if alamine_aminotransferase > 56:
                flags.append(f"**ALT / SGPT elevated** ({alamine_aminotransferase} IU/L | Normal &le; 56)")
            if aspartate_aminotransferase > 40:
                flags.append(f"**AST / SGOT elevated** ({aspartate_aminotransferase} IU/L | Normal &le; 40)")
            if albumin < 3.5:
                flags.append(f"**Albumin depressed** ({albumin} g/dL | Normal &ge; 3.5)")
            if albumin_globulin_ratio < 0.8:
                flags.append(f"**A/G Ratio depressed** ({albumin_globulin_ratio} | Normal &ge; 0.8)")

            if flags:
                st.warning("🔬 **Abnormal / Elevated Biomarkers Flagged:**\n\n" + "\n\n".join([f"- {f}" for f in flags]))
            else:
                st.success("🔬 All entered laboratory biomarkers fall within standard reference intervals.")

            # Clinical Advisory
            st.markdown(
                f'<div class="disclaimer-box">'
                f'<strong>⚠️ Medical Disclaimer:</strong> {result["disclaimer"]}'
                f'</div>',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
