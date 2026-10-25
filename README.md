# 🩺 HepatoScan: End-to-End Liver Disease Prediction System

A complete, production-grade Machine Learning system and interactive web application for clinical liver disease risk assessment, built using the Indian Liver Patient Dataset (LPD).

Developed from the perspective of an ML student focusing on rigorous data science principles: preventing data leakage, addressing class imbalance using SMOTE within cross-validated pipelines, evaluating models using clinical screening metrics (Recall, F1-Score, ROC-AUC), and deploying an interactive Streamlit application with input validation and clinical disclaimers.

---

## 📌 1. Project Overview

Liver diseases are among the top causes of mortality globally, frequently presenting asymptomatically until hepatic impairment is advanced. Routine Liver Function Tests (LFTs) measure circulating hepatic enzymes and proteins.

This project delivers:
1. **Automated Dataset Inspection & Sanitization:** Detects raw encodings, handles non-breaking space characters (`\xa0`), deduplicates records, and sanitizes feature schema.
2. **Leakage-Free Preprocessing Pipeline:** Uses a `scikit-learn` `ColumnTransformer` (median imputation for skewed enzyme outliers, standard scaling, and one-hot encoding).
3. **Imbalanced Learning with SMOTE:** Overcomes class imbalance using `imblearn.pipeline.Pipeline`, applying SMOTE strictly on training splits.
4. **Multi-Model Benchmark:** Compares **Logistic Regression**, **Decision Tree**, and **Random Forest** across 6 clinical metrics.
5. **Model Persistence & Robust Inference:** Serializes the complete fitted pipeline via `Joblib` and exposes validated prediction functions.
6. **Responsive Web Application:** Built using `Streamlit` with interactive dashboards, biomarker sliders, reference intervals, sample patient profiles, and medical disclaimers.
7. **Automated Testing Suite:** 13 unit tests via `pytest` validating preprocessing, models, and boundaries.

---

## 📊 2. Dataset Description

- **Source Dataset:** `Liver Patient Dataset (LPD)_train.csv`
- **Total Initial Records:** 30,691
- **Deduplicated Records:** 19,368 unique patient profiles (preventing duplicate leakage between train and test sets)
- **Features:** 10 features (9 Numerical Biomarkers, 1 Categorical Demographic)
- **Target Variable (`Result`):**
  - Original encoding: `1` = Liver Disease Patient (71.3%), `2` = Healthy Patient (28.7%)
  - Cleaned ML encoding: `1` = Positive (Liver Disease), `0` = Negative (Healthy / No Disease)

### Biomarker Reference Table

| Feature Name | Clinical Name | Unit | Normal Clinical Range |
| :--- | :--- | :--- | :--- |
| `age` | Age of the Patient | Years | 1 – 100 |
| `gender` | Biological Sex | Categorical | Male / Female |
| `total_bilirubin` | Total Bilirubin | mg/dL | 0.1 – 1.2 |
| `direct_bilirubin` | Direct (Conjugated) Bilirubin | mg/dL | 0.0 – 0.3 |
| `alkaline_phosphotase` | Alkaline Phosphatase (ALP) | IU/L | 44 – 147 |
| `alamine_aminotransferase` | Alamine Aminotransferase (ALT/SGPT) | IU/L | 7 – 56 |
| `aspartate_aminotransferase`| Aspartate Aminotransferase (AST/SGOT) | IU/L | 10 – 40 |
| `total_proteins` | Total Serum Proteins | g/dL | 6.0 – 8.3 |
| `albumin` | Serum Albumin | g/dL | 3.5 – 5.5 |
| `albumin_globulin_ratio` | Albumin to Globulin (A/G) Ratio | Ratio | 0.8 – 2.0 |

---

## ⚙️ 3. Machine Learning Workflow

```
┌────────────────────────────────────────────────────────┐
│               Raw CSV Dataset (LPD)                    │
│    (30,691 rows, latin1 encoding, non-breaking spaces) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│         Phase 1: Inspection & Standardization          │
│    - Standardize column names (strip \xa0)             │
│    - Drop synthetic duplicates (19,368 unique rows)    │
│    - Map target: 1 -> Liver Disease (1), 2 -> Healthy (0)
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│     Phase 2: Stratified Split & Preprocessor           │
│    - Train (80% = 15,494), Test (20% = 3,874)          │
│    - Numerical: SimpleImputer(median) + StandardScaler │
│    - Categorical: SimpleImputer(mode) + OneHotEncoder  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│   Phase 3: Pipeline Model Training (SMOTE inside)      │
│    - SMOTE applied ONLY during training folds          │
│    - Logistic Regression vs Decision Tree vs RF        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        Phase 4: Evaluation & Model Selection           │
│    - Clinical composite: 0.40*Recall + 0.35*F1 + 0.25*AUC│
│    - Winner: Random Forest (Composite Score = 0.9769)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Phase 5: Export & Deployment               │
│    - Joblib pipeline export (`best_liver_model.joblib`)│
│    - Reusable inference engine (`src/predict.py`)      │
│    - Streamlit 6-section UI (`app.py`)                 │
└────────────────────────────────────────────────────────┘
```

---

## 🏆 4. Model Evaluation & Benchmark Results

All models were evaluated on the **unseen held-out test cohort of 3,874 patients**:

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Specificity | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 64.74% | 91.40% | 55.79% | 0.6929 | 0.7645 | 86.96% | 0.6568 |
| **Decision Tree** | 84.77% | 92.03% | 86.10% | 0.8896 | 0.9200 | 81.47% | 0.8858 |
| **Random Forest (Final)** | **97.08%** | **99.77%** | **96.13%** | **0.9792** | **0.9988** | **99.46%** | **0.9769** |

### Why Random Forest Was Selected
In clinical screening, **Recall (Sensitivity)** is paramount: missing a sick patient (False Negative) can result in untreated liver degradation.
- **Random Forest achieved 96.13% Recall**, catching 2,655 out of 2,762 disease cases.
- **Low False Positives (6)** with a 99.46% Specificity, preventing unnecessary anxiety.
- **ROC-AUC of 0.9988** guarantees high confidence and reliable probability ranking.

---

## 📁 5. Project Directory Structure

```
liver disease pred/
├── dataset/
│   ├── liver_disease_data.csv            # Cleaned/copied project dataset
│   └── Liver Patient Dataset (LPD)_train.csv
├── models/
│   ├── figures/                          # Generated high-resolution plots
│   │   ├── best_model_cm.png
│   │   ├── confusion_matrices.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_distributions.png
│   │   ├── models_comparison.png
│   │   ├── roc_curves.png
│   │   └── target_distribution.png
│   ├── all_models_summary.csv            # Benchmark metrics table
│   ├── best_liver_model.joblib           # Trained, serialized scikit-learn pipeline
│   └── model_metadata.json               # Model metrics and class mappings
├── notebooks/
│   └── exploratory_analysis.ipynb        # Jupyter notebook walk-through
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py             # Preprocessing & ColumnTransformer logic
│   ├── dataset_inspector.py              # Phase 1 dataset discovery script
│   ├── eda_visualizer.py                 # Phase 5 EDA chart generator
│   ├── model_training.py                 # Model training, comparison, and saving
│   └── predict.py                        # Reusable inference functions with validation
├── tests/
│   ├── __init__.py
│   ├── test_app.py                       # App file and artifact existence tests
│   ├── test_model.py                     # Inference and input validation tests
│   └── test_preprocessing.py             # Cleaning and transformation tests
├── .gitignore                            # Version control exclusions
├── app.py                                # Full 6-section Streamlit Web Application
├── requirements.txt                      # Project dependency specification
└── README.md                             # Comprehensive project documentation
```

---

## 💻 6. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (Developed and validated on Python 3.12.10)
- Operating System: Windows, macOS, or Linux

### Step 1: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Training Pipeline
To inspect the dataset, train candidate models, compare them, and save the best pipeline:
```bash
python src/model_training.py
```

### Step 4: Generate All EDA Visualizations
```bash
python src/eda_visualizer.py
```

### Step 5: Run Automated Tests
Execute the unit testing suite to verify system integrity:
```bash
pytest -v tests
```
*All 13 tests should pass.*

### Step 6: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## 🌐 7. Streamlit Web Application Sections

The web application features 6 responsive sections:
1. **🏠 Home Dashboard:** Overview metrics, motivation, and clinical significance of LFT biomarkers.
2. **📊 Dataset Overview:** Interactive table, descriptive statistics, schema, and preprocessing details.
3. **🔬 Exploratory Data Analysis:** Target class balance charts, correlation heatmaps, and enzyme distributions.
4. **🏆 Model Comparison:** Metrics benchmark table, multi-bar chart, confusion matrices, and ROC curves.
5. **🧪 Prediction Tool:** Interactive patient form with quick-fill presets, validation checks, disease probability bar, biomarker flags, and clinical advisory.
6. **📖 About Project:** Technical design details, SMOTE pipeline architecture, and methodology.

---

## ⚠️ 8. Clinical Limitations & Future Scope

- **Non-Invasive Diagnostic Tool:** This model provides risk stratification based on laboratory values. It is not an alternative to liver biopsy or clinical diagnosis.
- **External Cohort Validation:** The model should be validated on multi-center international health databases to assess generalizability across demographics.
- **Serology & Imaging:** Integrating viral markers (Hepatitis B & C) and ultrasound elastography (FibroScan) will enhance diagnostic specificity.
#   L I V E R - D I S E A S E - P R E D I C T I O N  
 #   L I V E R - D I S E A S E - P R E D I C T I O N  
 