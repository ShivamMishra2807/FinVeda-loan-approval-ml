# FinVeda: AI-Powered Loan Approval & Financial Planning Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Streamlit-1.65%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)

**FinVeda** is a production-grade Machine Learning credit underwriting, financial planning, and debt analysis platform. Built with an institutional dark-mode fintech UI in Indian Rupees (₹ INR), reducing-balance EMI calculation engine, multi-scenario loan comparison, month-by-month amortization schedule, prudential cash-flow affordability analysis, and an ensemble machine learning decision engine (`Random Forest` + `Extra Trees` + `K-Nearest Neighbors`).

---

## 📑 Table of Contents
1. [Core Features & Upgrades](#core-features--upgrades)
2. [Verified Financial Formulation & Assumptions](#verified-financial-formulation--assumptions)
3. [Configurable Loan Categories & Defaults](#configurable-loan-categories--defaults)
4. [Machine Learning Underwriting Architecture](#machine-learning-underwriting-architecture)
5. [Empirical Experimental Results](#empirical-experimental-results)
6. [Project Structure](#project-structure)
7. [Installation & Windows Launch Instructions](#installation--windows-launch-instructions)
8. [Automated Test Suite](#automated-test-suite)
9. [GitHub & Streamlit Community Cloud Deployment](#github--streamlit-community-cloud-deployment)
10. [Academic Integrity & Ethical Disclaimer](#academic-integrity--ethical-disclaimer)

---

## Core Features & Upgrades

1. **Executive Dashboard**: Real-time loan portfolio metrics (sanction rate, median loan, median income in ₹ INR) and interactive demographic distributions.
2. **Loan Approval Prediction**: Real-time evaluation powered by the saved Soft-Voting Ensemble (`Random Forest` + `Extra Trees` + `KNN`). Provides calibrated probabilities, transparent factor attribution (credit track record, DTI, LTI), and actionable educational suggestions.
3. **Interactive EMI Calculator**: Implements the reducing-balance EMI formula with zero-interest edge-case protection, breakdown donut chart, and fee-inclusive borrowing costs.
4. **Loan Scenario Comparison**: Side-by-side trade-off analysis of up to 3 loan packages. Automatically isolates lowest monthly outflow (lowest EMI) vs lowest lifetime cost (least interest).
5. **Monthly Amortization Repayment Schedule**: Dynamic month-by-month ledger detailing opening balance, principal portion, interest portion, and closing balance tapering exactly to ₹0.00, with one-click CSV export.
6. **Loan Affordability Analysis**: Cash-flow health diagnostics assessing debt-to-income (DTI) ratio against prudential guidelines (<35% low risk, 35-50% moderate, >50% stretched).

---

## Verified Financial Formulation & Assumptions

### Equated Monthly Instalment (EMI) Formula
For reducing-balance amortizing credit facilities:

$$\text{EMI} = \frac{P \times r \times (1 + r)^n}{(1 + r)^n - 1}$$

Where:
- $P$ = Principal loan amount in ₹ INR
- $r$ = Monthly interest rate = $\frac{\text{Annual Interest Rate}}{12 \times 100}$
- $n$ = Repayment tenure in months

**Special Edge Case**: For zero-interest loans ($r = 0$), $\text{EMI} = \frac{P}{n}$.

**Benchmark Test Case Verification (Tests A & B)**:
- **Test Case A**:
  - Principal ($P$) = ₹5,00,000 | Annual Rate = $10.0\%$ | Tenure ($n$) = 5 years (60 months)
  - **Computed Monthly EMI** = **₹10,623.52** (approx. ₹10,624)
  - **Total Lifetime Interest** = **₹1,37,411.20** (approx. ₹1.37 Lakh)
  - **Total Repayment Amount** = **₹6,37,411.20** (approx. ₹6.37 Lakh)
- **Test Case B**:
  - Principal ($P$) = ₹3,00,000 | Annual Rate = $12.0\%$ | Tenure ($n$) = 12 months (1 year)
  - **Computed Monthly EMI** = **₹26,654.64**
  - **Total Lifetime Interest** = **₹19,855.68**
  - **Total Repayment Amount** = **₹3,19,855.68**
  - *Total repayment strictly equals Principal + Total Interest.*

---

## Configurable Loan Product Pricing Rules & Rate Determination

FinVeda implements an institutional **interest-rate determination workflow** (`src/pricing.py`):

1. **Normal Applicant Workflow**:
   - The applicant selects the loan product (`Personal Loan`, `Home Loan`, `Car Loan`, `Education Loan`, or `Business Loan`).
   - The applicable annual interest rate is **automatically system-determined** via documented pricing rules (incorporating base illustrative benchmarks and transparent credit risk loadings).
   - The rate is displayed as a **read-only value** with complete attribution and assumptions disclosed.
   - Retail applicants cannot arbitrarily self-assign lending rates.

2. **Underwriter Simulation Mode (Override Facility)**:
   - For credit risk officers and internal testing, an explicit Underwriter Simulation Mode allows testing rate sensitivity within regulatory product boundaries (`min_rate` to `max_rate`).
   - Any rate override is strictly labeled as a simulation.

| Loan Category | Base Benchmark Rate | Illustrative Range | Benchmark Tenure | Standard Fee | Risk Loading Rules |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Personal Loan** | 12.00% | 10.5% – 24.0% | 3 Years (36 Mos) | 1.50% | Adverse credit loading: +3.5% |
| **Home Loan** | 8.50% | 7.5% – 11.0% | 20 Years (240 Mos) | 0.50% | Adverse credit loading: +1.5% |
| **Car Loan** | 9.50% | 8.0% – 15.0% | 5 Years (60 Mos) | 1.00% | Adverse credit loading: +2.0% |
| **Education Loan** | 9.50% | 8.0% – 14.0% | 7 Years (84 Mos) | 0.75% | Co-borrower credit loading: +2.0% |
| **Business Loan** | 14.00% | 11.0% – 24.0% | 4 Years (48 Mos) | 2.00% | Adverse credit: +3.0%; Self-employed: +0.5% |

*Disclaimer: Rates and spreads are demonstration benchmark guidelines based on Indian retail banking standards, not binding lender contracts.*



---

## Machine Learning Underwriting Architecture

```
Raw Synthetic Data (5,250 records in ₹ INR)
             │
             ▼
Clean Raw Data (Deduplication & Schema Enforcement)
             │
             ▼
Domain Feature Engineering (Total Income, Debt Burden, LTI)
             │
             ▼
Stratified Train-Test Split (80% Train, 20% Held-out Test)
             │
             ▼
Pipeline Preprocessor (Median Imputer, Standard Scaler, OneHotEncoder)
             │
             ▼
SMOTE Oversampling (Applied Strictly to Training Folds)
             │
             ▼
Model Training & 5-Fold Stratified Cross-Validation
             │
             ▼
Soft-Voting Ensemble (Random Forest + Extra Trees + KNN)
             │
             ▼
Serialized Model Artifacts (Joblib) & Streamlit Platform Interface
```

---

## Empirical Experimental Results

Evaluated on the held-out test split ($N = 1,040$ applications, Stratified 80/20):

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Accuracy | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees** | **91.83%** | 96.16% | 93.80% | 0.9497 | 0.9249 | 93.17% (±0.81%) | 0.677s |
| **Decision Tree** | 91.83% | 96.27% | 93.68% | 0.9496 | 0.9118 | 92.72% (±1.02%) | 0.166s |
| **AdaBoost** | 91.83% | 96.16% | 93.80% | 0.9497 | 0.9127 | 93.34% (±0.87%) | 1.660s |
| **Logistic Regression** | 91.63% | 96.38% | 93.33% | 0.9483 | 0.9212 | 93.17% (±0.78%) | 0.161s |
| **Random Forest** | 91.63% | 96.15% | 93.57% | 0.9484 | 0.9257 | 93.17% (±0.80%) | 0.976s |
| **Gradient Boosting** | 91.44% | 95.81% | 93.68% | 0.9474 | 0.9148 | 92.98% (±0.88%) | 4.084s |
| **Support Vector Machine** | 91.35% | 96.14% | 93.22% | 0.9466 | 0.9015 | 93.05% (±0.99%) | 8.104s |
| **Voting Ensemble (RF+ET+KNN)** | 91.25% | 96.25% | 92.98% | 0.9459 | 0.9238 | 92.98% (±0.86%) | 1.508s |
| **Gaussian Naive Bayes** | 89.23% | 96.38% | 90.29% | 0.9324 | 0.9115 | 91.06% (±0.78%) | 0.086s |
| **K-Nearest Neighbors** | 87.50% | 96.77% | 87.72% | 0.9202 | 0.8994 | 89.21% (±0.68%) | 0.062s |

---

## Project Structure

```
loan-approval-ml/
│
├── app.py                     # Streamlit 6-module interactive financial application
├── requirements.txt           # Pinned dependencies matching serialized Joblib models
├── README.md                  # Comprehensive technical and deployment documentation
├── LICENSE                    # MIT Open Source License
├── .gitignore                 # Exclusion rules for caches and virtual environments
│
├── data/
│   └── loan_dataset.csv       # Synthetic academic loan applications (5,250 records in ₹ INR)
│
├── notebooks/
│   └── loan_approval_prediction.ipynb # Google Colab & local Jupyter notebook
│
├── src/
│   ├── financial.py           # Verified EMI, amortization, affordability, & comparison logic
│   ├── prediction.py          # ML inference service, factor attribution, & suggestions
│   ├── preprocessing.py       # Domain ratios (DTI, LTI), ColumnTransformers & SMOTE
│   ├── models.py              # 9 classifiers + Voting ensemble training engine
│   ├── generate_dataset.py    # Probabilistic underwriting dataset synthesizer
│   └── evaluation.py          # Metrics generation and matplotlib/seaborn charts
│
├── models/
│   ├── voting_ensemble_pipeline.joblib # Serialized production Voting Ensemble
│   ├── random_forest_pipeline.joblib   # Serialized tree explainability pipeline
│   └── test_splits.joblib              # Preserved evaluation test splits
│
├── reports/
│   ├── model_metrics.csv      # Formatted CSV model comparison leaderboard
│   ├── classification_reports.json # Full precision/recall/F1 metrics per model
│   └── dataset_summary.json   # High-level dataset metadata and metrics
│
├── visualizations/
│   ├── model_comparison_bar.png      # Offline accuracy vs F1 benchmark plot
│   ├── ensemble_confusion_matrix.png # Held-out test confusion matrix
│   └── feature_importance.png        # Gini importance plot from Random Forest
│
└── tests/
    ├── test_financial.py      # EMI math, zero-rate, negative inputs, & amortization tests (9 tests)
    ├── test_pipeline.py       # Data generation, cleaning, ratios, & split tests (6 tests)
    └── test_prediction.py     # End-to-end model inference and validation tests (3 tests)
```

---

## Installation & Windows Launch Instructions

### 1. Open Terminal in Project Directory
```powershell
cd C:\Users\shivam\.gemini\antigravity\scratch\loan-approval-ml
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Automated Test Suite

Execute all 18 automated tests across financial calculations, preprocessing pipelines, and ML inference:

```powershell
python -m pytest tests/ -v
```

All 18 tests cover:
- Standard reducing-balance EMI calculations (PRD test case)
- Zero-interest and zero-principal edge cases
- Defensive exception handling on negative inputs
- Complete amortization schedule closure to ₹0.00
- Affordability cash-flow math and zero-income protection
- Multi-scenario comparison trade-off ordering
- Pipeline preprocessing and SMOTE leakage prevention
- End-to-end positive and negative ML inference

---

## GitHub & Streamlit Community Cloud Deployment

### Push to GitHub
```powershell
git add .
git commit -m "Upgrade FinVeda: Complete EMI calculator, loan comparison, amortization, and affordability"
git push origin main
```

### Deploy to Streamlit Community Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Select repository: `ShivamMishra2807/FinVeda-loan-approval-ml`
3. Branch: `main`
4. Main file path: `app.py`
5. Click **Deploy!**

---

## Academic Integrity & Ethical Disclaimer

1. **Synthetic Academic Data**: This platform uses an academic synthetic dataset. No real customer banking records or personally identifiable financial data (PII) were utilized.
2. **Decision Support Only**: This platform is an educational prototype and does not replace certified banking underwriting systems.
3. **Fair Lending Notice**: Real-world underwriting must adhere to statutory requirements such as RBI guidelines, Equal Credit Opportunity Act (ECOA), and Fair Credit Reporting Act (FCRA).
