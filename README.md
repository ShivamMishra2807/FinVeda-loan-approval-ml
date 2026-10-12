# FinVeda: Ensemble ML Based Bank Loan Approval Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Streamlit-1.65%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)

An academic, production-grade Machine Learning analytics platform and underwriting decision support system. Built with an institutional dark-mode fintech UI, leakage-free pipeline engineering with SMOTE oversampling, empirical benchmarking of 9 candidate classifiers, and a soft-voting ensemble (`Random Forest` + `Extra Trees` + `K-Nearest Neighbors`).

---
Live Demo:- https://finveda-loan-approval-ml-vjrgp5aedbdgwznx4d8qym.streamlit.app/
## 📑 Table of Contents
1. [Executive Summary](#executive-summary)
2. [Key Architecture & Data Pipeline](#key-architecture--data-pipeline)
3. [Empirical Experimental Results](#empirical-experimental-results)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Pipeline Execution Guide](#pipeline-execution-guide)
7. [Running the Streamlit Web Application](#running-the-streamlit-web-application)
8. [Google Colab Instructions](#google-colab-instructions)
9. [Automated Testing](#automated-testing)
10. [Academic Integrity & Ethical Disclaimer](#academic-integrity--ethical-disclaimer)

---

## Executive Summary

The **ApexCredit Ensemble ML System** addresses risk mitigation in credit underwriting by synthesizing historical applicant characteristics, financial ratios, and credit history indicators to forecast loan approval likelihood (`Y` / `N`).

### Primary Objectives:
- **Leakage-Free Feature Engineering**: Computation of domain-specific debt burden metrics (`Debt_Burden`, `Loan_Income_Ratio`, `Total_Income`).
- **Resampling Without Bias**: Integrated `imblearn.pipeline.Pipeline` ensuring `SMOTE` is fitted strictly within training cross-validation folds.
- **Nine-Classifier Empirical Benchmark**: Direct comparative evaluation of 9 baseline algorithms on an identical held-out test set ($N=1,040$).
- **Soft-Voting Ensemble**: Variance-reduced meta-classifier uniting Random Forest, Extra Trees, and KNN.
- **Production Fintech UI/UX**: 7-page interactive dashboard with applicant simulation, real-time probability calibration, and explainability heatmaps.

---

## Key Architecture & Data Pipeline

```
Raw Synthetic Data (5,250 records)
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
Serialized Model Artifacts (Joblib) & Streamlit Dashboard Interface
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

> **Academic Benchmark Reference**: The reference paper in the project specifications reported **87.26% accuracy** for its three-classifier voting ensemble. In our experiments, the soft-voting ensemble attained **91.25% accuracy** ($F_1 = 0.9459$, $\text{AUC} = 0.9238$) while providing calibrated probabilities.

---

## Project Structure

```
loan-approval-ml/
│
├── app.py                     # Streamlit 7-page interactive fintech dashboard
├── requirements.txt           # Dependency requirements specification
├── README.md                  # Comprehensive academic and technical documentation
├── LICENSE                    # MIT Open Source License
├── .gitignore                 # Cache and environment exclusion rules
│
├── data/
│   └── loan_dataset.csv       # Synthetic academic loan applications (5,250 rows)
│
├── notebooks/
│   └── loan_approval_prediction.ipynb # Google Colab & local Jupyter notebook
│
├── src/
│   ├── generate_dataset.py    # Probabilistic underwriting dataset synthesizer
│   ├── preprocessing.py       # Cleaning, domain ratios, ColumnTransformers & SMOTE
│   ├── models.py              # 9 classifiers + Voting ensemble training engine
│   ├── prediction.py          # Real-time inference service with schema validation
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
    ├── test_pipeline.py       # Data generation, cleaning, ratios, & split tests
    └── test_prediction.py     # End-to-end model inference and validation tests
```

---

## Installation & Setup

### 1. Clone or Open Workspace
```bash
cd loan-approval-ml
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

---

## Pipeline Execution Guide

Run each phase of the machine learning pipeline from the command line:

### Step 1: Synthesize Academic Dataset
```bash
python -m src.generate_dataset
```
*Generates `data/loan_dataset.csv` with 5,250 records and controlled missingness/duplicates.*

### Step 2: Train and Benchmark All Classifiers
```bash
python -m src.models
```
*Executes 5-fold cross-validation on 9 classifiers + Voting Ensemble, evaluates held-out test performance, and serializes artifacts into `models/` and `reports/`.*

### Step 3: Generate Visual Artifacts
```bash
python -m src.evaluation
```
*Creates static figures in `visualizations/`.*

---

## Running the Streamlit Web Application

Launch the 7-page interactive dashboard:
```bash
streamlit run app.py
```

### Dashboard Features:
1. **Executive Dashboard**: Key performance indicator cards (Approval Rate, Median Loan, Total Volume in ₹) and portfolio distributions.
2. **Loan Prediction Studio**: Real-time form with inline validation, financial ratio calculation (DTI, EMI in ₹, LTI), and probability gauge.
3. **Exploratory Data Analysis**: Histograms, violin distributions, and Spearman correlation matrices.
4. **Model Performance Lab**: Interactive model leaderboard, per-classifier confusion matrices, and cross-validation summaries.
5. **Model Explainability**: Gini feature importance rankings from tree ensembles.
6. **Dataset Explorer**: Searchable, filterable table with one-click CSV export.
7. **About & Methodology**: Architectural data-flow diagrams and research citations.

---

## 🚀 GitHub Push & Streamlit Cloud Deployment Guide

### 1. Push to Your GitHub Repository
1. Open your browser and create a new repository on [GitHub](https://github.com/new) named: `loan-approval-ml` (or your choice).
2. Link your local repository and push:
```powershell
# Open terminal in project directory
cd C:\Users\shivam\.gemini\antigravity\scratch\loan-approval-ml

# Rename branch to main
git branch -M main

# Add your GitHub remote URL (replace <YOUR_USERNAME> with your GitHub username)
git remote add origin https://github.com/<YOUR_USERNAME>/loan-approval-ml.git

# Push code to GitHub
git push -u origin main
```

### 2. Deploy Instantly on Streamlit Community Cloud (Free)
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Fill in:
   - **Repository**: `<YOUR_USERNAME>/loan-approval-ml`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**
5. Streamlit Cloud automatically installs packages from `requirements.txt` and loads the saved `models/voting_ensemble_pipeline.joblib`. Your app will be live with a public URL!

---

## Google Colab Instructions

The notebook `notebooks/loan_approval_prediction.ipynb` is designed to run self-contained in Google Colab:

1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `notebooks/loan_approval_prediction.ipynb`.
3. If running without the local repository, the included **Academic Fallback Generator** cell will automatically synthesize the dataset.
4. Run all cells sequentially via `Runtime -> Run all`.

---

## Automated Testing

Execute the test suite with verbose reporting:
```bash
python -m pytest tests/ -v
```

All 9 test assertions verify:
- Synthetic dataset generation schema and constraints
- Deduplication and string whitespace stripping
- Domain financial ratios (`Debt_Burden`, `Loan_Income_Ratio`)
- Imputation and scaling without data leakage
- Stratified sampling balance
- Out-of-bounds input rejection
- Joblib model persistence and inference

---

## Academic Integrity & Ethical Disclaimer

1. **Synthetic Data**: This platform uses an academic synthetic dataset. No real customer banking records or personally identifiable financial data (PII) were utilized.
2. **Decision Support Only**: This platform is an educational prototype and cannot be utilized as an autonomous credit approval system.
3. **Fair Lending Notice**: Real-world underwriting must adhere to statutory requirements such as the Equal Credit Opportunity Act (ECOA) and the Fair Credit Reporting Act (FCRA), including algorithmic bias auditing across protected classes.
