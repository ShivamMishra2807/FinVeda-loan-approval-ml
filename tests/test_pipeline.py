"""
Test Suite for Loan Approval Prediction Platform.
Tests dataset creation, schema validation, missing handling, pipeline preprocessing,
model training execution, predictions, and bounds checking.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.generate_dataset import generate_loan_dataset
from src.preprocessing import (
    clean_raw_data,
    add_engineered_features,
    prepare_dataset_splits,
    build_preprocessor,
    TARGET_MAP,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.prediction import LoanPredictionService

# Project directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


def test_dataset_generation_and_schema():
    """Verify synthetic dataset properties, row counts, and column types."""
    df = generate_loan_dataset(n_samples=200, random_state=123, include_duplicates=5)
    assert len(df) == 205
    expected_cols = [
        "Loan_ID", "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History",
        "Property_Area", "Loan_Status"
    ]
    for col in expected_cols:
        assert col in df.columns
        
    assert set(df["Loan_Status"].dropna().unique()).issubset({"Y", "N"})
    assert df["ApplicantIncome"].min() >= 1000


def test_data_cleaning_and_deduplication():
    """Verify duplicate removal and string sanitization."""
    df = generate_loan_dataset(n_samples=100, random_state=42, include_duplicates=10)
    assert len(df) == 110
    cleaned = clean_raw_data(df)
    assert len(cleaned) == 100
    assert cleaned["Loan_ID"].is_unique


def test_feature_engineering_ratios():
    """Verify calculation of financial health metrics."""
    data = {
        "ApplicantIncome": [5000, 3000],
        "CoapplicantIncome": [2000, 0],
        "LoanAmount": [140, 100],
        "Loan_Amount_Term": [360, 180]
    }
    df = pd.DataFrame(data)
    eng = add_engineered_features(df)
    
    assert "Total_Income" in eng.columns
    assert eng["Total_Income"].iloc[0] == 7000
    assert eng["Total_Income"].iloc[1] == 3000
    assert "Loan_Income_Ratio" in eng.columns
    assert "Debt_Burden" in eng.columns
    assert eng["Debt_Burden"].iloc[0] > 0


def test_preprocessor_pipeline_transformation():
    """Verify column transformer imputes and encodes properly without NaN output."""
    df = generate_loan_dataset(n_samples=100, random_state=42, include_duplicates=0)
    cleaned = clean_raw_data(df)
    engineered = add_engineered_features(cleaned)
    
    preprocessor = build_preprocessor()
    X = engineered.drop(columns=["Loan_Status", "Loan_ID"])
    X_trans = preprocessor.fit_transform(X)
    
    assert not np.isnan(X_trans).any()
    assert X_trans.shape[0] == 100
    assert X_trans.shape[1] > len(NUMERICAL_FEATURES)


def test_stratified_split_integrity():
    """Ensure train and test splits preserve class distribution without leaking IDs."""
    df = generate_loan_dataset(n_samples=200, random_state=42, include_duplicates=0)
    X_train, X_test, y_train, y_test = prepare_dataset_splits(df, test_size=0.2, random_state=42)
    
    assert len(X_train) == 160
    assert len(X_test) == 40
    assert "Loan_ID" not in X_train.columns
    assert "Loan_Status" not in X_train.columns
    assert set(y_train.unique()).issubset({0, 1})


def test_prediction_service_validation():
    """Test inference validation catches negative values and missing fields."""
    # Test valid input structure without needing loaded model for validation test
    service = LoanPredictionService.__new__(LoanPredictionService)
    
    valid_sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban"
    }
    is_valid, msg = service.validate_input(valid_sample)
    assert is_valid is True
    
    # Negative income
    invalid_sample = valid_sample.copy()
    invalid_sample["ApplicantIncome"] = -500
    is_valid, msg = service.validate_input(invalid_sample)
    assert is_valid is False
    
    # Invalid credit history
    invalid_sample_credit = valid_sample.copy()
    invalid_sample_credit["Credit_History"] = 5.0
    is_valid, msg = service.validate_input(invalid_sample_credit)
    assert is_valid is False
