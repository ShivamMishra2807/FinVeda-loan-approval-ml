"""
End-to-End Prediction Inference Tests.
Validates loading trained model artifacts, predicting approval and rejection cases,
probability confidence intervals, and schema compliance.
"""

import os
import pytest
from src.prediction import LoanPredictionService

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "voting_ensemble_pipeline.joblib")


def test_ensemble_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Trained ensemble model artifact must exist."


def test_positive_loan_prediction():
    service = LoanPredictionService(MODEL_PATH)
    applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 8500,
        "CoapplicantIncome": 3500,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban"
    }
    res = service.predict(applicant)
    assert res["prediction_status"] == "Y"
    assert res["is_approved"] is True
    assert res["approval_confidence"] > 0.50
    assert "debt_to_income_pct" in res["financial_indicators"]


def test_negative_loan_prediction():
    service = LoanPredictionService(MODEL_PATH)
    # Poor credit history and excessive debt burden
    applicant = {
        "Gender": "Male",
        "Married": "No",
        "Dependents": "3+",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 1800,
        "CoapplicantIncome": 0,
        "LoanAmount": 380,
        "Loan_Amount_Term": 180,
        "Credit_History": 0.0,
        "Property_Area": "Rural"
    }
    res = service.predict(applicant)
    assert res["prediction_status"] == "N"
    assert res["is_approved"] is False
    assert res["rejection_confidence"] > 0.50
