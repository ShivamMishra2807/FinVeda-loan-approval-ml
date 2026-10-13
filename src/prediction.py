"""
Prediction and Inference Service.
Loads the serialized production Voting Ensemble pipeline, validates input schemas,
applies feature engineering, generates probability calibrations, and formats results.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from src.preprocessing import (
    add_engineered_features,
    BASE_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    REVERSE_TARGET_MAP
)

class LoanPredictionService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default to models directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, ".."))
            model_path = os.path.join(project_root, "models", "voting_ensemble_pipeline.joblib")
            
        self.model_path = model_path
        self.pipeline = None
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model artifact not found at: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        
    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate presence and reasonable bounds of applicant input attributes."""
        required_fields = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]
        for field in required_fields:
            if field not in data or data[field] is None:
                return False, f"Missing required application field: {field}"
                
        # Numeric validation
        if data["ApplicantIncome"] < 0:
            return False, "Applicant Income cannot be negative."
        if data["CoapplicantIncome"] < 0:
            return False, "Co-applicant Income cannot be negative."
        if data["LoanAmount"] <= 0:
            return False, "Loan Amount must be greater than zero."
        if data["Loan_Amount_Term"] <= 0:
            return False, "Loan Amount Term must be greater than zero."
        if data["Credit_History"] not in [0.0, 1.0, 0, 1]:
            return False, "Credit History must be 0 (unfavorable) or 1 (meets guidelines)."
            
        return True, "Valid"
        
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts dictionary of raw applicant inputs, applies domain feature engineering,
        and generates model prediction and calibrated probabilities.
        """
        is_valid, msg = self.validate_input(input_data)
        if not is_valid:
            raise ValueError(f"Input validation error: {msg}")
            
        # Standardize Dependents format if passed as string '3+'
        dep_val = str(input_data["Dependents"]).strip()
        if dep_val == "3+":
            dep_val = "3"
            
        record = {
            "Gender": str(input_data["Gender"]),
            "Married": str(input_data["Married"]),
            "Dependents": dep_val,
            "Education": str(input_data["Education"]),
            "Self_Employed": str(input_data["Self_Employed"]),
            "ApplicantIncome": float(input_data["ApplicantIncome"]),
            "CoapplicantIncome": float(input_data["CoapplicantIncome"]),
            "LoanAmount": float(input_data["LoanAmount"]),
            "Loan_Amount_Term": float(input_data["Loan_Amount_Term"]),
            "Credit_History": float(input_data["Credit_History"]),
            "Property_Area": str(input_data["Property_Area"])
        }
        
        df_input = pd.DataFrame([record])
        df_engineered = add_engineered_features(df_input)
        
        # Inference
        pred_label_idx = self.pipeline.predict(df_engineered)[0]
        status = REVERSE_TARGET_MAP.get(pred_label_idx, "Unknown")
        
        probabilities = None
        approval_confidence = 0.0
        rejection_confidence = 0.0
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df_engineered)[0]
            rejection_confidence = float(probs[0])
            approval_confidence = float(probs[1])
            probabilities = {
                "Approved": round(approval_confidence, 4),
                "Rejected": round(rejection_confidence, 4)
            }
            
        # Key Financial Indicators
        tot_income = record["ApplicantIncome"] + record["CoapplicantIncome"]
        monthly_installment = (record["LoanAmount"] * 1000) / max(record["Loan_Amount_Term"], 1)
        debt_to_income = (monthly_installment / max(tot_income, 1)) * 100.0
        
        # Transparent factor explanations based on underwriting parameters
        explanations = []
        suggestions = []
        
        if record["Credit_History"] == 1.0:
            explanations.append("Positive credit track record meets benchmark underwriting criteria.")
        else:
            explanations.append("Derogatory or missing credit history serves as a major risk barrier.")
            suggestions.append("Improve CIBIL track record by resolving delinquent dues before re-applying.")
            
        if debt_to_income > 50.0:
            explanations.append(f"High debt-to-income ratio ({debt_to_income:.1f}%) exceeds the 50% prudential guideline.")
            suggestions.append("Consider opting for a longer repayment tenure or reducing requested loan amount to lower monthly burden.")
        elif debt_to_income <= 35.0:
            explanations.append(f"Healthy debt-to-income ratio ({debt_to_income:.1f}%) indicates adequate cash-flow bandwidth.")
            
        loan_inc_ratio = (record["LoanAmount"] * 1000) / max(tot_income, 1)
        if loan_inc_ratio > 30.0:
            explanations.append(f"Requested loan amount is high relative to monthly household income ({loan_inc_ratio:.1f}x).")
            suggestions.append("Adding a co-applicant with independent verifiable income can strengthen application leverage.")
            
        if not suggestions:
            suggestions.append("Maintain timely debt service and preserve liquidity reserves to sustain approval eligibility.")

        return {
            "prediction_status": status,
            "is_approved": status == "Y",
            "approval_confidence": approval_confidence,
            "rejection_confidence": rejection_confidence,
            "probabilities": probabilities,
            "financial_indicators": {
                "total_income": tot_income,
                "monthly_installment": round(monthly_installment, 2),
                "debt_to_income_pct": round(debt_to_income, 2),
                "loan_to_income_ratio": round((record["LoanAmount"] * 1000) / max(tot_income, 1), 2)
            },
            "explanations": explanations,
            "educational_suggestions": suggestions,
            "model_used": "Soft Voting Ensemble (Random Forest + Extra Trees + KNN)",
            "limitations_note": "This prediction is generated by a synthetic ML model prototype for academic evaluation and does not constitute a guaranteed lending offer or regulatory credit score."
        }

