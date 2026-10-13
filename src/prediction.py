"""
Prediction and Inference Service.
Loads the serialized production Voting Ensemble pipeline, validates input schemas,
applies feature engineering, generates probability calibrations, and formats results.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.preprocessing import (
    add_engineered_features,
    BASE_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    REVERSE_TARGET_MAP
)
from src.financial import calculate_emi, calculate_lti
from src.pricing import determine_interest_rate


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
        
    def predict(
        self,
        input_data: Dict[str, Any],
        annual_interest_rate: Optional[float] = None,
        existing_monthly_debt: float = 0.0,
        loan_product: Optional[str] = None,
        is_simulation_mode: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Accepts dictionary of raw applicant inputs, applies domain feature engineering,
        generates model prediction and calibrated probabilities, and calculates
        financially verified reducing-balance EMI, DTI, and LTI metrics using
        institutional pricing rules.
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
        
        # ML Inference using original preprocessor schema
        df_input = pd.DataFrame([record])
        df_engineered = add_engineered_features(df_input)
        
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
            
        # Institutional Rate Determination
        product = loan_product or input_data.get("Loan_Category") or "Personal Loan"
        existing_debt = float(input_data.get("Existing_Monthly_Debt", existing_monthly_debt))
        total_monthly_income = record["ApplicantIncome"] + record["CoapplicantIncome"]
        pre_dti = (existing_debt / total_monthly_income * 100.0) if total_monthly_income > 0 else 0.0

        if annual_interest_rate is not None and is_simulation_mode:
            pricing_res = determine_interest_rate(
                loan_product=product,
                override_rate=annual_interest_rate,
                is_simulation_mode=True
            )
        elif annual_interest_rate is not None and not is_simulation_mode:
            # Explicit rate provided (e.g. from upstream pricing module)
            pricing_res = {
                "applicable_annual_rate": float(annual_interest_rate),
                "base_rate": float(annual_interest_rate),
                "risk_adjustments": [],
                "is_simulation": False,
                "pricing_source": f"Institutional Pricing Rules ({product})",
                "pricing_rationale": "Applicable rate evaluated via institutional product rules.",
                "disclaimer": "Illustrative demonstration pricing rule."
            }
        else:
            # Standard automated applicant pricing workflow
            pricing_res = determine_interest_rate(
                loan_product=product,
                credit_history=record["Credit_History"],
                is_self_employed=(record["Self_Employed"] == "Yes"),
                existing_dti_pct=pre_dti,
                is_simulation_mode=False
            )

        rate = pricing_res["applicable_annual_rate"]
        principal_inr = record["LoanAmount"] * 1000.0
        tenure_months = int(record["Loan_Amount_Term"])
        
        emi_calc = calculate_emi(
            principal=principal_inr,
            annual_interest_rate=rate,
            tenure_months=tenure_months
        )
        proposed_emi = emi_calc["monthly_emi"]
        total_monthly_debt = existing_debt + proposed_emi
        
        if total_monthly_income > 0:
            dti_pct = round((total_monthly_debt / total_monthly_income) * 100.0, 2)
        else:
            dti_pct = 100.0 if total_monthly_debt > 0 else 0.0
            
        annual_gross_income = total_monthly_income * 12.0
        lti_ratio = calculate_lti(principal_inr, annual_gross_income)
        monthly_loan_to_income = round(principal_inr / max(total_monthly_income, 1.0), 2)

        
        # Transparent factor explanations based on underwriting parameters
        explanations = []
        suggestions = []
        
        if record["Credit_History"] == 1.0:
            explanations.append("Positive credit track record meets benchmark underwriting criteria.")
        else:
            explanations.append("Derogatory or missing credit history serves as a primary risk flag.")
            suggestions.append("Improve CIBIL track record by resolving delinquent dues before re-applying.")
            
        if dti_pct > 50.0:
            explanations.append(f"Total post-loan debt service ({dti_pct:.1f}%) exceeds the 50% prudential guideline.")
            suggestions.append("Consider opting for a longer repayment tenure or reducing requested loan amount to lower monthly debt service.")
        elif dti_pct <= 35.0:
            explanations.append(f"Healthy debt-to-income ratio ({dti_pct:.1f}%) demonstrates comfortable repayment buffer.")
            
        if lti_ratio > 4.5:
            explanations.append(f"Loan-to-income ratio ({lti_ratio:.2f}x annual income) is high relative to household earnings.")
            suggestions.append("Adding a co-applicant with verifiable independent income can enhance borrowing capacity.")
            
        if not suggestions:
            suggestions.append("Maintain timely debt service and preserve liquidity reserves to sustain strong borrowing eligibility.")

        return {
            "prediction_status": status,
            "is_approved": status == "Y",
            "approval_confidence": approval_confidence,
            "rejection_confidence": rejection_confidence,
            "probabilities": probabilities,
            "financial_indicators": {
                "principal_inr": principal_inr,
                "annual_interest_rate": rate,
                "pricing_source": pricing_res.get("pricing_source", "Institutional Pricing Rules"),
                "pricing_rationale": pricing_res.get("pricing_rationale", ""),
                "is_simulation": pricing_res.get("is_simulation", False),
                "risk_adjustments": pricing_res.get("risk_adjustments", []),
                "pricing_disclaimer": pricing_res.get("disclaimer", ""),
                "tenure_months": tenure_months,
                "total_income": total_monthly_income,
                "monthly_income": total_monthly_income,
                "existing_monthly_debt": existing_debt,
                "monthly_installment": proposed_emi,
                "proposed_emi": proposed_emi,
                "total_monthly_debt": round(total_monthly_debt, 2),
                "total_interest": emi_calc["total_interest"],
                "total_repayment": emi_calc["total_repayment"],
                "debt_to_income_pct": dti_pct,
                "loan_to_income_ratio": lti_ratio,
                "loan_to_monthly_income": monthly_loan_to_income,
                "repayment_method": "Reducing Balance Amortization"
            },

            "explanations": explanations,
            "educational_suggestions": suggestions,
            "model_used": "Soft Voting Ensemble (Random Forest + Extra Trees + KNN)",
            "limitations_note": "This prediction is generated by a synthetic ML model prototype for academic evaluation and does not constitute a guaranteed lending offer or regulatory credit score."
        }


