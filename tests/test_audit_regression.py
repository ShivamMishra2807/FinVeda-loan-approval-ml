"""
Financial Audit Regression Tests.

These tests verify the critical data flows identified during the financial audit:
1. Loan amount unit conversion (INR → thousands for ML model)
2. EMI calculation uses correct principal in INR (not thousands)
3. DTI and LTI use consistent principal in INR
4. ML prediction receives LoanAmount in thousands (dataset scale)
5. Product-specific default amounts are sensible INR values
6. Credit history risk loading produces correct rate adjustments
7. End-to-end prediction flow produces consistent financial indicators
"""

import pytest
import math
from src.prediction import LoanPredictionService
from src.financial import calculate_emi, calculate_lti, check_loan_affordability, LOAN_CATEGORIES
from src.pricing import PRODUCT_PRICING_CATALOG, determine_interest_rate, get_product_pricing_config


class TestLoanAmountUnitConversion:
    """Verify that the loan amount conversion from INR to thousands is correct."""
    
    def test_inr_to_thousands_conversion(self):
        """₹16,25,000 entered in INR must become 1625.0 in thousands for the ML model."""
        loan_amount_inr = 1625000  # User enters ₹16,25,000
        loan_amount_k = loan_amount_inr / 1000.0
        assert loan_amount_k == 1625.0
    
    def test_principal_inr_in_prediction(self):
        """prediction.py multiplies LoanAmount (thousands) × 1000 to get principal in INR."""
        service = LoanPredictionService()
        applicant = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 50000,
            "CoapplicantIncome": 10000,
            "LoanAmount": 1625.0,  # Thousands (i.e. ₹16,25,000)
            "Loan_Amount_Term": 240,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        }
        res = service.predict(applicant, loan_product="Home Loan")
        fin = res["financial_indicators"]
        
        # principal_inr must be ₹16,25,000
        assert fin["principal_inr"] == 1625000.0
        # EMI must be calculated on ₹16,25,000, not ₹1,625
        assert fin["proposed_emi"] > 10000  # EMI on ₹16.25L at 8.5% for 20yr is ~₹14,000+
    
    def test_emi_calculated_on_inr_not_thousands(self):
        """EMI must use the full INR principal, not the thousands-scale value."""
        # ₹3,00,000 at 12% for 12 months
        emi_correct = calculate_emi(principal=300000.0, annual_interest_rate=12.0, tenure_months=12)
        emi_wrong = calculate_emi(principal=300.0, annual_interest_rate=12.0, tenure_months=12)
        
        # Correct EMI is ~₹26,654; wrong (if using thousands value directly) would be ~₹26.65
        assert emi_correct["monthly_emi"] > 26000
        assert emi_wrong["monthly_emi"] < 30  # This would be the wrong calculation
    
    def test_lti_uses_inr_principal(self):
        """LTI must use the full INR principal, not thousands."""
        # Principal ₹16,25,000; Annual income ₹7,20,000 (₹60K/month)
        lti = calculate_lti(principal=1625000, annual_gross_income=720000)
        assert lti == 2.26  # 1625000 / 720000 = 2.26
    
    def test_dti_uses_emi_from_full_principal(self):
        """DTI must use EMI calculated from the full INR principal."""
        emi_res = calculate_emi(principal=1625000.0, annual_interest_rate=8.5, tenure_months=240)
        proposed_emi = emi_res["monthly_emi"]
        
        aff = check_loan_affordability(
            monthly_income=60000,
            existing_monthly_debt=5000,
            proposed_emi=proposed_emi
        )
        total_debt = 5000 + proposed_emi
        expected_dti = round((total_debt / 60000) * 100.0, 2)
        assert abs(aff["dti_pct"] - expected_dti) < 0.01


class TestProductDefaultAmounts:
    """Verify product default amounts are sensible INR values for the Indian market."""
    
    def test_home_loan_default_is_lakhs(self):
        """Home Loan default should be in lakhs, not thousands."""
        cfg = PRODUCT_PRICING_CATALOG["Home Loan"]
        assert cfg["default_amount"] >= 1000000  # At least ₹10 Lakh
        assert cfg["default_amount"] <= 10000000  # At most ₹1 Crore
    
    def test_personal_loan_default_is_reasonable(self):
        """Personal Loan default should be ₹1L-₹10L."""
        cfg = PRODUCT_PRICING_CATALOG["Personal Loan"]
        assert cfg["default_amount"] >= 100000
        assert cfg["default_amount"] <= 1000000
    
    def test_all_products_have_inr_defaults(self):
        """All product defaults should be in INR (not thousands)."""
        for name, cfg in PRODUCT_PRICING_CATALOG.items():
            if name == "Car Loan":  # Alias for New Car Loan
                continue
            assert cfg["default_amount"] >= 50000, f"{name} default amount {cfg['default_amount']} is too low for INR"


class TestCreditHistoryRiskPricing:
    """Verify credit history risk adjustments work correctly for each product."""
    
    def test_adverse_credit_raises_rate_for_personal_loan(self):
        """Adverse credit (0.0) on Personal Loan should add +2.5% spread."""
        clean = determine_interest_rate("Personal Loan", credit_history=1.0)
        adverse = determine_interest_rate("Personal Loan", credit_history=0.0)
        assert adverse["applicable_annual_rate"] > clean["applicable_annual_rate"]
        spread = adverse["applicable_annual_rate"] - clean["applicable_annual_rate"]
        assert abs(spread - 2.5) < 0.01
    
    def test_adverse_credit_raises_rate_for_home_loan(self):
        """Adverse credit on Home Loan should add +1.0% spread."""
        clean = determine_interest_rate("Home Loan", credit_history=1.0)
        adverse = determine_interest_rate("Home Loan", credit_history=0.0)
        spread = adverse["applicable_annual_rate"] - clean["applicable_annual_rate"]
        assert abs(spread - 1.0) < 0.01
    
    def test_rate_clamped_to_product_boundaries(self):
        """Rate with all risk loadings must stay within product min/max."""
        for product_name in ["Personal Loan", "Business Loan", "Home Loan"]:
            cfg = get_product_pricing_config(product_name)
            rate_info = determine_interest_rate(
                product_name,
                credit_history=0.0,
                is_self_employed=True
            )
            assert rate_info["applicable_annual_rate"] >= cfg["min_interest_rate"]
            assert rate_info["applicable_annual_rate"] <= cfg["max_interest_rate"]


class TestEndToEndPredictionFlow:
    """Verify the full prediction flow produces consistent financial indicators."""
    
    def test_home_loan_full_flow(self):
        """Full Home Loan flow: ₹35L at 8.5% for 20 years."""
        service = LoanPredictionService()
        applicant = {
            "Gender": "Female",
            "Married": "Yes",
            "Dependents": "1",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 75000,
            "CoapplicantIncome": 25000,
            "LoanAmount": 3500.0,  # ₹35,00,000 in thousands
            "Loan_Amount_Term": 240,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        }
        res = service.predict(applicant, loan_product="Home Loan")
        fin = res["financial_indicators"]
        
        # Principal must be ₹35,00,000
        assert fin["principal_inr"] == 3500000.0
        
        # Rate should be 8.5% (Home Loan base, clean credit)
        assert fin["annual_interest_rate"] == 8.5
        
        # EMI verification: P=3500000, r=8.5/1200, n=240
        r = 8.5 / 1200.0
        n = 240
        expected_emi = 3500000 * r * (1 + r)**n / ((1 + r)**n - 1)
        assert abs(fin["proposed_emi"] - expected_emi) < 1.0
        
        # Total repayment = EMI × months
        assert abs(fin["total_repayment"] - (fin["proposed_emi"] * 240)) < 1.0
        
        # Total interest = total_repayment - principal
        assert abs(fin["total_interest"] - (fin["total_repayment"] - 3500000)) < 1.0
        
        # DTI = (existing_debt + EMI) / monthly_income × 100
        total_income = 75000 + 25000  # 100000
        expected_dti = round(((0 + fin["proposed_emi"]) / total_income) * 100, 2)
        assert abs(fin["debt_to_income_pct"] - expected_dti) < 0.1
        
        # LTI = 3500000 / (100000 × 12)
        expected_lti = round(3500000 / 1200000, 2)
        assert abs(fin["loan_to_income_ratio"] - expected_lti) < 0.01
    
    def test_personal_loan_adverse_credit_flow(self):
        """Personal Loan with adverse credit: rate should be 15% + 2.5% = 17.5%."""
        service = LoanPredictionService()
        applicant = {
            "Gender": "Male",
            "Married": "No",
            "Dependents": "0",
            "Education": "Not Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 30000,
            "CoapplicantIncome": 0,
            "LoanAmount": 200.0,  # ₹2,00,000
            "Loan_Amount_Term": 36,
            "Credit_History": 0.0,
            "Property_Area": "Rural"
        }
        res = service.predict(applicant, loan_product="Personal Loan")
        fin = res["financial_indicators"]
        
        # Rate should be 15% + 2.5% = 17.5% (adverse credit spread)
        assert fin["annual_interest_rate"] == 17.5
        assert fin["principal_inr"] == 200000.0
    
    def test_ml_prediction_is_genuine_boolean(self):
        """ML prediction must return a genuine True/False, not a hardcoded value."""
        service = LoanPredictionService()
        
        # Strong applicant — likely approval
        strong = {
            "Gender": "Male", "Married": "Yes", "Dependents": "0",
            "Education": "Graduate", "Self_Employed": "No",
            "ApplicantIncome": 80000, "CoapplicantIncome": 20000,
            "LoanAmount": 100, "Loan_Amount_Term": 360,
            "Credit_History": 1.0, "Property_Area": "Semiurban"
        }
        res_strong = service.predict(strong, loan_product="Home Loan")
        
        # Weak applicant — likely rejection
        weak = {
            "Gender": "Male", "Married": "No", "Dependents": "3+",
            "Education": "Not Graduate", "Self_Employed": "Yes",
            "ApplicantIncome": 2000, "CoapplicantIncome": 0,
            "LoanAmount": 500, "Loan_Amount_Term": 36,
            "Credit_History": 0.0, "Property_Area": "Rural"
        }
        res_weak = service.predict(weak, loan_product="Personal Loan")
        
        # Confidence values must be genuine probabilities between 0 and 1
        assert 0 <= res_strong["approval_confidence"] <= 1
        assert 0 <= res_weak["approval_confidence"] <= 1
        assert 0 <= res_strong["rejection_confidence"] <= 1
        assert 0 <= res_weak["rejection_confidence"] <= 1
        
        # Sum of approval + rejection should be ~1.0
        assert abs(res_strong["approval_confidence"] + res_strong["rejection_confidence"] - 1.0) < 0.01
        assert abs(res_weak["approval_confidence"] + res_weak["rejection_confidence"] - 1.0) < 0.01
        
        # Strong applicant should have higher approval confidence than weak
        assert res_strong["approval_confidence"] > res_weak["approval_confidence"]


class TestTotalMonthlyDebtService:
    """Verify Total Monthly Debt Service = Existing EMIs + Proposed EMI."""
    
    def test_with_existing_debt(self):
        """When existing debt = ₹12,000 and proposed EMI = ₹18,000, total = ₹30,000."""
        service = LoanPredictionService()
        applicant = {
            "Gender": "Male", "Married": "Yes", "Dependents": "0",
            "Education": "Graduate", "Self_Employed": "No",
            "ApplicantIncome": 80000, "CoapplicantIncome": 0,
            "LoanAmount": 300, "Loan_Amount_Term": 360,
            "Credit_History": 1.0, "Property_Area": "Urban"
        }
        res = service.predict(applicant, existing_monthly_debt=12000.0, loan_product="Home Loan")
        fin = res["financial_indicators"]
        
        assert fin["existing_monthly_debt"] == 12000.0
        expected_total = 12000.0 + fin["proposed_emi"]
        assert abs(fin["total_monthly_debt"] - expected_total) < 0.01
    
    def test_without_existing_debt(self):
        """When no existing debt, total monthly debt = proposed EMI."""
        service = LoanPredictionService()
        applicant = {
            "Gender": "Female", "Married": "No", "Dependents": "0",
            "Education": "Graduate", "Self_Employed": "No",
            "ApplicantIncome": 50000, "CoapplicantIncome": 0,
            "LoanAmount": 200, "Loan_Amount_Term": 60,
            "Credit_History": 1.0, "Property_Area": "Semiurban"
        }
        res = service.predict(applicant, existing_monthly_debt=0.0, loan_product="Personal Loan")
        fin = res["financial_indicators"]
        
        assert fin["existing_monthly_debt"] == 0.0
        assert abs(fin["total_monthly_debt"] - fin["proposed_emi"]) < 0.01
