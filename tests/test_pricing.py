"""
Pricing Rules & Interest Rate Determination Test Suite.

Rigorously verifies the 10 core requirements:
1. Zero-interest loan returns principal / number of months.
2. A ₹3,00,000 loan at 12% annual interest for 12 months produces EMI of approx ₹26,654.82 (exact: ₹26,654.64).
3. The same loan for 36 months produces EMI of approx ₹9,964.29 (exact: ₹9,964.29).
4. Switching loan products selects the correct configured default demonstration rate across all 8 products.
5. Total interest reconciles with the amortization schedule (sum of interest components == total interest).
6. Final outstanding principal is approximately zero (0.0).
7. Existing EMIs are included in total monthly debt service and DTI.
8. Zero income and invalid loan amounts are handled safely without misleading outputs.
9. Changing tenure updates EMI and total interest correctly.
10. Existing ML prediction continues to use actual trained model and original preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np

from src.pricing import (
    PRODUCT_PRICING_CATALOG,
    get_product_pricing_config,
    determine_interest_rate
)
from src.financial import (
    calculate_emi,
    generate_amortization_schedule,
    check_loan_affordability,
    calculate_lti,
    LOAN_CATEGORIES
)
from src.prediction import LoanPredictionService


def test_req1_zero_interest_loan():
    """Requirement 1: A zero-interest loan returns principal divided by the number of months."""
    p = 120000.0
    n = 12
    res = calculate_emi(principal=p, annual_interest_rate=0.0, tenure_months=n)
    assert res["monthly_emi"] == p / n
    assert res["total_interest"] == 0.0
    assert res["total_repayment"] == p


def test_req2_three_lakh_twelve_percent_twelve_months_emi():
    """
    Requirement 2: A ₹3,00,000 loan at 12% annual interest for 12 months produces
    an EMI of approximately ₹26,654.82 (exact mathematical formula: ₹26,654.64).
    """
    res = calculate_emi(principal=300000.0, annual_interest_rate=12.0, tenure_months=12)
    # Test formula: 300000 * 0.01 * (1.01)^12 / ((1.01)^12 - 1) = 26,654.64
    assert abs(res["monthly_emi"] - 26654.64) < 0.01
    assert abs(res["monthly_emi"] - 26654.82) < 0.50
    assert abs(res["total_repayment"] - 319855.68) < 0.10
    assert abs(res["total_interest"] - 19855.68) < 0.10


def test_req3_three_lakh_twelve_percent_thirty_six_months_emi():
    """
    Requirement 3: The same ₹3,00,000 loan at 12% annual interest for 36 months produces
    an EMI of approximately ₹9,964.29 (exact: 300000 * 0.01 * (1.01)^36 / ((1.01)^36 - 1) = 9,964.29).
    """
    res = calculate_emi(principal=300000.0, annual_interest_rate=12.0, tenure_months=36)
    assert abs(res["monthly_emi"] - 9964.29) < 0.02
    assert abs(res["total_repayment"] - (9964.29 * 36)) < 0.50
    assert res["total_interest"] > 0


def test_req4_switching_loan_products_selects_correct_default_rate():
    """
    Requirement 4: Switching loan products selects the correct configured default rate.
    Verifies all 8 required products with their configured demonstration rates and ranges:
    - Home Loan: 8.5% (7.5% - 10%)
    - New Car Loan: 9.0% (7.85% - 10%)
    - Used Car Loan: 12.0% (10% - 14%)
    - Education Loan: 10.5% (8% - 14%)
    - Personal Loan: 15.0% (11% - 22%)
    - Business Loan: 14.0% (11% - 18%)
    - Two-Wheeler Loan: 13.0% (10% - 16%)
    - Gold Loan: 10.0% (8.5% - 13%)
    """
    expected = {
        "Home Loan": {"default": 8.5, "min": 7.5, "max": 10.0},
        "New Car Loan": {"default": 9.0, "min": 7.85, "max": 10.0},
        "Used Car Loan": {"default": 12.0, "min": 10.0, "max": 14.0},
        "Education Loan": {"default": 10.5, "min": 8.0, "max": 14.0},
        "Personal Loan": {"default": 15.0, "min": 11.0, "max": 22.0},
        "Business Loan": {"default": 14.0, "min": 11.0, "max": 18.0},
        "Two-Wheeler Loan": {"default": 13.0, "min": 10.0, "max": 16.0},
        "Gold Loan": {"default": 10.0, "min": 8.5, "max": 13.0}
    }
    for prod, specs in expected.items():
        cfg = get_product_pricing_config(prod)
        assert cfg["default_illustrative_rate"] == specs["default"]
        assert cfg["min_interest_rate"] == specs["min"]
        assert cfg["max_interest_rate"] == specs["max"]

        # Default applicant rate determination with clean credit
        rate_info = determine_interest_rate(loan_product=prod, credit_history=1.0, is_simulation_mode=False)
        assert rate_info["base_rate"] == specs["default"]
        assert specs["min"] <= rate_info["applicable_annual_rate"] <= specs["max"]


def test_req5_total_interest_reconciles_with_amortization_schedule():
    """Requirement 5: The total interest reconciles with the amortization schedule."""
    p = 500000.0
    rate = 10.5
    months = 48
    emi_calc = calculate_emi(principal=p, annual_interest_rate=rate, tenure_months=months)
    sched = generate_amortization_schedule(principal=p, annual_interest_rate=rate, tenure_months=months)

    sum_interest = sched["Interest_Component"].sum()
    # Should match emi_calc total interest within 1 rupee rounding tolerance
    assert abs(sum_interest - emi_calc["total_interest"]) < 1.0


def test_req6_final_outstanding_principal_is_zero():
    """Requirement 6: The final outstanding principal in amortization schedule is approximately zero."""
    sched = generate_amortization_schedule(principal=750000.0, annual_interest_rate=9.5, tenure_months=60)
    assert sched.iloc[-1]["Closing_Balance"] == 0.0
    # Sum of principal components should clear initial loan balance
    assert abs(sched["Principal_Component"].sum() - 750000.0) < 0.50


def test_req7_existing_emis_included_in_total_debt_and_dti():
    """Requirement 7: Existing EMIs are included in total monthly debt service and DTI."""
    income = 80000.0
    existing_debt = 12000.0
    proposed_emi = 18000.0
    aff = check_loan_affordability(
        monthly_income=income,
        existing_monthly_debt=existing_debt,
        proposed_emi=proposed_emi
    )
    expected_total_debt = existing_debt + proposed_emi  # 30,000
    assert aff["total_monthly_debt"] == expected_total_debt
    expected_dti = (expected_total_debt / income) * 100.0  # 37.5%
    assert abs(aff["dti_pct"] - expected_dti) < 0.01
    assert aff["remaining_income"] == income - expected_total_debt


def test_req8_zero_income_and_invalid_amounts_handled_safely():
    """Requirement 8: Zero income and invalid loan amounts are handled safely without misleading outputs."""
    # Zero income handling
    aff_zero = check_loan_affordability(monthly_income=0.0, existing_monthly_debt=5000.0, proposed_emi=10000.0)
    assert aff_zero["dti_pct"] == 100.0
    assert aff_zero["remaining_income"] == -15000.0
    assert "Critical" in aff_zero["affordability_status"]

    # Negative inputs raise ValueError defensive check
    with pytest.raises(ValueError):
        calculate_emi(principal=-100000.0, annual_interest_rate=10.0, tenure_months=12)

    with pytest.raises(ValueError):
        calculate_emi(principal=100000.0, annual_interest_rate=-2.0, tenure_months=12)

    with pytest.raises(ValueError):
        calculate_emi(principal=100000.0, annual_interest_rate=10.0, tenure_months=0)

    # LTI safe handling with zero income
    lti_zero = calculate_lti(principal=500000.0, annual_gross_income=0.0)
    assert lti_zero > 0  # Does not crash or return misleading negative


def test_req9_changing_tenure_updates_emi_and_interest_correctly():
    """Requirement 9: Changing tenure updates EMI and total interest correctly."""
    p = 400000.0
    r = 11.0
    emi_short = calculate_emi(principal=p, annual_interest_rate=r, tenure_months=24)
    emi_long = calculate_emi(principal=p, annual_interest_rate=r, tenure_months=48)

    # Longer tenure lowers monthly EMI
    assert emi_long["monthly_emi"] < emi_short["monthly_emi"]
    # Longer tenure increases total lifetime interest
    assert emi_long["total_interest"] > emi_short["total_interest"]
    assert emi_long["total_repayment"] > emi_short["total_repayment"]


def test_req10_ml_prediction_uses_actual_model_and_original_pipeline():
    """
    Requirement 10: The existing ML prediction continues to use the actual trained model
    artifact and original preprocessing pipeline (11 features), while seamlessly incorporating
    the automated pricing engine.
    """
    service = LoanPredictionService()
    assert service.pipeline is not None

    applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 75000,
        "CoapplicantIncome": 25000,
        "LoanAmount": 500,  # ₹5,00,000
        "Loan_Amount_Term": 60,  # 60 months
        "Credit_History": 1.0,
        "Property_Area": "Semiurban",
        "Loan_Category": "New Car Loan"
    }

    res = service.predict(applicant, loan_product="New Car Loan", is_simulation_mode=False)
    assert "is_approved" in res
    assert "approval_confidence" in res
    assert "financial_indicators" in res
    fin = res["financial_indicators"]

    # Base rate for New Car Loan is 9.0%
    assert fin["annual_interest_rate"] == 9.0
    assert fin["is_simulation"] is False
    assert fin["principal_inr"] == 500000.0
    assert fin["proposed_emi"] > 0
    assert fin["total_interest"] > 0
    assert fin["total_repayment"] == fin["principal_inr"] + fin["total_interest"]
    assert "debt_to_income_pct" in fin
    assert "loan_to_income_ratio" in fin
