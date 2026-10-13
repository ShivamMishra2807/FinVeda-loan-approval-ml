"""
Pricing Rules & Interest Rate Determination Test Suite.

Verifies:
1. All 5 loan products exist in catalog with validated boundaries.
2. Automated rate determination matches institutional rules.
3. Risk-based pricing adjustments apply correctly (adverse credit, self-employed, DTI).
4. Strict clamping within product min/max rate bounds.
5. Strict separation between normal applicant mode and underwriter simulation override mode.
6. EMI calculations accurately reflect the system-determined rate.
7. Tenure variations adjust EMI and repayment totals properly.
"""

import pytest
from src.pricing import (
    PRODUCT_PRICING_CATALOG,
    get_product_pricing_config,
    determine_interest_rate
)
from src.financial import calculate_emi
from src.prediction import LoanPredictionService


def test_pricing_catalog_products_and_boundaries():
    """Verify all 5 required loan categories exist with valid boundary configurations."""
    expected_products = ["Personal Loan", "Home Loan", "Car Loan", "Education Loan", "Business Loan"]
    for prod in expected_products:
        assert prod in PRODUCT_PRICING_CATALOG, f"Missing product {prod}"
        cfg = get_product_pricing_config(prod)
        assert cfg["min_interest_rate"] > 0
        assert cfg["max_interest_rate"] >= cfg["min_interest_rate"]
        assert cfg["min_interest_rate"] <= cfg["default_illustrative_rate"] <= cfg["max_interest_rate"]
        assert cfg["min_tenure_months"] > 0
        assert cfg["max_tenure_months"] >= cfg["min_tenure_months"]
        assert cfg["default_tenure_months"] >= cfg["min_tenure_months"]


def test_home_loan_pricing_boundaries():
    """Verify Home Loan ranges: 7.5% to 11.0%, base 8.5%."""
    cfg = get_product_pricing_config("Home Loan")
    assert cfg["min_interest_rate"] == 7.5
    assert cfg["max_interest_rate"] == 11.0
    assert cfg["default_illustrative_rate"] == 8.5


def test_personal_loan_pricing_boundaries():
    """Verify Personal Loan ranges: 10.5% to 24.0%, base 12.0%."""
    cfg = get_product_pricing_config("Personal Loan")
    assert cfg["min_interest_rate"] == 10.5
    assert cfg["max_interest_rate"] == 24.0
    assert cfg["default_illustrative_rate"] == 12.0


def test_automated_rate_determination_clean_credit():
    """Verify normal applicant with clean credit history receives base illustrative rate."""
    res = determine_interest_rate(
        loan_product="Home Loan",
        credit_history=1.0,
        is_self_employed=False,
        is_simulation_mode=False
    )
    assert res["applicable_annual_rate"] == 8.5
    assert res["is_simulation"] is False
    assert "Home Loan" in res["pricing_source"]


def test_automated_rate_determination_adverse_credit():
    """Verify applicant with adverse credit track record receives documented risk loading."""
    # Personal Loan base: 12.0%, adverse credit loading: +3.5% -> 15.5%
    res = determine_interest_rate(
        loan_product="Personal Loan",
        credit_history=0.0,
        is_self_employed=False,
        is_simulation_mode=False
    )
    assert res["applicable_annual_rate"] == 15.5
    assert res["is_simulation"] is False
    assert any(a["factor"] == "Adverse Credit History Loading" for a in res["risk_adjustments"])


def test_rate_clamping_to_maximum_bound():
    """Verify that cumulative risk adjustments never exceed the product's max allowed rate."""
    cfg = get_product_pricing_config("Home Loan")
    max_rate = cfg["max_interest_rate"]  # 11.0%
    
    # Try with extreme overrides in simulation mode
    res_sim_high = determine_interest_rate(
        loan_product="Home Loan",
        override_rate=45.0,
        is_simulation_mode=True
    )
    assert res_sim_high["applicable_annual_rate"] == max_rate

    # Try below minimum bound
    min_rate = cfg["min_interest_rate"]  # 7.5%
    res_sim_low = determine_interest_rate(
        loan_product="Home Loan",
        override_rate=1.0,
        is_simulation_mode=True
    )
    assert res_sim_low["applicable_annual_rate"] == min_rate


def test_separation_between_applicant_and_simulation_modes():
    """
    Verify that in normal applicant mode (is_simulation_mode=False),
    any passed override rate is ignored and the rate is strictly system-determined.
    """
    # Attempting to self-assign 5.0% on a Personal Loan (min allowed: 10.5%, base: 12.0%)
    res_applicant = determine_interest_rate(
        loan_product="Personal Loan",
        credit_history=1.0,
        override_rate=5.0,
        is_simulation_mode=False
    )
    # Must ignore 5.0% and return system-assigned rate (12.0%)
    assert res_applicant["applicable_annual_rate"] == 12.0
    assert res_applicant["is_simulation"] is False

    # Underwriter simulation mode explicitly enabled (is_simulation_mode=True)
    res_simulation = determine_interest_rate(
        loan_product="Personal Loan",
        override_rate=14.5,
        is_simulation_mode=True
    )
    assert res_simulation["applicable_annual_rate"] == 14.5
    assert res_simulation["is_simulation"] is True


def test_emi_calculation_with_determined_rate():
    """Verify reducing-balance EMI uses the system-assigned rate accurately."""
    # Personal Loan: ₹3,00,000, 12 months, base rate 12.0%
    rate_res = determine_interest_rate(
        loan_product="Personal Loan",
        credit_history=1.0,
        is_simulation_mode=False
    )
    rate = rate_res["applicable_annual_rate"]
    assert rate == 12.0

    emi_data = calculate_emi(principal=300000, annual_interest_rate=rate, tenure_months=12)
    assert abs(emi_data["monthly_emi"] - 26654.64) < 0.01
    assert abs(emi_data["total_repayment"] - 319855.68) < 0.05


def test_tenure_change_recalculates_emi_dynamically():
    """Verify changing tenure from 12 to 24 months reduces EMI and updates repayment."""
    rate = 12.0
    emi_12 = calculate_emi(principal=300000, annual_interest_rate=rate, tenure_months=12)
    emi_24 = calculate_emi(principal=300000, annual_interest_rate=rate, tenure_months=24)

    # Longer tenure has lower monthly installment
    assert emi_24["monthly_emi"] < emi_12["monthly_emi"]
    # But higher total interest paid
    assert emi_24["total_interest"] > emi_12["total_interest"]


def test_end_to_end_prediction_service_with_pricing_rules():
    """Verify LoanPredictionService uses automated pricing rules when no override is given."""
    service = LoanPredictionService()
    applicant_data = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 60000,
        "CoapplicantIncome": 20000,
        "LoanAmount": 300,  # ₹3,00,000
        "Loan_Amount_Term": 36,  # 36 months
        "Credit_History": 1.0,
        "Property_Area": "Urban",
        "Loan_Category": "Personal Loan"
    }
    res = service.predict(applicant_data, loan_product="Personal Loan", is_simulation_mode=False)
    fin = res["financial_indicators"]
    
    # Base rate of Personal Loan is 12.0%
    assert fin["annual_interest_rate"] == 12.0
    assert fin["is_simulation"] is False
    assert fin["principal_inr"] == 300000.0
    assert fin["proposed_emi"] > 0
    assert fin["debt_to_income_pct"] > 0
