"""
Financial Calculation and Planning Test Suite.
Tests reducing-balance EMI calculations, zero-interest edge cases, amortization closure,
affordability metrics, input bounds validation, and multi-loan scenario comparison.
"""

import pytest
import pandas as pd
import numpy as np

from src.financial import (
    calculate_emi,
    generate_amortization_schedule,
    check_loan_affordability,
    compare_loan_scenarios,
    LOAN_CATEGORIES
)


def test_standard_emi_calculation_prd_example():
    """
    Test PRD benchmark example:
    Principal = ₹5,00,000, Annual Rate = 10%, Tenure = 5 years (60 months)
    Expected:
    - Monthly EMI ≈ ₹10,624 (exact formula: ₹10,623.52)
    - Total Interest ≈ ₹1.37 lakh (exact: ₹1,37,411.20)
    - Total Repayment ≈ ₹6.37 lakh (exact: ₹6,37,411.20)
    """
    res = calculate_emi(principal=500000, annual_interest_rate=10.0, tenure_months=60)
    assert round(res["monthly_emi"]) == 10624
    assert res["principal"] == 500000.0
    assert abs(res["total_interest"] - 137411.20) < 1.0
    assert abs(res["total_repayment"] - 637411.20) < 1.0
    assert abs(res["total_repayment"] - (res["principal"] + res["total_interest"])) < 0.01


def test_zero_interest_loan():
    """Verify zero-interest loans handle division-by-zero safely: EMI = P / n."""
    res = calculate_emi(principal=120000, annual_interest_rate=0.0, tenure_months=12)
    assert res["monthly_emi"] == 10000.0
    assert res["total_interest"] == 0.0
    assert res["total_repayment"] == 120000.0


def test_zero_principal_loan():
    """Verify zero principal handles gracefully."""
    res = calculate_emi(principal=0, annual_interest_rate=12.0, tenure_months=24)
    assert res["monthly_emi"] == 0.0
    assert res["total_interest"] == 0.0
    assert res["total_repayment"] == 0.0


def test_negative_inputs_raise_errors():
    """Defensive validation against negative inputs."""
    with pytest.raises(ValueError):
        calculate_emi(principal=-50000, annual_interest_rate=10.0, tenure_months=12)
        
    with pytest.raises(ValueError):
        calculate_emi(principal=50000, annual_interest_rate=-5.0, tenure_months=12)
        
    with pytest.raises(ValueError):
        calculate_emi(principal=50000, annual_interest_rate=10.0, tenure_months=0)


def test_amortization_schedule_closure():
    """Verify amortization schedule starts at opening balance and closes at 0.0."""
    p = 250000.0
    rate = 12.0
    months = 24
    df = generate_amortization_schedule(p, rate, months)
    
    assert len(df) == months
    assert df.iloc[0]["Opening_Balance"] == p
    assert df.iloc[-1]["Closing_Balance"] == 0.0
    assert (df["Principal_Component"] > 0).all()
    assert (df["Interest_Component"] >= 0).all()
    # Sum of principal components should equal initial principal
    assert abs(df["Principal_Component"].sum() - p) < 1.0


def test_affordability_checker():
    """Verify debt-to-income and cash flow math."""
    res = check_loan_affordability(monthly_income=100000, existing_monthly_debt=15000, proposed_emi=20000)
    assert res["total_monthly_debt"] == 35000.0
    assert res["remaining_income"] == 65000.0
    assert res["dti_pct"] == 35.0
    assert "Comfortable" in res["affordability_status"]
    
    # Critical threshold
    res_high = check_loan_affordability(monthly_income=50000, existing_monthly_debt=25000, proposed_emi=15000)
    assert res_high["dti_pct"] == 80.0
    assert "Severe" in res_high["affordability_status"]


def test_affordability_zero_income():
    """Ensure zero income is handled without crash."""
    res = check_loan_affordability(monthly_income=0, existing_monthly_debt=5000, proposed_emi=5000)
    assert res["dti_pct"] == 100.0
    assert res["remaining_income"] == -10000.0
    assert "Critical" in res["affordability_status"]


def test_loan_scenario_comparison():
    """Verify comparison table generation across multiple configurations."""
    scenarios = [
        {"name": "Bank A (Short)", "principal": 500000, "annual_interest_rate": 10.0, "tenure_months": 36},
        {"name": "Bank B (Medium)", "principal": 500000, "annual_interest_rate": 9.5, "tenure_months": 60},
        {"name": "Bank C (Long)", "principal": 500000, "annual_interest_rate": 9.0, "tenure_months": 84}
    ]
    df = compare_loan_scenarios(scenarios)
    assert len(df) == 3
    assert "Monthly EMI (₹)" in df.columns
    assert "Total Interest (₹)" in df.columns
    # Longer tenure should have lower EMI but higher total interest
    assert df.iloc[0]["Monthly EMI (₹)"] > df.iloc[2]["Monthly EMI (₹)"]
    assert df.iloc[0]["Total Interest (₹)"] < df.iloc[2]["Total Interest (₹)"]


def test_loan_categories_structure():
    """Ensure all required categories are present with reasonable defaults."""
    required = ["Personal Loan", "Home Loan", "Car Loan", "Education Loan", "Business Loan"]
    for cat in required:
        assert cat in LOAN_CATEGORIES
        assert LOAN_CATEGORIES[cat]["default_interest_rate"] > 0
        assert LOAN_CATEGORIES[cat]["default_tenure_years"] > 0
