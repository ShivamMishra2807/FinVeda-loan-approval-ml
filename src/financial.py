"""
Financial Planning, EMI Calculation, Amortization, and Affordability Module.
Provides verified reducing-balance EMI computations, monthly amortization schedules,
debt-to-income affordability analytics, and multi-scenario loan comparisons.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Configurable Loan Category Defaults
# Values are illustrative benchmark market ranges for Indian retail banking
LOAN_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Personal Loan": {
        "default_interest_rate": 12.0,
        "min_interest_rate": 10.5,
        "max_interest_rate": 24.0,
        "default_tenure_years": 3,
        "min_tenure_years": 1,
        "max_tenure_years": 7,
        "default_amount": 300000.0,
        "processing_fee_pct": 1.5,
        "description": "Unsecured personal credit facility for immediate personal expenses or medical needs."
    },
    "Home Loan": {
        "default_interest_rate": 8.5,
        "min_interest_rate": 7.5,
        "max_interest_rate": 11.0,
        "default_tenure_years": 20,
        "min_tenure_years": 5,
        "max_tenure_years": 30,
        "default_amount": 3500000.0,
        "processing_fee_pct": 0.5,
        "description": "Long-term secured facility for residential property purchase, renovation, or construction."
    },
    "Car Loan": {
        "default_interest_rate": 9.5,
        "min_interest_rate": 8.0,
        "max_interest_rate": 15.0,
        "default_tenure_years": 5,
        "min_tenure_years": 1,
        "max_tenure_years": 8,
        "default_amount": 800000.0,
        "processing_fee_pct": 1.0,
        "description": "Secured vehicle financing facility with fixed repayment terms."
    },
    "Education Loan": {
        "default_interest_rate": 9.5,
        "min_interest_rate": 8.0,
        "max_interest_rate": 14.0,
        "default_tenure_years": 7,
        "min_tenure_years": 2,
        "max_tenure_years": 15,
        "default_amount": 1200000.0,
        "processing_fee_pct": 0.75,
        "description": "Academic finance facility for domestic or overseas higher education."
    },
    "Business Loan": {
        "default_interest_rate": 14.0,
        "min_interest_rate": 11.0,
        "max_interest_rate": 24.0,
        "default_tenure_years": 4,
        "min_tenure_years": 1,
        "max_tenure_years": 10,
        "default_amount": 1500000.0,
        "processing_fee_pct": 2.0,
        "description": "Commercial loan facility for enterprise expansion, machinery, or working capital."
    }
}


def calculate_lti(principal: float, annual_gross_income: float) -> float:
    """
    Computes Loan-to-Income (LTI) ratio: Principal / Annual Gross Income.
    
    Where Annual Gross Income = Monthly Income * 12.
    Returns 0.0 if principal is 0 or negative.
    Returns float('inf') or bounded ratio if annual income is 0.
    """
    p = float(principal)
    income = float(annual_gross_income)
    if p <= 0:
        return 0.0
    if income <= 0:
        return 999.99
    return round(p / income, 2)



def calculate_emi(
    principal: float,
    annual_interest_rate: float,
    tenure_months: int,
    processing_fee_pct: float = 0.0,
    additional_charges: float = 0.0
) -> Dict[str, Any]:
    """
    Computes standard reducing-balance Equated Monthly Instalment (EMI).
    
    Formula:
    EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)
    
    Where:
    - P = principal loan amount
    - r = monthly interest rate = annual_interest_rate / (12 * 100)
    - n = tenure in months
    
    Handles zero-interest (0%) edge case gracefully: EMI = P / n.
    """
    # Defensive validations
    if principal < 0:
        raise ValueError("Principal loan amount cannot be negative.")
    if annual_interest_rate < 0:
        raise ValueError("Annual interest rate cannot be negative.")
    if tenure_months <= 0:
        raise ValueError("Tenure in months must be greater than zero.")
    if processing_fee_pct < 0:
        raise ValueError("Processing fee percentage cannot be negative.")
    if additional_charges < 0:
        raise ValueError("Additional charges cannot be negative.")
        
    p = float(principal)
    n = int(tenure_months)
    annual_rate = float(annual_interest_rate)
    
    if p == 0:
        return {
            "principal": 0.0,
            "annual_interest_rate": annual_rate,
            "tenure_months": n,
            "monthly_interest_rate_pct": annual_rate / 12.0,
            "monthly_emi": 0.0,
            "total_interest": 0.0,
            "total_repayment": 0.0,
            "processing_fees": 0.0,
            "additional_charges": 0.0,
            "total_cost_with_fees": 0.0
        }
        
    # Zero interest special handling
    if annual_rate == 0.0:
        monthly_emi = p / n
        total_interest = 0.0
        total_repayment = p
    else:
        r = annual_rate / 1200.0  # monthly interest fraction
        # EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)
        compounded = (1.0 + r) ** n
        monthly_emi = (p * r * compounded) / (compounded - 1.0)
        
        # Rounding EMI to standard precision
        monthly_emi = round(monthly_emi, 2)
        total_repayment = monthly_emi * n
        total_interest = max(total_repayment - p, 0.0)
        
    processing_fee_amt = round(p * (processing_fee_pct / 100.0), 2)
    fee_inclusive_cost = round(total_repayment + processing_fee_amt + additional_charges, 2)
    
    return {
        "principal": round(p, 2),
        "annual_interest_rate": annual_rate,
        "tenure_months": n,
        "monthly_interest_rate_pct": round(annual_rate / 12.0, 4),
        "monthly_emi": monthly_emi,
        "total_interest": round(total_interest, 2),
        "total_repayment": round(total_repayment, 2),
        "processing_fees": processing_fee_amt,
        "additional_charges": round(additional_charges, 2),
        "total_cost_with_fees": fee_inclusive_cost
    }


def generate_amortization_schedule(
    principal: float,
    annual_interest_rate: float,
    tenure_months: int
) -> pd.DataFrame:
    """
    Generates a month-by-month reducing balance repayment schedule.
    
    Columns:
    - Instalment_No
    - Opening_Balance
    - Monthly_EMI
    - Interest_Component
    - Principal_Component
    - Closing_Balance
    """
    if principal <= 0 or tenure_months <= 0:
        return pd.DataFrame(columns=[
            "Instalment_No", "Opening_Balance", "Monthly_EMI",
            "Interest_Component", "Principal_Component", "Closing_Balance"
        ])
        
    emi_calc = calculate_emi(principal, annual_interest_rate, tenure_months)
    monthly_emi = emi_calc["monthly_emi"]
    annual_rate = float(annual_interest_rate)
    monthly_rate = annual_rate / 1200.0 if annual_rate > 0 else 0.0
    
    records = []
    current_balance = float(principal)
    
    for month in range(1, tenure_months + 1):
        opening_balance = current_balance
        
        if monthly_rate > 0:
            interest_comp = round(opening_balance * monthly_rate, 2)
        else:
            interest_comp = 0.0
            
        # On final instalment, adjust principal component to clear remaining balance exactly
        if month == tenure_months:
            principal_comp = round(opening_balance, 2)
            actual_emi = round(principal_comp + interest_comp, 2)
            closing_balance = 0.0
        else:
            principal_comp = round(monthly_emi - interest_comp, 2)
            # Guard against edge-case negative principal if EMI < interest
            if principal_comp < 0:
                principal_comp = 0.0
            closing_balance = max(round(opening_balance - principal_comp, 2), 0.0)
            actual_emi = monthly_emi
            
        current_balance = closing_balance
        
        records.append({
            "Instalment_No": month,
            "Opening_Balance": round(opening_balance, 2),
            "Monthly_EMI": actual_emi,
            "Interest_Component": interest_comp,
            "Principal_Component": principal_comp,
            "Closing_Balance": closing_balance
        })
        
    return pd.DataFrame(records)


def check_loan_affordability(
    monthly_income: float,
    existing_monthly_debt: float,
    proposed_emi: float
) -> Dict[str, Any]:
    """
    Evaluates applicant repayment bandwidth against monthly cash flows.
    
    Metrics:
    - total_monthly_debt = existing_monthly_debt + proposed_emi
    - debt_to_income_ratio (DTI) = total_monthly_debt / monthly_income * 100
    - net_disposable_income = monthly_income - total_monthly_debt
    """
    income = float(monthly_income)
    existing_debt = float(existing_monthly_debt)
    emi = float(proposed_emi)
    
    if income < 0 or existing_debt < 0 or emi < 0:
        raise ValueError("Income and debt parameters must be non-negative.")
        
    total_debt = existing_debt + emi
    
    if income == 0.0:
        dti_pct = 100.0 if total_debt > 0 else 0.0
        remaining_income = -total_debt
        status = "Critical (Zero Income)"
        status_color = "red"
        advice = "No documented income provided. Borrowing without cash flows poses severe default risk."
    else:
        dti_pct = round((total_debt / income) * 100.0, 2)
        remaining_income = round(income - total_debt, 2)
        
        if dti_pct <= 35.0:
            status = "Comfortable (Low Risk)"
            status_color = "green"
            advice = "Monthly debt service is within healthy prudential bounds (below 35% of gross income)."
        elif dti_pct <= 50.0:
            status = "Moderate Bandwidth"
            status_color = "amber"
            advice = "Debt obligations are manageable (35% - 50%), but discretionary buffers may be tight."
        elif dti_pct <= 65.0:
            status = "High Debt Burden (Stretched)"
            status_color = "orange"
            advice = "Debt consumes more than 50% of monthly income. Banks typically require co-borrowers or debt consolidation."
        else:
            status = "Severe Financial Strain"
            status_color = "red"
            advice = "Debt-to-income exceeds 65%. High vulnerability to income shocks or interest rate rises."
            
    return {
        "monthly_income": income,
        "existing_monthly_debt": existing_debt,
        "proposed_emi": emi,
        "total_monthly_debt": round(total_debt, 2),
        "remaining_income": remaining_income,
        "dti_pct": dti_pct,
        "affordability_status": status,
        "status_color": status_color,
        "guidance": advice,
        "assumptions_note": "Prudential benchmark guidelines assume consistent gross monthly receipts and do not account for variable bonuses or non-debt living expenses."
    }


def compare_loan_scenarios(
    scenarios: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compares up to 3 loan alternatives.
    
    Each scenario dictionary must contain:
    - name: str
    - principal: float
    - annual_interest_rate: float
    - tenure_months: int
    - processing_fee_pct: float (optional)
    - additional_charges: float (optional)
    """
    rows = []
    for s in scenarios:
        name = s.get("name", "Scenario")
        p = float(s.get("principal", 0.0))
        rate = float(s.get("annual_interest_rate", 0.0))
        tenure = int(s.get("tenure_months", 12))
        fee_pct = float(s.get("processing_fee_pct", 0.0))
        extra_charges = float(s.get("additional_charges", 0.0))
        
        res = calculate_emi(
            principal=p,
            annual_interest_rate=rate,
            tenure_months=tenure,
            processing_fee_pct=fee_pct,
            additional_charges=extra_charges
        )
        
        rows.append({
            "Scenario": name,
            "Principal (₹)": res["principal"],
            "Annual Rate (%)": res["annual_interest_rate"],
            "Tenure (Months)": res["tenure_months"],
            "Tenure (Years)": round(res["tenure_months"] / 12.0, 1),
            "Monthly EMI (₹)": res["monthly_emi"],
            "Total Interest (₹)": res["total_interest"],
            "Total Repayment (₹)": res["total_repayment"],
            "Fees & Charges (₹)": round(res["processing_fees"] + res["additional_charges"], 2),
            "Total Cost with Fees (₹)": res["total_cost_with_fees"]
        })
        
    return pd.DataFrame(rows)
