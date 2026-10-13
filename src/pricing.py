"""
FinVeda Loan Product Pricing Rules & Interest Rate Determination Module.

Provides institutional pricing configurations, benchmark ranges, documented risk-based
rate adjustments, and automated interest rate determination for retail and commercial loan products.
"""

from typing import Dict, Any, List, Optional, Tuple


# Configurable Institutional Loan Product Pricing Catalog
# Note: Benchmark rates and spreads are illustrative demonstration defaults for Indian retail banking.
PRODUCT_PRICING_CATALOG: Dict[str, Dict[str, Any]] = {
    "Personal Loan": {
        "product_name": "Personal Loan",
        "default_illustrative_rate": 12.0,
        "min_interest_rate": 10.5,
        "max_interest_rate": 24.0,
        "min_tenure_months": 12,
        "max_tenure_months": 84,
        "default_tenure_months": 36,
        "processing_fee_pct": 1.5,
        "collateral_type": "Unsecured",
        "description": "Unsecured retail credit for medical emergencies, debt consolidation, or immediate liquidity.",
        "pricing_rationale": "Base rate set at 12.0%. Risk loading applies for adverse credit history (+3.5%) and high debt burden.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Home Loan": {
        "product_name": "Home Loan",
        "default_illustrative_rate": 8.5,
        "min_interest_rate": 7.5,
        "max_interest_rate": 11.0,
        "min_tenure_months": 60,
        "max_tenure_months": 360,
        "default_tenure_months": 240,
        "processing_fee_pct": 0.5,
        "collateral_type": "Secured (Property Mortgage)",
        "description": "Long-term mortgage facility for purchase, construction, or extension of residential property.",
        "pricing_rationale": "Secured benchmark base rate at 8.5%. Adverse credit track record adds +1.5% prudential loading.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Car Loan": {
        "product_name": "Car Loan",
        "default_illustrative_rate": 9.5,
        "min_interest_rate": 8.0,
        "max_interest_rate": 15.0,
        "min_tenure_months": 12,
        "max_tenure_months": 96,
        "default_tenure_months": 60,
        "processing_fee_pct": 1.0,
        "collateral_type": "Secured (Vehicle Hypothecation)",
        "description": "Secured auto financing facility against hypothecated private or commercial passenger vehicle.",
        "pricing_rationale": "Asset-backed auto base rate at 9.5%. Delinquent credit record incurs +2.0% risk loading.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Education Loan": {
        "product_name": "Education Loan",
        "default_illustrative_rate": 9.5,
        "min_interest_rate": 8.0,
        "max_interest_rate": 14.0,
        "min_tenure_months": 24,
        "max_tenure_months": 180,
        "default_tenure_months": 84,
        "processing_fee_pct": 0.75,
        "collateral_type": "Semi-Secured / Parent Guarantee",
        "description": "Academic finance facility covering tuition, travel, and campus living expenses for domestic or overseas study.",
        "pricing_rationale": "Priority academic base rate at 9.5%. Adverse co-borrower credit profile incurs +2.0% risk adjustment.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Business Loan": {
        "product_name": "Business Loan",
        "default_illustrative_rate": 14.0,
        "min_interest_rate": 11.0,
        "max_interest_rate": 24.0,
        "min_tenure_months": 12,
        "max_tenure_months": 120,
        "default_tenure_months": 48,
        "processing_fee_pct": 2.0,
        "collateral_type": "Commercial / Business Floating Charge",
        "description": "Commercial loan facility for MSME working capital, machinery procurement, or enterprise scaling.",
        "pricing_rationale": "Commercial base rate at 14.0%. Self-employment revenue variance adds +0.5%; delinquent history adds +3.0%.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    }
}


def get_product_pricing_config(loan_product: str) -> Dict[str, Any]:
    """
    Retrieves pricing rules and boundaries for a given loan product.
    Falls back to Personal Loan if unrecognized.
    """
    if loan_product not in PRODUCT_PRICING_CATALOG:
        return PRODUCT_PRICING_CATALOG["Personal Loan"]
    return PRODUCT_PRICING_CATALOG[loan_product]


def determine_interest_rate(
    loan_product: str,
    credit_history: float = 1.0,
    is_self_employed: bool = False,
    existing_dti_pct: Optional[float] = None,
    override_rate: Optional[float] = None,
    is_simulation_mode: bool = False
) -> Dict[str, Any]:
    """
    Determines the applicable annual interest rate based on institutional pricing rules.
    
    In Normal Applicant Workflow:
    - Rate is derived automatically from the selected product and documented credit risk parameters.
    - Rate is read-only for applicants.
    
    In Underwriter / Simulation Override Mode:
    - Allows authorized sensitivity testing and stress-testing override.
    - Explicitly flagged as simulation.
    
    Returns:
    - applicable_annual_rate: float
    - base_rate: float
    - rate_breakdown: List[Dict[str, Any]]
    - min_allowed_rate: float
    - max_allowed_rate: float
    - is_simulation: bool
    - pricing_source_description: str
    - disclaimer: str
    """
    config = get_product_pricing_config(loan_product)
    min_rate = float(config["min_interest_rate"])
    max_rate = float(config["max_interest_rate"])
    base_rate = float(config["default_illustrative_rate"])

    # If simulation override mode is explicitly activated
    if is_simulation_mode and override_rate is not None:
        clamped_override = max(min_rate, min(max_rate, float(override_rate)))
        return {
            "applicable_annual_rate": round(clamped_override, 2),
            "base_rate": base_rate,
            "risk_adjustments": [{
                "factor": "Simulation / Underwriter Override",
                "adjustment_pct": round(clamped_override - base_rate, 2),
                "rationale": "Manual rate stress-test scenario by authorized underwriter."
            }],
            "min_allowed_rate": min_rate,
            "max_allowed_rate": max_rate,
            "is_simulation": True,
            "pricing_source": "Underwriter Simulation Override (Testing Mode Only)",
            "pricing_rationale": f"Rate manually adjusted to {clamped_override:.2f}% for scenario stress-testing.",
            "disclaimer": "⚠️ Simulated test rate. Not an applicant-assigned or binding rate."
        }

    # Normal Applicant Workflow: Automated Pricing Determination Rule
    adjustments: List[Dict[str, Any]] = []
    current_rate = base_rate

    # Rule 1: Credit History Risk Loading
    # Applicants without clean track record (Credit_History = 0.0) face adverse credit spread
    if float(credit_history) == 0.0:
        if loan_product in ["Personal Loan", "Business Loan"]:
            credit_spread = 3.5
        else:
            credit_spread = 2.0
        current_rate += credit_spread
        adjustments.append({
            "factor": "Adverse Credit History Loading",
            "adjustment_pct": credit_spread,
            "rationale": "Past delinquent credit obligations warrant institutional risk margin."
        })
    else:
        adjustments.append({
            "factor": "Clean Credit History Track Record",
            "adjustment_pct": 0.0,
            "rationale": "Meets benchmark credit underwriting criteria with zero risk penalty."
        })

    # Rule 2: Self-Employed Commercial Risk Loading
    if is_self_employed and loan_product in ["Business Loan", "Personal Loan"]:
        self_employed_spread = 0.5
        current_rate += self_employed_spread
        adjustments.append({
            "factor": "Self-Employed Cash-Flow Variance Premium",
            "adjustment_pct": self_employed_spread,
            "rationale": "Unsalaried variable receipts require moderate liquidity spread."
        })

    # Rule 3: Elevated Debt Service Loading
    if existing_dti_pct is not None and existing_dti_pct > 50.0:
        dti_spread = 0.5
        current_rate += dti_spread
        adjustments.append({
            "factor": "High Existing Debt-To-Income Spread",
            "adjustment_pct": dti_spread,
            "rationale": "Pre-existing leverage exceeds 50% prudential guideline."
        })

    # Enforce product regulatory boundaries
    final_rate = max(min_rate, min(max_rate, current_rate))
    final_rate = round(final_rate, 2)

    return {
        "applicable_annual_rate": final_rate,
        "base_rate": base_rate,
        "risk_adjustments": adjustments,
        "min_allowed_rate": min_rate,
        "max_allowed_rate": max_rate,
        "is_simulation": False,
        "pricing_source": f"FinVeda Institutional Pricing Engine ({loan_product} Rule Matrix)",
        "pricing_rationale": (
            f"Base illustrative rate of {base_rate:.2f}% with "
            f"{len([a for a in adjustments if a['adjustment_pct'] > 0])} risk adjustment(s) applied."
        ),
        "disclaimer": "Institutional demonstration pricing rule based on retail banking spreads. Not a guaranteed lender contract."
    }
