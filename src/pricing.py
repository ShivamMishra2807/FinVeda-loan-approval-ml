"""
FinVeda Loan Product Pricing Rules & Interest Rate Determination Module.

Provides institutional pricing configurations, benchmark ranges, documented risk-based
rate adjustments, and automated interest rate determination for retail and commercial loan products.
"""

from typing import Dict, Any, List, Optional, Tuple


# Configurable Institutional Loan Product Pricing Catalog
# Note: Benchmark rates and spreads are illustrative demonstration defaults informed by Indian retail banking,
# not guaranteed current market rates. Keep them configurable in one central pricing configuration file.
PRODUCT_PRICING_CATALOG: Dict[str, Dict[str, Any]] = {
    "Home Loan": {
        "product_name": "Home Loan",
        "default_illustrative_rate": 8.5,
        "min_interest_rate": 7.5,
        "max_interest_rate": 10.0,
        "min_tenure_months": 60,
        "max_tenure_months": 360,
        "default_tenure_months": 240,
        "default_amount": 3500000.0,
        "processing_fee_pct": 0.5,
        "collateral_type": "Secured (Residential Property Mortgage)",
        "description": "Long-term mortgage facility for purchase, construction, or extension of residential property.",
        "pricing_rationale": "Illustrative base demonstration rate at 8.5%. Benchmark market corridor: 7.5% - 10.0%.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "New Car Loan": {
        "product_name": "New Car Loan",
        "default_illustrative_rate": 9.0,
        "min_interest_rate": 7.85,
        "max_interest_rate": 10.0,
        "min_tenure_months": 12,
        "max_tenure_months": 96,
        "default_tenure_months": 60,
        "default_amount": 800000.0,
        "processing_fee_pct": 1.0,
        "collateral_type": "Secured (New Vehicle Hypothecation)",
        "description": "Secured auto financing facility against hypothecation of brand-new passenger car.",
        "pricing_rationale": "Illustrative base demonstration rate at 9.0%. Benchmark market corridor: 7.85% - 10.0%.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Used Car Loan": {
        "product_name": "Used Car Loan",
        "default_illustrative_rate": 12.0,
        "min_interest_rate": 10.0,
        "max_interest_rate": 14.0,
        "min_tenure_months": 12,
        "max_tenure_months": 60,
        "default_tenure_months": 36,
        "default_amount": 400000.0,
        "processing_fee_pct": 1.5,
        "collateral_type": "Secured (Pre-Owned Vehicle Hypothecation)",
        "description": "Financing facility for certified pre-owned passenger vehicles with age-based risk margin.",
        "pricing_rationale": "Illustrative base demonstration rate at 12.0%. Benchmark market corridor: 10.0% - 14.0%.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Education Loan": {
        "product_name": "Education Loan",
        "default_illustrative_rate": 10.5,
        "min_interest_rate": 8.0,
        "max_interest_rate": 14.0,
        "min_tenure_months": 24,
        "max_tenure_months": 180,
        "default_tenure_months": 84,
        "default_amount": 1200000.0,
        "processing_fee_pct": 0.75,
        "collateral_type": "Semi-Secured / Co-applicant Guarantee",
        "description": "Academic finance facility covering tuition, travel, and campus living expenses for domestic or overseas study.",
        "pricing_rationale": "Illustrative base demonstration rate at 10.5%. Benchmark market corridor: 8.0% - 14.0%.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Personal Loan": {
        "product_name": "Personal Loan",
        "default_illustrative_rate": 15.0,
        "min_interest_rate": 11.0,
        "max_interest_rate": 22.0,
        "min_tenure_months": 12,
        "max_tenure_months": 84,
        "default_tenure_months": 36,
        "default_amount": 300000.0,
        "processing_fee_pct": 1.5,
        "collateral_type": "Unsecured",
        "description": "Unsecured retail credit for medical emergencies, personal travel, or immediate liquidity needs.",
        "pricing_rationale": "Illustrative base demonstration rate at 15.0%. Benchmark market corridor: 11.0% - 22.0%.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Business Loan": {
        "product_name": "Business Loan",
        "default_illustrative_rate": 14.0,
        "min_interest_rate": 11.0,
        "max_interest_rate": 18.0,
        "min_tenure_months": 12,
        "max_tenure_months": 120,
        "default_tenure_months": 48,
        "default_amount": 1500000.0,
        "processing_fee_pct": 2.0,
        "collateral_type": "Commercial / Working Capital Floating Charge",
        "description": "Commercial loan facility for MSME working capital, machinery procurement, or enterprise scaling.",
        "pricing_rationale": "Illustrative base demonstration rate at 14.0%. Benchmark commercial MSME corridor: 11.0% - 18.0%.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Two-Wheeler Loan": {
        "product_name": "Two-Wheeler Loan",
        "default_illustrative_rate": 13.0,
        "min_interest_rate": 10.0,
        "max_interest_rate": 16.0,
        "min_tenure_months": 12,
        "max_tenure_months": 60,
        "default_tenure_months": 36,
        "default_amount": 100000.0,
        "processing_fee_pct": 1.5,
        "collateral_type": "Secured (Motorcycle/Scooter Hypothecation)",
        "description": "Secured financing facility for personal two-wheelers, motorcycles, and electric scooters.",
        "pricing_rationale": "Illustrative base demonstration rate at 13.0%. Benchmark two-wheeler corridor: 10.0% - 16.0%.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    },
    "Gold Loan": {
        "product_name": "Gold Loan",
        "default_illustrative_rate": 10.0,
        "min_interest_rate": 8.5,
        "max_interest_rate": 13.0,
        "min_tenure_months": 6,
        "max_tenure_months": 36,
        "default_tenure_months": 12,
        "default_amount": 200000.0,
        "processing_fee_pct": 0.5,
        "collateral_type": "Fully Secured (Physical Gold Jewelry Pledge)",
        "description": "Instant secured credit facility against pledged physical gold jewelry and ornaments.",
        "pricing_rationale": "Illustrative base demonstration rate at 10.0%. Benchmark bullion pledge corridor: 8.5% - 13.0%.",
        "disclaimer": "Demonstration benchmark rate. Does not constitute a binding bank offer or guaranteed clearance."
    }
}

# Alias for legacy compatibility
PRODUCT_PRICING_CATALOG["Car Loan"] = PRODUCT_PRICING_CATALOG["New Car Loan"]


def get_product_pricing_config(loan_product: str) -> Dict[str, Any]:
    """
    Retrieves pricing rules and boundaries for a given loan product.
    Falls back to Personal Loan if unrecognized.
    """
    if loan_product in PRODUCT_PRICING_CATALOG:
        return PRODUCT_PRICING_CATALOG[loan_product]
    return PRODUCT_PRICING_CATALOG["Personal Loan"]


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
    - Automatically assigns the configured demonstration rate for the selected loan product.
    - Rate is read-only for applicants (no arbitrary self-assignment).
    - If adverse credit history is detected, applies documented risk spread within limits.
    
    In Underwriter / Simulation Override Mode:
    - Allows authorized sensitivity testing and stress-testing override within product boundaries.
    - Explicitly flagged as simulation mode.
    
    Returns:
    - applicable_annual_rate: float
    - base_rate: float
    - risk_adjustments: List[Dict[str, Any]]
    - min_allowed_rate: float
    - max_allowed_rate: float
    - is_simulation: bool
    - pricing_source: str
    - pricing_rationale: str
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
                "factor": "Underwriter Simulation Override",
                "adjustment_pct": round(clamped_override - base_rate, 2),
                "rationale": "Manual rate stress-test scenario by authorized underwriter."
            }],
            "min_allowed_rate": min_rate,
            "max_allowed_rate": max_rate,
            "is_simulation": True,
            "pricing_source": "Underwriter Simulation Override (Scenario Testing Mode)",
            "pricing_rationale": f"Rate manually adjusted to {clamped_override:.2f}% for scenario stress-testing.",
            "disclaimer": "⚠️ Simulated test rate. Not an applicant-assigned or binding lender rate."
        }

    # Normal Applicant Workflow: Automated Pricing Determination Rule
    adjustments: List[Dict[str, Any]] = []
    current_rate = base_rate

    # Rule 1: Credit History Risk Loading
    # Applicants without clean track record (Credit_History = 0.0) face adverse credit spread
    if float(credit_history) == 0.0:
        if loan_product in ["Personal Loan", "Business Loan", "Used Car Loan"]:
            credit_spread = 2.5
        else:
            credit_spread = 1.0
        current_rate += credit_spread
        adjustments.append({
            "factor": "Adverse Credit Track Record Loading",
            "adjustment_pct": credit_spread,
            "rationale": "Past credit defaults warrant institutional risk margin within product cap."
        })
    else:
        adjustments.append({
            "factor": "Standard Credit Guidelines Met",
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
            "rationale": "Unsalaried variable revenue profile requires moderate liquidity margin."
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
        "pricing_source": f"FinVeda Institutional Pricing Engine ({loan_product} Illustrative Matrix)",
        "pricing_rationale": (
            f"Illustrative demonstration base rate: {base_rate:.2f}% (benchmark range: {min_rate}% – {max_rate}%). "
            f"Adjusted rate: {final_rate:.2f}%."
        ),
        "disclaimer": "Demonstration benchmark rate informed by retail banking spreads. Not a guaranteed lender offer."
    }
