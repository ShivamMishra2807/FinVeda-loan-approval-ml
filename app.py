"""
FinVeda | AI Credit Underwriting Platform
Modern, High-Performance Banking & Financial Planning Platform.
Includes 6 dedicated modules:
1. Executive Dashboard
2. Loan Approval Prediction
3. EMI Calculator (Reducing-balance formula, verified)
4. Loan Comparison (Up to 3 custom scenarios)
5. Repayment Schedule (Monthly amortization & CSV export)
6. Loan Affordability Analysis (DTI analysis & cash-flow health)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

from src.preprocessing import clean_raw_data, add_engineered_features, get_feature_names
from src.prediction import LoanPredictionService
from src.financial import (
    calculate_emi,
    generate_amortization_schedule,
    check_loan_affordability,
    compare_loan_scenarios,
    LOAN_CATEGORIES
)
from src.pricing import (
    PRODUCT_PRICING_CATALOG,
    determine_interest_rate,
    get_product_pricing_config
)


# Page Configuration - Set to Wide Layout
st.set_page_config(
    page_title="FinVeda | AI Credit Underwriting & Planning",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Design Styling (Modern Fintech UI: Clean, Sleek, Professional)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background & Global Palette */
    .stApp {
        background: radial-gradient(circle at 10% 15%, #0f172a 0%, #080d1a 100%);
        color: #f1f5f9;
    }
    
    /* Reduce default top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 95% !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #0ea5e9, #38bdf8, #6366f1);
        opacity: 0.85;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 14px 40px 0 rgba(14, 165, 233, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .metric-val {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 0.78rem;
        font-weight: 500;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(23, 32, 51, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    
    /* Decision Banner Styles */
    .result-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.28) 100%);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 26px 22px;
        text-align: center;
        box-shadow: 0 10px 32px rgba(16, 185, 129, 0.25);
    }
    .result-rejected {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(220, 38, 38, 0.28) 100%);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 26px 22px;
        text-align: center;
        box-shadow: 0 10px 32px rgba(239, 68, 68, 0.25);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.25s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.6) !important;
    }
    
    /* Custom badge tags */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-blue {
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.35);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-red {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_dataset.csv")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "model_metrics.csv")
REPORTS_PATH = os.path.join(BASE_DIR, "reports", "classification_reports.json")
SUMMARY_PATH = os.path.join(BASE_DIR, "reports", "dataset_summary.json")
ENSEMBLE_PATH = os.path.join(BASE_DIR, "models", "voting_ensemble_pipeline.joblib")
RF_PATH = os.path.join(BASE_DIR, "models", "random_forest_pipeline.joblib")

@st.cache_data
def load_cached_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_data
def load_cached_metrics():
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH)
    return None

@st.cache_data
def load_cached_reports():
    if os.path.exists(REPORTS_PATH):
        with open(REPORTS_PATH, "r") as f:
            return json.load(f)
    return None

@st.cache_resource
def get_prediction_engine():
    if os.path.exists(ENSEMBLE_PATH):
        return LoanPredictionService(ENSEMBLE_PATH)
    return None

# Formatting Helper for Indian Rupees (₹)
def format_inr(number: float) -> str:
    """Format numbers into Indian Rupee style or compact Lakhs/Crores."""
    if abs(number) >= 10000000:
        return f"₹{number/10000000:.2f} Cr"
    elif abs(number) >= 100000:
        return f"₹{number/100000:.2f} Lakh"
    else:
        return f"₹{number:,.0f}"

# ------------------------------------------------------------------------------------------
# BRAND HEADER & SLEEK HORIZONTAL NAVIGATION BAR
# ------------------------------------------------------------------------------------------
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 4px 14px 4px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 1.8rem;">🏦</span>
        <div>
            <h1 style="margin: 0; font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                FINVEDA AI
            </h1>
            <p style="margin: 0; color: #94a3b8; font-size: 0.8rem; font-weight: 500;">
                AI-Powered Loan Approval & Advanced Financial Planning Platform (₹ INR)
            </p>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <span class="badge-pill badge-blue">⚡ Soft-Voting Ensemble</span>
        <span class="badge-pill badge-green">91.25% Benchmark Acc</span>
    </div>
</div>
""", unsafe_allow_html=True)

selected_page = option_menu(
    menu_title=None,
    options=[
        "Home / Dashboard",
        "Loan Approval Prediction",
        "EMI Calculator",
        "Loan Comparison",
        "Repayment Schedule",
        "Loan Affordability Analysis"
    ],
    icons=[
        "speedometer2",
        "shield-check",
        "calculator",
        "bar-chart-steps",
        "calendar3",
        "wallet2"
    ],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "6px 10px",
            "background-color": "rgba(15, 23, 42, 0.8)",
            "border-radius": "16px",
            "border": "1px solid rgba(255, 255, 255, 0.08)",
            "margin-bottom": "22px",
            "box-shadow": "0 8px 24px rgba(0, 0, 0, 0.25)"
        },
        "icon": {"color": "#38bdf8", "font-size": "14px"},
        "nav-link": {
            "font-size": "13px",
            "font-weight": "600",
            "text-align": "center",
            "margin": "0px 3px",
            "padding": "10px 12px",
            "border-radius": "12px",
            "color": "#94a3b8",
            "transition": "all 0.2s ease"
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, #0284c7 0%, #2563eb 100%)",
            "color": "#ffffff",
            "box-shadow": "0 4px 14px rgba(37, 99, 235, 0.45)"
        }
    }
)

# ------------------------------------------------------------------------------------------
# PAGE 1: HOME / DASHBOARD
# ------------------------------------------------------------------------------------------
if selected_page == "Home / Dashboard":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span class="badge-pill badge-green">Live Portfolio Statistics</span>
            <span style="color: #64748b; font-size: 0.82rem;">Currency Standard: Indian Rupees (₹)</span>
        </div>
        <h2 style="color: #ffffff; font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">Executive Underwriting Overview</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">Aggregated sanction metrics, portfolio volumes, debt burden ratios, and risk distributions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    df_raw = load_cached_dataset()
    if df_raw is None:
        st.warning("⚠️ Dataset not detected. Please generate the dataset from the repository root.")
    else:
        df_clean = clean_raw_data(df_raw)
        
        # Key Metrics Row
        total_apps = len(df_clean)
        approved_count = (df_clean["Loan_Status"] == "Y").sum()
        rejected_count = (df_clean["Loan_Status"] == "N").sum()
        approval_rate = (approved_count / total_apps) * 100
        median_loan_amt_inr = df_clean["LoanAmount"].median() * 1000
        median_income_inr = df_clean["ApplicantIncome"].median()
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Applications</div>
                <div class="metric-val">{total_apps:,}</div>
                <div class="metric-sub">Cleaned records</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Sanctioned</div>
                <div class="metric-val" style="color:#10b981;">{approved_count:,}</div>
                <div class="metric-sub">{approval_rate:.1f}% portfolio share</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Declined</div>
                <div class="metric-val" style="color:#ef4444;">{rejected_count:,}</div>
                <div class="metric-sub">{100-approval_rate:.1f}% portfolio share</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Approval Ratio</div>
                <div class="metric-val">{approval_rate:.1f}%</div>
                <div class="metric-sub">Clearance rate</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Median Loan</div>
                <div class="metric-val">{format_inr(median_loan_amt_inr)}</div>
                <div class="metric-sub">Requested principal</div>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Median Income</div>
                <div class="metric-val">{format_inr(median_income_inr)}</div>
                <div class="metric-sub">Monthly baseline</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_credit = px.histogram(
                df_clean, x="Credit_History", color="Loan_Status",
                barmode="group",
                title="<b>Approval Rate by CIBIL / Credit History Track Record</b>",
                labels={"Credit_History": "Credit History (1=Clean Record / Meets Guidelines, 0=Default)", "count": "Applications"},
                color_discrete_map={"Y": "#10b981", "N": "#ef4444"},
                template="plotly_dark"
            )
            fig_credit.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_credit, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_prop = px.histogram(
                df_clean, x="Property_Area", color="Loan_Status",
                barmode="group",
                title="<b>Loan Approvals Across Property Geography</b>",
                labels={"Property_Area": "Collateral Property Area", "count": "Applications"},
                color_discrete_map={"Y": "#0ea5e9", "N": "#f43f5e"},
                template="plotly_dark"
            )
            fig_prop.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_prop, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            df_clean_sample = df_clean.sample(min(800, len(df_clean)), random_state=42).copy()
            df_clean_sample["LoanAmount_INR_Lakhs"] = df_clean_sample["LoanAmount"] / 100.0
            
            fig_scatter = px.scatter(
                df_clean_sample,
                x="ApplicantIncome", y="LoanAmount_INR_Lakhs", color="Loan_Status",
                title="<b>Applicant Monthly Income (₹) vs Requested Loan (₹ in Lakhs)</b>",
                labels={"ApplicantIncome": "Applicant Monthly Income (₹)", "LoanAmount_INR_Lakhs": "Loan Amount (₹ Lakhs)"},
                color_discrete_map={"Y": "#10b981", "N": "#ef4444"},
                opacity=0.75,
                hover_data=["Education", "Property_Area"],
                template="plotly_dark"
            )
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_edu = px.histogram(
                df_clean, x="Education", color="Loan_Status",
                barmode="group",
                title="<b>Approval Decisions by Educational Qualification</b>",
                labels={"Education": "Applicant Qualification", "count": "Applications"},
                color_discrete_map={"Y": "#38bdf8", "N": "#fb7185"},
                template="plotly_dark"
            )
            fig_edu.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_edu, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------
# PAGE 2: LOAN APPROVAL PREDICTION
# ------------------------------------------------------------------------------------------
elif selected_page == "Loan Approval Prediction":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span class="badge-pill badge-blue">ML Underwriting Engine</span>
            <span style="color: #64748b; font-size: 0.82rem;">Trained Soft-Voting Ensemble (Random Forest + Extra Trees + KNN)</span>
        </div>
        <h2 style="color: #ffffff; font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">Loan Approval Prediction</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">Evaluate prospective borrower metrics and obtain calibrated approval probabilities with transparent factor explanations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    predictor = get_prediction_engine()
    col_form, col_result = st.columns([1.2, 1])
    
    with col_form:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-size:1.2rem; font-weight:700; color:#f8fafc;'>📋 Applicant Profile & Financial Parameters</h3>", unsafe_allow_html=True)
        
        # Loan Product Selector for institutional illustrative defaults
        available_products = [
            "Home Loan",
            "New Car Loan",
            "Used Car Loan",
            "Education Loan",
            "Personal Loan",
            "Business Loan",
            "Two-Wheeler Loan",
            "Gold Loan"
        ]
        loan_cat = st.selectbox(
            "Select Loan Product",
            available_products,
            help="Configures standard tenure range, illustrative pricing corridor, and financial terms."
        )
        cat_info = LOAN_CATEGORIES[loan_cat]
        pricing_cfg = get_product_pricing_config(loan_cat)
        st.caption(f"ℹ️ *{cat_info['description']} (Illustrative demonstration benchmark)*")
        
        st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:12px;'><strong>Demographic & Employment Details</strong></p>", unsafe_allow_html=True)
        c_g, c_m, c_d = st.columns(3)
        with c_g:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c_m:
            married = st.selectbox("Marital Status", ["Yes", "No"])
        with c_d:
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            
        c_e, c_se, c_p = st.columns(3)
        with c_e:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with c_se:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        with c_p:
            property_area = st.selectbox("Collateral Property Area", ["Semiurban", "Urban", "Rural"])
            
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)
        st.markdown("<p style='color:#38bdf8; font-size:0.85rem; margin-bottom:12px;'><strong>Monthly Incomes & Requested Facility (₹ INR)</strong></p>", unsafe_allow_html=True)
        
        c_ai, c_cai, c_deb = st.columns(3)
        with c_ai:
            applicant_income = st.number_input(
                "Applicant Monthly Income (₹)",
                min_value=5000,
                max_value=1000000,
                value=45000,
                step=2500,
                help="Gross monthly salary or business receipts in Rupees (₹)"
            )
            st.caption(f"**{format_inr(applicant_income)} / month**")
        with c_cai:
            coapplicant_income = st.number_input(
                "Co-Applicant Monthly Income (₹)",
                min_value=0,
                max_value=500000,
                value=15000,
                step=2500,
                help="Spouse or co-borrower monthly income in Rupees (₹)"
            )
            st.caption(f"**{format_inr(coapplicant_income)} / month**")
        with c_deb:
            existing_debt = st.number_input(
                "Existing Monthly EMIs (₹)",
                min_value=0,
                max_value=300000,
                value=0,
                step=1000,
                help="Current ongoing debt obligations and loan EMIs in Rupees (₹)"
            )
            st.caption(f"**{format_inr(existing_debt)} / month**")
            
        c_la, c_lat = st.columns(2)
        with c_la:
            default_amount_inr = int(cat_info["default_amount"])
            loan_amount_inr = st.number_input(
                "Requested Loan Amount (₹ INR)",
                min_value=50000,
                max_value=50000000,
                value=default_amount_inr,
                step=25000,
                help="Enter the full loan amount in Indian Rupees. Example: ₹16,25,000 for 16.25 Lakhs."
            )
            loan_amount_k = loan_amount_inr / 1000.0  # Convert to thousands for ML model (dataset scale)
            actual_loan_rupees = loan_amount_inr
            st.caption(f"Principal: **{format_inr(actual_loan_rupees)}**")
        with c_lat:
            suggested_months = pricing_cfg.get("default_tenure_months", cat_info["default_tenure_years"] * 12)
            min_m = pricing_cfg.get("min_tenure_months", 12)
            max_m = pricing_cfg.get("max_tenure_months", 360)
            standard_terms = [360, 240, 180, 120, 84, 60, 48, 36, 24, 12, 6]
            term_options = [m for m in standard_terms if min_m <= m <= max_m]
            if suggested_months not in term_options:
                term_options.append(suggested_months)
            term_options = sorted(list(set(term_options)), reverse=True)
            default_idx = term_options.index(suggested_months) if suggested_months in term_options else 0
            loan_term = st.selectbox(
                "Repayment Tenure",
                term_options,
                index=default_idx,
                format_func=lambda m: f"{m} Months ({m//12} Yrs)" if m >= 12 else f"{m} Months"
            )
            
        c_ch, c_disp_rate = st.columns(2)
        with c_ch:
            credit_history = st.selectbox(
                "CIBIL / Credit History Record",
                [1.0, 0.0],
                format_func=lambda x: "1.0 - Meets Guidelines (Clean)" if x == 1.0 else "0.0 - Derogatory / Delinquent"
            )
            custom_rate_toggle = st.checkbox(
                "⚙️ Customize Interest Rate",
                value=False,
                help="Check to manually adjust the annual interest rate for this assessment"
            )
        with c_disp_rate:
            base_r = pricing_cfg["default_illustrative_rate"]
            min_r = pricing_cfg["min_interest_rate"]
            max_r = pricing_cfg["max_interest_rate"]
            
            # Pre-calculate indicative rate for display in read-only box
            indicative_res = determine_interest_rate(
                loan_product=loan_cat,
                credit_history=credit_history,
                is_self_employed=(self_employed == "Yes"),
                is_simulation_mode=False
            )
            disp_rate = indicative_res["applicable_annual_rate"]
            
            if custom_rate_toggle:
                chosen_rate = st.number_input(
                    "Annual Interest Rate (%)",
                    min_value=1.0,
                    max_value=35.0,
                    value=float(disp_rate),
                    step=0.25,
                    help=f"Benchmark corridor for {loan_cat}: {min_r}% – {max_r}% p.a."
                )
                effective_rate = float(chosen_rate)
                badge_text = "Custom User Rate"
                badge_class = "badge-amber"
            else:
                effective_rate = float(disp_rate)
                badge_text = "Read-Only Institutional Rate"
                badge_class = "badge-blue"
            
            st.markdown(f"""
            <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 10px 14px; margin-top: 2px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Applicable Annual Rate</span>
                    <span class="badge-pill {badge_class}" style="font-size: 0.68rem; padding: 3px 8px;">{badge_text}</span>
                </div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #38bdf8; margin: 3px 0;">{effective_rate:.2f}% p.a.</div>
                <div style="font-size: 0.74rem; color: #64748b;">
                    Corridor: {min_r}% – {max_r}% ({loan_cat})
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        submit_btn = st.button("⚡ Run Institutional Assessment", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_result:

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-size:1.2rem; font-weight:700; color:#f8fafc;'>📈 Underwriting Decision Output</h3>", unsafe_allow_html=True)
        
        input_payload = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount_k,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }
        
        current_input_signature = (
            gender, married, dependents, education, self_employed, property_area,
            applicant_income, coapplicant_income, existing_debt,
            loan_amount_inr, loan_term, credit_history, loan_cat,
            custom_rate_toggle, effective_rate
        )
        
        if submit_btn:
            if predictor is None:
                st.error("⚠️ Trained model artifact not found. Please train models first via `python -m src.models`.")
            else:
                try:
                    res = predictor.predict(
                        input_payload,
                        annual_interest_rate=effective_rate if custom_rate_toggle else None,
                        existing_monthly_debt=existing_debt,
                        loan_product=loan_cat
                    )
                    st.session_state["assessment_result"] = res
                    st.session_state["assessment_signature"] = current_input_signature
                    st.session_state["assessment_payload"] = input_payload
                except Exception as e:
                    st.error(f"Inference error encountered: {str(e)}")
                    st.session_state["assessment_result"] = None
                    
        has_valid_assessment = (
            st.session_state.get("assessment_signature") == current_input_signature
            and st.session_state.get("assessment_result") is not None
        )
        
        if not has_valid_assessment:
            st.info("👈 Enter applicant's profile and click **Run Institutional Assessment** to execute the trained ensemble pipeline.")
        else:
            res = st.session_state["assessment_result"]
            is_appr = res["is_approved"]
            conf_appr = res["approval_confidence"] * 100
            conf_rej = res["rejection_confidence"] * 100
            fin = res["financial_indicators"]
                    
            if is_appr:
                st.markdown(f"""
                <div class="result-approved">
                    <span class="badge-pill badge-green" style="margin-bottom: 8px;">ASSESSMENT OUTCOME</span>
                    <h2 style="color: #10b981; margin: 6px 0; font-size: 2rem; font-weight: 800;">PREDICTED APPROVAL</h2>
                    <p style="color: #a7f3d0; margin-top: 4px; font-weight: 500; font-size: 0.95rem;">
                        Model Clearance Probability: <strong>{conf_appr:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-rejected">
                    <span class="badge-pill badge-red" style="margin-bottom: 8px;">CREDIT RISK ALERT</span>
                    <h2 style="color: #ef4444; margin: 6px 0; font-size: 2rem; font-weight: 800;">PREDICTED REJECTION</h2>
                    <p style="color: #fca5a5; margin-top: 4px; font-weight: 500; font-size: 0.95rem;">
                        Underwriting Risk Confidence: <strong>{conf_rej:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interest Rate Source & Pricing Transparency Banner
            rate_badge_display = '<span class="badge-pill badge-amber">✓ Custom Rate Applied</span>' if custom_rate_toggle else '<span class="badge-pill badge-blue">✓ Institutional Pricing Rule</span>'
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px 16px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Applicable Interest Rate</span>
                    {rate_badge_display}
                </div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #38bdf8;">{fin['annual_interest_rate']:.2f}% p.a.</div>
                <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">
                    <strong>Source:</strong> {fin.get('pricing_source', 'Institutional Pricing Rules')}<br>
                    <strong>Details:</strong> {fin.get('pricing_rationale', 'Evaluated based on product pricing matrix and risk profile.')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h4 style='color:#f8fafc; font-size:1rem; font-weight:600;'>Financial Affordability & Cash-Flow Diagnostics</h4>", unsafe_allow_html=True)
            
            r1, r2 = st.columns(2)
            r1.metric("Proposed Monthly EMI", f"₹{fin['proposed_emi']:,.2f}", help="Calculated via reducing-balance amortization formula with interest")
            r2.metric("Total Monthly Debt Service", f"₹{fin['total_monthly_debt']:,.2f}", help="Existing Monthly EMIs + Proposed EMI")
            
            r3, r4 = st.columns(2)
            dti_val = fin['debt_to_income_pct']
            dti_delta = "Healthy" if dti_val <= 35 else ("Moderate" if dti_val <= 50 else "High Risk")
            r3.metric("Debt-to-Income (DTI)", f"{dti_val:.1f}%", delta=dti_delta, delta_color="inverse" if dti_val > 50 else "normal", help="Formula: (Existing EMIs + Proposed EMI) / Monthly Income × 100")
            r4.metric("Loan-to-Income (LTI)", f"{fin['loan_to_income_ratio']:.2f}x", help="Formula: Principal / Annual Gross Household Income")
            
            r5, r6 = st.columns(2)
            r5.metric("Total Interest Payable", format_inr(fin['total_interest']), help="Total interest over full tenure")
            r6.metric("Total Repayment Amount", format_inr(fin['total_repayment']), help="Principal + Total Interest")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf_appr,
                title={'text': "Approval Probability Index (%)", 'font': {'size': 14, 'color': "#94a3b8"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#cbd5e1"},
                    'bar': {'color': "#10b981" if is_appr else "#ef4444"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [50, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "#38bdf8", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=190, margin=dict(t=25, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Explainable Factor Attribution & Suggestions
            st.markdown("<h4 style='color:#f8fafc; font-size:0.95rem; font-weight:600; margin-top:10px;'>Factor Attribution</h4>", unsafe_allow_html=True)
            for exp in res.get("explanations", []):
                st.markdown(f"• <span style='font-size:0.85rem; color:#cbd5e1;'>{exp}</span>", unsafe_allow_html=True)
                
            st.markdown("<h4 style='color:#f8fafc; font-size:0.95rem; font-weight:600; margin-top:10px;'>Educational Guidance</h4>", unsafe_allow_html=True)
            for sug in res.get("educational_suggestions", []):
                st.markdown(f"💡 <span style='font-size:0.85rem; color:#94a3b8;'>{sug}</span>", unsafe_allow_html=True)
            
            st.caption(f"<br>ℹ️ *{res['limitations_note']}*", unsafe_allow_html=True)
            
            pred_export_df = pd.DataFrame([{
                **input_payload,
                "Loan_Category": loan_cat,
                "Pricing_Mode": "Custom Rate Specification" if custom_rate_toggle else "Automated Institutional Pricing",
                "Annual_Interest_Rate_Pct": fin['annual_interest_rate'],
                "ApplicantIncome_INR": applicant_income,
                "CoapplicantIncome_INR": coapplicant_income,
                "Existing_Monthly_Debt_INR": existing_debt,
                "LoanAmount_INR": actual_loan_rupees,
                "Proposed_EMI_INR": fin['proposed_emi'],
                "Total_Interest_INR": fin['total_interest'],
                "Total_Repayment_INR": fin['total_repayment'],
                "DTI_Pct": fin['debt_to_income_pct'],
                "LTI_Ratio": fin['loan_to_income_ratio'],
                "Status": res["prediction_status"],
                "Approval_Probability": f"{conf_appr:.2f}%"
            }])
            csv_bytes = pred_export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Assessment Memo (CSV)",
                data=csv_bytes,
                file_name="loan_sanction_assessment.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------
# PAGE 3: EMI CALCULATOR
# ------------------------------------------------------------------------------------------
elif selected_page == "EMI Calculator":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span class="badge-pill badge-blue">Standard Reducing-Balance Formula</span>
            <span style="color: #64748b; font-size: 0.82rem;">EMI = P × r × (1 + r)ⁿ / ((1 + r)ⁿ − 1)</span>
        </div>
        <h2 style="color: #ffffff; font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">Interactive EMI Calculator</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">Compute verified monthly instalments, total interest burden, and fee-inclusive borrowing costs dynamically.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_output = st.columns([1.1, 1.2])
    
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-size:1.2rem; font-weight:700; color:#f8fafc;'>⚙️ Loan Parameters</h3>", unsafe_allow_html=True)
        
        selected_cat = st.selectbox(
            "Loan Category",
            list(LOAN_CATEGORIES.keys()),
            key="emi_cat"
        )
        cat_defaults = LOAN_CATEGORIES[selected_cat]
        st.caption(f"ℹ️ *{cat_defaults['description']} (Illustrative market benchmark)*")
        
        loan_amount_input = st.number_input(
            "Loan Principal Amount (₹ INR)",
            min_value=10000.0,
            max_value=100000000.0,
            value=float(cat_defaults["default_amount"]),
            step=50000.0,
            format="%.0f",
            help="Principal amount borrowed in Indian Rupees."
        )
        st.caption(f"Principal formatted: **{format_inr(loan_amount_input)}**")
        
        interest_rate_input = st.slider(
            "Annual Interest Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=float(cat_defaults["default_interest_rate"]),
            step=0.25,
            help="Annual rate of interest. Monthly rate is Annual Rate / 12."
        )
        st.caption(f"Equivalent Monthly Rate: **{interest_rate_input / 12.0:.4f}% per month**")
        
        tenure_unit = st.radio("Tenure Specification", ["Years", "Months"], horizontal=True)
        if tenure_unit == "Years":
            tenure_years = st.slider(
                "Tenure (Years)",
                min_value=1,
                max_value=30,
                value=int(cat_defaults["default_tenure_years"]),
                step=1
            )
            tenure_months_calc = tenure_years * 12
        else:
            tenure_months_calc = st.slider(
                "Tenure (Months)",
                min_value=1,
                max_value=360,
                value=int(cat_defaults["default_tenure_years"] * 12),
                step=6
            )
            
        with st.expander("Additional Fees & Charges (Optional)"):
            fee_pct_input = st.number_input(
                "Processing Fee (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(cat_defaults["processing_fee_pct"]),
                step=0.25
            )
            extra_charges_input = st.number_input(
                "Other Charges (₹ INR) (Documentation/Legal)",
                min_value=0.0,
                max_value=200000.0,
                value=2500.0,
                step=500.0
            )
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_output:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-size:1.2rem; font-weight:700; color:#f8fafc;'>📊 Repayment Summary</h3>", unsafe_allow_html=True)
        
        emi_res = calculate_emi(
            principal=loan_amount_input,
            annual_interest_rate=interest_rate_input,
            tenure_months=tenure_months_calc,
            processing_fee_pct=fee_pct_input,
            additional_charges=extra_charges_input
        )
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Monthly EMI</div>
                <div class="metric-val" style="color: #38bdf8;">{format_inr(emi_res['monthly_emi'])}</div>
                <div class="metric-sub">₹{emi_res['monthly_emi']:,.2f} / month</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Repayment</div>
                <div class="metric-val" style="color: #10b981;">{format_inr(emi_res['total_repayment'])}</div>
                <div class="metric-sub">Principal + Total Interest</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        m3, m4, m5 = st.columns(3)
        with m3:
            st.metric("Loan Principal", format_inr(emi_res["principal"]))
        with m4:
            st.metric("Total Interest", format_inr(emi_res["total_interest"]))
        with m5:
            st.metric("Estimated Fees", format_inr(emi_res["processing_fees"] + emi_res["additional_charges"]))
            
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)
        st.metric(
            label="Total Estimated Borrowing Cost (Fee-Inclusive)",
            value=format_inr(emi_res["total_cost_with_fees"]),
            help="Total repayment amount plus upfront processing fees and additional documentation charges."
        )
        
        # Donut Chart Breakdown of Principal vs Interest
        if emi_res["total_repayment"] > 0:
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Principal Amount", "Total Interest Payable"],
                values=[emi_res["principal"], emi_res["total_interest"]],
                hole=0.6,
                marker=dict(colors=["#0284c7", "#f43f5e"]),
                textinfo="label+percent",
                hoverinfo="value+percent"
            )])
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=220,
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        st.caption("ℹ️ *Total repayment amount strictly equals Principal + Total Interest. Applicable fees are shown as a separate total cost figure.*")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------
# PAGE 4: LOAN COMPARISON
# ------------------------------------------------------------------------------------------
elif selected_page == "Loan Comparison":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span class="badge-pill badge-blue">Scenario Trade-Off Engine</span>
            <span style="color: #64748b; font-size: 0.82rem;">Compare up to 3 loan options side-by-side</span>
        </div>
        <h2 style="color: #ffffff; font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">Multi-Loan Scenario Comparison</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">Evaluate tenure vs interest trade-offs: longer tenures reduce monthly EMI but increase lifetime interest payout.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; font-size:1.2rem; font-weight:700; color:#f8fafc;'>⚙️ Customize Loan Scenarios</h3>", unsafe_allow_html=True)
    
    s_col1, s_col2, s_col3 = st.columns(3)
    
    with s_col1:
        st.markdown("##### 🔹 Scenario A (Short Term)")
        p_a = st.number_input("Principal (₹)", min_value=50000.0, max_value=50000000.0, value=1000000.0, step=50000.0, key="p_a")
        r_a = st.number_input("Annual Rate (%)", min_value=0.0, max_value=25.0, value=9.5, step=0.25, key="r_a")
        t_a = st.slider("Tenure (Years)", min_value=1, max_value=30, value=5, key="t_a")
        fee_a = st.number_input("Fee (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.25, key="fee_a")
        
    with s_col2:
        st.markdown("##### 🔹 Scenario B (Medium Term)")
        p_b = st.number_input("Principal (₹)", min_value=50000.0, max_value=50000000.0, value=1000000.0, step=50000.0, key="p_b")
        r_b = st.number_input("Annual Rate (%)", min_value=0.0, max_value=25.0, value=9.0, step=0.25, key="r_b")
        t_b = st.slider("Tenure (Years)", min_value=1, max_value=30, value=10, key="t_b")
        fee_b = st.number_input("Fee (%)", min_value=0.0, max_value=5.0, value=0.75, step=0.25, key="fee_b")
        
    with s_col3:
        st.markdown("##### 🔹 Scenario C (Long Term)")
        p_c = st.number_input("Principal (₹)", min_value=50000.0, max_value=50000000.0, value=1000000.0, step=50000.0, key="p_c")
        r_c = st.number_input("Annual Rate (%)", min_value=0.0, max_value=25.0, value=8.75, step=0.25, key="r_c")
        t_c = st.slider("Tenure (Years)", min_value=1, max_value=30, value=15, key="t_c")
        fee_c = st.number_input("Fee (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.25, key="fee_c")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Run Comparison
    scenarios_list = [
        {"name": "Scenario A (5 Yrs)", "principal": p_a, "annual_interest_rate": r_a, "tenure_months": t_a * 12, "processing_fee_pct": fee_a},
        {"name": "Scenario B (10 Yrs)", "principal": p_b, "annual_interest_rate": r_b, "tenure_months": t_b * 12, "processing_fee_pct": fee_b},
        {"name": "Scenario C (15 Yrs)", "principal": p_c, "annual_interest_rate": r_c, "tenure_months": t_c * 12, "processing_fee_pct": fee_c}
    ]
    comp_df = compare_loan_scenarios(scenarios_list)
    
    # Highlights
    lowest_emi_row = comp_df.loc[comp_df["Monthly EMI (₹)"].idxmin()]
    lowest_cost_row = comp_df.loc[comp_df["Total Cost with Fees (₹)"].idxmin()]
    
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #38bdf8;">
            <div style="color:#94a3b8; font-size:0.8rem; font-weight:600; text-transform:uppercase;">Lowest Monthly Outflow (Lowest EMI)</div>
            <div style="font-size:1.6rem; font-weight:800; color:#38bdf8; margin:4px 0;">{lowest_emi_row['Scenario']}</div>
            <div style="color:#e2e8f0; font-size:0.9rem;">Monthly EMI: <strong>{format_inr(lowest_emi_row['Monthly EMI (₹)'])}</strong></div>
            <p style="color:#94a3b8; font-size:0.78rem; margin-top:6px;">Suitable if minimizing monthly cash-flow strain is the primary priority.</p>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #10b981;">
            <div style="color:#94a3b8; font-size:0.8rem; font-weight:600; text-transform:uppercase;">Lowest Lifetime Cost (Least Interest)</div>
            <div style="font-size:1.6rem; font-weight:800; color:#10b981; margin:4px 0;">{lowest_cost_row['Scenario']}</div>
            <div style="color:#e2e8f0; font-size:0.9rem;">Total Cost: <strong>{format_inr(lowest_cost_row['Total Cost with Fees (₹)'])}</strong></div>
            <p style="color:#94a3b8; font-size:0.78rem; margin-top:6px;">Suitable if saving overall interest expenditure is the primary goal.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; font-size:1.2rem; font-weight:700; color:#f8fafc;'>📊 Scenario Comparison Table</h3>", unsafe_allow_html=True)
    
    # Format table for visual presentation
    display_df = comp_df.copy()
    display_df["Principal (₹)"] = display_df["Principal (₹)"].apply(format_inr)
    display_df["Monthly EMI (₹)"] = display_df["Monthly EMI (₹)"].apply(format_inr)
    display_df["Total Interest (₹)"] = display_df["Total Interest (₹)"].apply(format_inr)
    display_df["Total Repayment (₹)"] = display_df["Total Repayment (₹)"].apply(format_inr)
    display_df["Fees & Charges (₹)"] = display_df["Fees & Charges (₹)"].apply(format_inr)
    display_df["Total Cost with Fees (₹)"] = display_df["Total Cost with Fees (₹)"].apply(format_inr)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Comparison Bar Chart
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        fig_bar_emi = px.bar(
            comp_df,
            x="Scenario",
            y="Monthly EMI (₹)",
            color="Scenario",
            title="<b>Monthly EMI Outflow by Scenario (₹)</b>",
            text_auto=".2s",
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399"]
        )
        fig_bar_emi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_bar_emi, use_container_width=True)
        
    with c_chart2:
        fig_bar_interest = px.bar(
            comp_df,
            x="Scenario",
            y="Total Interest (₹)",
            color="Scenario",
            title="<b>Total Lifetime Interest Burden by Scenario (₹)</b>",
            text_auto=".2s",
            template="plotly_dark",
            color_discrete_sequence=["#f43f5e", "#fb923c", "#facc15"]
        )
        fig_bar_interest.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_bar_interest, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------
# PAGE 5: REPAYMENT SCHEDULE
# ------------------------------------------------------------------------------------------
elif selected_page == "Repayment Schedule":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span class="badge-pill badge-blue">Amortization Table</span>
            <span style="color: #64748b; font-size: 0.82rem;">Month-by-Month Principal & Interest Depletion</span>
        </div>
        <h2 style="color: #ffffff; font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">Monthly Amortization Repayment Schedule</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">Track outstanding balance depletion, principal vs interest distribution, and export the schedule as CSV.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        sched_p = st.number_input("Principal (₹ INR)", min_value=10000.0, max_value=100000000.0, value=500000.0, step=25000.0)
    with f2:
        sched_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.25)
    with f3:
        sched_months = st.number_input("Tenure (Months)", min_value=1, max_value=360, value=60, step=12)
        
    sched_df = generate_amortization_schedule(sched_p, sched_rate, sched_months)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Line chart showing balance decreasing over time
    if not sched_df.empty:
        fig_amort = go.Figure()
        fig_amort.add_trace(go.Scatter(
            x=sched_df["Instalment_No"],
            y=sched_df["Closing_Balance"],
            mode="lines",
            name="Outstanding Balance",
            line=dict(color="#38bdf8", width=3)
        ))
        fig_amort.add_trace(go.Scatter(
            x=sched_df["Instalment_No"],
            y=sched_df["Principal_Component"],
            mode="lines",
            name="Principal Component",
            line=dict(color="#10b981", width=2, dash="dot")
        ))
        fig_amort.add_trace(go.Scatter(
            x=sched_df["Instalment_No"],
            y=sched_df["Interest_Component"],
            mode="lines",
            name="Interest Component",
            line=dict(color="#f43f5e", width=2, dash="dot")
        ))
        fig_amort.update_layout(
            title="<b>Loan Depletion Curve & Monthly Component Trends</b>",
            xaxis_title="Instalment Month",
            yaxis_title="Amount (₹ INR)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark",
            height=320,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_amort, use_container_width=True)
        
        # Format table for display
        st.markdown("##### Detailed Monthly Ledger")
        display_sched = sched_df.copy()
        for col in ["Opening_Balance", "Monthly_EMI", "Interest_Component", "Principal_Component", "Closing_Balance"]:
            display_sched[col] = display_sched[col].apply(lambda v: f"₹{v:,.2f}")
            
        st.dataframe(display_sched, use_container_width=True, hide_index=True)
        
        # Download Schedule as CSV
        csv_sched = sched_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Amortization Schedule (CSV)",
            data=csv_sched,
            file_name="amortization_schedule.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------
# PAGE 6: LOAN AFFORDABILITY ANALYSIS
# ------------------------------------------------------------------------------------------
elif selected_page == "Loan Affordability Analysis":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span class="badge-pill badge-blue">Cash-Flow Underwriting</span>
            <span style="color: #64748b; font-size: 0.82rem;">Prudential Debt-To-Income (DTI) Bandwidth Check</span>
        </div>
        <h2 style="color: #ffffff; font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">Loan Affordability & Debt Capacity Analysis</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">Assess how new loan obligations impact disposable monthly income and debt servicing comfort.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_aff_in, col_aff_out = st.columns([1, 1.2])
    
    with col_aff_in:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-size:1.2rem; font-weight:700; color:#f8fafc;'>💼 Monthly Cash Flows</h3>", unsafe_allow_html=True)
        
        aff_income = st.number_input(
            "Monthly Gross Income (₹ INR)",
            min_value=0.0,
            max_value=2000000.0,
            value=85000.0,
            step=5000.0,
            help="Total monthly gross salary or verified business receipts."
        )
        st.caption(f"Formatted: **{format_inr(aff_income)} / month**")
        
        aff_existing_debt = st.number_input(
            "Existing Monthly Debt Obligations (₹ INR)",
            min_value=0.0,
            max_value=1000000.0,
            value=12000.0,
            step=1000.0,
            help="Current running EMIs (e.g. car loan, credit card minimums, personal loan)."
        )
        st.caption(f"Formatted: **{format_inr(aff_existing_debt)} / month**")
        
        aff_proposed_emi = st.number_input(
            "Proposed Loan EMI (₹ INR)",
            min_value=0.0,
            max_value=1000000.0,
            value=18500.0,
            step=1000.0,
            help="Estimated monthly EMI of the requested facility."
        )
        st.caption(f"Formatted: **{format_inr(aff_proposed_emi)} / month**")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_aff_out:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-size:1.2rem; font-weight:700; color:#f8fafc;'>🛡️ Affordability Diagnostics</h3>", unsafe_allow_html=True)
        
        aff_res = check_loan_affordability(aff_income, aff_existing_debt, aff_proposed_emi)
        
        # Color styling based on status
        color_hex = "#10b981" if aff_res["status_color"] == "green" else (
            "#f59e0b" if aff_res["status_color"] == "amber" else "#ef4444"
        )
        
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border-left: 4px solid {color_hex}; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
            <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Affordability Status</div>
            <div style="color: {color_hex}; font-size: 1.6rem; font-weight: 800; margin: 4px 0;">{aff_res['affordability_status']}</div>
            <div style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.5;">{aff_res['guidance']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Total Monthly Debt", format_inr(aff_res["total_monthly_debt"]))
        with k2:
            st.metric("Disposable Income Left", format_inr(aff_res["remaining_income"]))
        with k3:
            st.metric("DTI Percentage", f"{aff_res['dti_pct']:.1f}%")
            
        # Gauge Meter for DTI
        fig_dti_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aff_res["dti_pct"],
            title={'text': "Debt-to-Income (DTI) Ratio (%)", 'font': {'size': 13, 'color': "#94a3b8"}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#cbd5e1"},
                'bar': {'color': color_hex},
                'steps': [
                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [35, 50], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "#f43f5e", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_dti_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=180, margin=dict(t=25, b=10, l=20, r=20))
        st.plotly_chart(fig_dti_gauge, use_container_width=True)
        
        st.caption(f"ℹ️ *{aff_res['assumptions_note']}*")
        st.markdown('</div>', unsafe_allow_html=True)
