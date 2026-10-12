"""
Synthetic Academic Loan Approval Dataset Generator
Generates a realistic, diverse dataset for bank loan approval prediction.
Complies with PRD specifications, academic reproducibility, and data-integrity guidelines.
"""

import os
import numpy as np
import pandas as pd

def generate_loan_dataset(
    n_samples: int = 5200,
    random_state: int = 42,
    include_duplicates: int = 50,
    missing_rate: float = 0.02
) -> pd.DataFrame:
    """
    Generate synthetic academic loan application dataset.
    
    Attributes:
    - Loan_ID: Unique application identifier (LP000001 ...)
    - Gender: Male / Female
    - Married: Yes / No
    - Dependents: 0, 1, 2, 3+
    - Education: Graduate / Not Graduate
    - Self_Employed: Yes / No
    - ApplicantIncome: Applicant's monthly income
    - CoapplicantIncome: Co-applicant's monthly income
    - LoanAmount: Requested loan amount in thousands
    - Loan_Amount_Term: Loan repayment period in months
    - Credit_History: Simulated credit history indicator (1.0 or 0.0)
    - Property_Area: Urban / Semiurban / Rural
    - Loan_Status: Yes / No target variable
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Categorical Demographics
    genders = rng.choice(["Male", "Female"], size=n_samples, p=[0.78, 0.22])
    married = rng.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    
    # Dependents: correlated slightly with marital status
    dependents = []
    for m in married:
        if m == "Yes":
            dep = rng.choice(["0", "1", "2", "3+"], p=[0.45, 0.25, 0.20, 0.10])
        else:
            dep = rng.choice(["0", "1", "2", "3+"], p=[0.82, 0.10, 0.06, 0.02])
        dependents.append(dep)
        
    education = rng.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.78, 0.22])
    self_employed = rng.choice(["No", "Yes"], size=n_samples, p=[0.86, 0.14])
    property_area = rng.choice(["Semiurban", "Urban", "Rural"], size=n_samples, p=[0.38, 0.33, 0.29])
    
    # 2. Numerical Features
    # Applicant Income: log-normal distribution with educational and employment variation
    base_income_mean = np.where(education == "Graduate", 8.4, 8.0)
    base_income_mean += np.where(self_employed == "Yes", 0.2, 0.0)
    applicant_income = np.exp(rng.normal(loc=base_income_mean, scale=0.6, size=n_samples)).astype(int)
    # Clip to realistic range: $1,200 to $85,000
    applicant_income = np.clip(applicant_income, 1200, 85000)
    
    # Coapplicant Income: ~45% have 0 co-applicant income, others follow log-normal
    has_coapplicant = rng.choice([0, 1], size=n_samples, p=[0.42, 0.58])
    coapplicant_income_raw = np.exp(rng.normal(loc=7.6, scale=0.7, size=n_samples)).astype(int)
    coapplicant_income = np.where(has_coapplicant == 1, np.clip(coapplicant_income_raw, 800, 40000), 0)
    
    total_income = applicant_income + coapplicant_income
    
    # Loan Term: common mortgage/loan durations (360 is typical 30-year, 180 is 15-year, etc.)
    term_choices = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
    term_probs = [0.005, 0.01, 0.02, 0.015, 0.02, 0.08, 0.04, 0.02, 0.77, 0.02]
    loan_term = rng.choice(term_choices, size=n_samples, p=term_probs)
    
    # Requested Loan Amount (in thousands) correlated with Total Income and Term
    # E.g., multiplier ~ 0.02 to 0.04 of annual income (approx 0.25 to 0.45 of monthly income * 10)
    loan_amount_mean = (total_income * 0.024) + rng.normal(30, 25, size=n_samples)
    loan_amount = np.clip(loan_amount_mean, 15, 750).astype(int)
    
    # Credit History: 1.0 (meets guidelines) or 0.0 (does not meet)
    # Educated, higher income, married have marginally higher credit scores
    credit_prob = 0.75 + (education == "Graduate") * 0.05 + (total_income > 6000) * 0.06 - (loan_amount > 350) * 0.05
    credit_prob = np.clip(credit_prob, 0.1, 0.95)
    credit_history = rng.binomial(1, credit_prob, size=n_samples).astype(float)
    
    # 3. Target Simulation (Loan_Status: 'Y' / 'N')
    # Probabilistic latent score based on underwriting fundamentals:
    # - Strongest predictor: Credit History (underwriting primary barrier)
    # - Debt-to-income ratio (LoanAmount / TotalIncome)
    # - Semiurban has historically higher approval odds in benchmark dataset
    # - Property Area, Education, Income stability
    
    # Monthly payment proxy (rough linear amortization)
    monthly_installment = (loan_amount * 1000) / np.maximum(loan_term, 12)
    debt_burden = monthly_installment / np.maximum(total_income, 1000)
    
    latent_score = (
        -1.5
        + 3.2 * credit_history
        - 2.8 * np.clip(debt_burden - 0.28, 0, 1.0)
        + 0.5 * (property_area == "Semiurban")
        + 0.1 * (property_area == "Urban")
        + 0.3 * (education == "Graduate")
        + 0.25 * (married == "Yes")
        + 0.35 * (np.log(total_income) - 8.5)
        - 0.2 * (dependents == "3+")
        + rng.logistic(loc=0, scale=0.65, size=n_samples) # Logistic noise for realistic overlap
    )
    
    prob_approval = 1.0 / (1.0 + np.exp(-latent_score))
    loan_status = np.where(prob_approval >= 0.50, "Y", "N")
    
    # Generate Loan IDs
    loan_ids = [f"LP{str(i+100001).zfill(6)}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        "Loan_ID": loan_ids,
        "Gender": genders,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status
    })
    
    # 4. Introduce missing values realistically across typical features
    cols_with_missing = ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    for col in cols_with_missing:
        mask = rng.random(n_samples) < missing_rate
        df.loc[mask, col] = np.nan
        
    # 5. Inject controlled duplicate rows (for testing data cleaning)
    if include_duplicates > 0:
        dup_indices = rng.choice(n_samples, size=include_duplicates, replace=False)
        dup_rows = df.iloc[dup_indices].copy()
        df = pd.concat([df, dup_rows], ignore_index=True)
        # Shuffle
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
    return df

def save_dataset(filepath: str, n_samples: int = 5200, random_state: int = 42):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = generate_loan_dataset(n_samples=n_samples, random_state=random_state)
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic dataset saved to: {filepath}")
    print(f"Total Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Class distribution:")
    print(df["Loan_Status"].value_counts(normalize=True))
    print("\nMissing values per column:")
    print(df.isnull().sum())
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    output_path = os.path.join(project_root, "data", "loan_dataset.csv")
    save_dataset(output_path)
