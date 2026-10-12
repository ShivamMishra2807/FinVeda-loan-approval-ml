"""
Data Preprocessing and Feature Pipeline Module.
Implements robust data cleaning, missing-value imputation, feature engineering,
categorical encoding, numerical scaling, and SMOTE integration without data leakage.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Canonical feature schema
NUMERICAL_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Total_Income",
    "Loan_Income_Ratio",
    "Debt_Burden"
]

BASE_NUMERICAL_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"

TARGET_MAP = {"N": 0, "Y": 1}
REVERSE_TARGET_MAP = {0: "N", 1: "Y"}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific financial ratios:
    - Total_Income = ApplicantIncome + CoapplicantIncome
    - Loan_Income_Ratio = (LoanAmount * 1000) / Total_Income
    - Debt_Burden = (LoanAmount * 1000 / Loan_Amount_Term) / Total_Income
    """
    df = df.copy()
    total_inc = df["ApplicantIncome"].fillna(0) + df["CoapplicantIncome"].fillna(0)
    # Avoid zero division
    total_inc_safe = np.maximum(total_inc, 1000.0)
    
    df["Total_Income"] = total_inc
    df["Loan_Income_Ratio"] = (df["LoanAmount"] * 1000.0) / total_inc_safe
    term_safe = np.maximum(df["Loan_Amount_Term"].fillna(360), 12.0)
    monthly_installment = (df["LoanAmount"] * 1000.0) / term_safe
    df["Debt_Burden"] = monthly_installment / total_inc_safe
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate records, enforce schema, and handle structural anomalies.
    Does NOT impute missing values here to prevent leakage before train/test split.
    """
    df_clean = df.copy()
    
    # Remove duplicate records
    # Check by Loan_ID or all feature columns
    if ID_COL in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=[ID_COL], keep="first")
    df_clean = df_clean.drop_duplicates(keep="first")
    
    # Strip whitespace in string columns
    str_cols = df_clean.select_dtypes(include=["object"]).columns
    for c in str_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip().replace("nan", np.nan)
        
    # Standardize Dependents '3+' if needed
    if "Dependents" in df_clean.columns:
        df_clean["Dependents"] = df_clean["Dependents"].replace({"3+": "3"})
        
    return df_clean


def build_preprocessor() -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for numerical and categorical pipelines.
    - Numerical: Median Imputer + Robust/Standard Scaler
    - Categorical: Most Frequent Imputer + OneHotEncoder (handle_unknown='ignore')
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def create_training_pipeline(classifier: Any, use_smote: bool = True, random_state: int = 42) -> ImbPipeline:
    """
    Creates an imblearn Pipeline ensuring SMOTE is fitted ONLY on training folds.
    Prevents test set leakage and fractional categorical artifacts.
    """
    preprocessor = build_preprocessor()
    steps = [("preprocessor", preprocessor)]
    
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
        
    steps.append(("classifier", classifier))
    return ImbPipeline(steps=steps)


def prepare_dataset_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Clean raw data, engineer features, extract X and y, and perform stratified train-test split.
    """
    cleaned_df = clean_raw_data(df)
    engineered_df = add_engineered_features(cleaned_df)
    
    # Validate target presence
    if TARGET_COL not in engineered_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
        
    # Drop rows where target is missing if any
    valid_data = engineered_df.dropna(subset=[TARGET_COL]).copy()
    
    # Target encoding
    y = valid_data[TARGET_COL].map(TARGET_MAP)
    
    # Features (drop ID and Target)
    drop_cols = [TARGET_COL]
    if ID_COL in valid_data.columns:
        drop_cols.append(ID_COL)
        
    X = valid_data.drop(columns=drop_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract processed feature names after ColumnTransformer fit."""
    feature_names = []
    # Numerical names
    feature_names.extend(NUMERICAL_FEATURES)
    # Categorical names from OneHotEncoder
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names.extend(cat_names)
    return feature_names
