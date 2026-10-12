"""
Model Training and Evaluation Pipeline.
Implements the 9 classifiers, the 3-model Voting Ensemble (RF + ET + KNN),
cross-validation, metric tracking, and model serialization.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import (
    prepare_dataset_splits,
    create_training_pipeline,
    build_preprocessor,
    get_feature_names
)


def get_base_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Define the 9 candidate classifiers specified in the PRD."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            C=1.0
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=6,
            random_state=random_state,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=6,
            random_state=random_state,
            n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            C=1.0,
            random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            n_jobs=-1
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.8,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state
        )
    }


def build_voting_ensemble(random_state: int = 42, voting: str = "soft") -> VotingClassifier:
    """
    Build Voting Classifier from the 3 specified benchmark models:
    - Random Forest
    - Extra Trees
    - KNN
    Using soft voting for calibrated probability estimates.
    """
    estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=6,
            random_state=random_state,
            n_jobs=-1
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=6,
            random_state=random_state,
            n_jobs=-1
        )),
        ("knn", KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            n_jobs=-1
        ))
    ]
    return VotingClassifier(estimators=estimators, voting=voting)


def train_and_evaluate_all_models(
    data_path: str,
    output_dir: str,
    random_state: int = 42,
    cv_folds: int = 5
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """
    Complete model exploration, cross-validation, test-set evaluation, and persistence.
    """
    print(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = prepare_dataset_splits(
        raw_df, test_size=0.2, random_state=random_state
    )
    
    print(f"Training split: {X_train.shape[0]} samples, Test split: {X_test.shape[0]} samples")
    print(f"Training positive ratio: {y_train.mean():.3f}, Test positive ratio: {y_test.mean():.3f}")
    
    classifiers = get_base_classifiers(random_state=random_state)
    classifiers["Voting Ensemble (RF+ET+KNN)"] = build_voting_ensemble(random_state=random_state)
    
    cv_strat = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    results = []
    classification_reports_dict = {}
    fitted_pipelines = {}
    
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    
    for name, clf in classifiers.items():
        print(f"\n--- Training and Evaluating: {name} ---")
        pipeline = create_training_pipeline(clf, use_smote=True, random_state=random_state)
        
        # 1. Stratified Cross Validation on Training Data Only
        cv_scoring = ["accuracy", "f1", "roc_auc"]
        cv_res = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_strat,
            scoring=cv_scoring,
            n_jobs=1
        )
        cv_acc_mean = float(np.mean(cv_res["test_accuracy"]))
        cv_acc_std = float(np.std(cv_res["test_accuracy"]))
        cv_f1_mean = float(np.mean(cv_res["test_f1"]))
        cv_auc_mean = float(np.mean(cv_res["test_roc_auc"]))
        
        # 2. Fit on full training set and measure time
        t_start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t_start
        
        # 3. Test Evaluation
        t_infer_start = time.time()
        y_pred = pipeline.predict(X_test)
        infer_time = time.time() - t_infer_start
        
        # Probabilities
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc_val = float(roc_auc_score(y_test, y_proba))
        else:
            auc_val = 0.0
            y_proba = None
            
        test_acc = float(accuracy_score(y_test, y_pred))
        test_prec = float(precision_score(y_test, y_pred, zero_division=0))
        test_rec = float(recall_score(y_test, y_pred, zero_division=0))
        test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": round(test_acc, 4),
            "Precision": round(test_prec, 4),
            "Recall": round(test_rec, 4),
            "F1_Score": round(test_f1, 4),
            "ROC_AUC": round(auc_val, 4),
            "CV_Accuracy_Mean": round(cv_acc_mean, 4),
            "CV_Accuracy_Std": round(cv_acc_std, 4),
            "CV_F1_Mean": round(cv_f1_mean, 4),
            "Train_Time_Sec": round(train_time, 3),
            "Infer_Time_Sec": round(infer_time, 4)
        })
        
        classification_reports_dict[name] = {
            "report": cls_report,
            "confusion_matrix": cm,
            "cv_scores": {k: [round(x, 4) for x in v.tolist()] for k, v in cv_res.items()}
        }
        
        fitted_pipelines[name] = pipeline
        
    metrics_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    # Save metrics table & detailed reports
    metrics_csv_path = os.path.join(output_dir, "reports", "model_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved metrics summary to: {metrics_csv_path}")
    
    reports_json_path = os.path.join(output_dir, "reports", "classification_reports.json")
    with open(reports_json_path, "w") as f:
        json.dump(classification_reports_dict, f, indent=2)
        
    # Save the Voting Ensemble model as primary production artifact
    ensemble_path = os.path.join(output_dir, "models", "voting_ensemble_pipeline.joblib")
    joblib.dump(fitted_pipelines["Voting Ensemble (RF+ET+KNN)"], ensemble_path)
    print(f"Saved Voting Ensemble Pipeline to: {ensemble_path}")
    
    # Also save Random Forest as primary tree explainability artifact
    rf_path = os.path.join(output_dir, "models", "random_forest_pipeline.joblib")
    joblib.dump(fitted_pipelines["Random Forest"], rf_path)
    
    # Save test dataset splits for evaluation lab visualization in Streamlit
    eval_splits_path = os.path.join(output_dir, "models", "test_splits.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, eval_splits_path)
    
    # Save summary dataset statistics
    summary_data = {
        "total_records": int(len(raw_df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_counts": raw_df["Loan_Status"].value_counts().to_dict(),
        "best_model_by_accuracy": metrics_df.iloc[0]["Model"],
        "ensemble_accuracy": float(metrics_df[metrics_df["Model"].str.contains("Voting")]["Accuracy"].values[0]),
        "ensemble_f1": float(metrics_df[metrics_df["Model"].str.contains("Voting")]["F1_Score"].values[0]),
        "benchmark_paper_accuracy": 0.8726
    }
    with open(os.path.join(output_dir, "reports", "dataset_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
        
    return metrics_df, classification_reports_dict, fitted_pipelines


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_file = os.path.join(project_root, "data", "loan_dataset.csv")
    train_and_evaluate_all_models(data_file, project_root)
