"""
Evaluation Metrics and Visualizations Helper Module.
Generates evaluation figures, ROC curves, confusion matrices, and feature importances.
Provides helper functions for Streamlit interactive visualizations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from src.preprocessing import get_feature_names


def get_project_root() -> str:
    """Returns absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


def load_model_metrics(project_root: str = None) -> pd.DataFrame:
    """Loads model performance metrics DataFrame from reports/model_metrics.csv."""
    root = project_root or get_project_root()
    metrics_file = os.path.join(root, "reports", "model_metrics.csv")
    if os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    return pd.DataFrame()


def load_classification_reports(project_root: str = None) -> dict:
    """Loads full classification report dictionary including confusion matrices and CV scores."""
    root = project_root or get_project_root()
    reports_file = os.path.join(root, "reports", "classification_reports.json")
    if os.path.exists(reports_file):
        with open(reports_file, "r") as f:
            return json.load(f)
    return {}


def get_feature_importances(project_root: str = None) -> pd.DataFrame:
    """Extracts feature importances from the trained Random Forest model pipeline."""
    root = project_root or get_project_root()
    rf_file = os.path.join(root, "models", "random_forest_pipeline.joblib")
    if not os.path.exists(rf_file):
        return pd.DataFrame(columns=["Feature", "Importance"])
    
    rf_pipeline = joblib.load(rf_file)
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)
    rf_clf = rf_pipeline.named_steps["classifier"]
    importances = rf_clf.feature_importances_
    
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return feat_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)


def get_roc_data(project_root: str = None) -> dict:
    """
    Computes ROC curves and AUC scores on the held-out test split
    for both Voting Ensemble and Random Forest.
    """
    root = project_root or get_project_root()
    splits_file = os.path.join(root, "models", "test_splits.joblib")
    ens_file = os.path.join(root, "models", "voting_ensemble_pipeline.joblib")
    rf_file = os.path.join(root, "models", "random_forest_pipeline.joblib")
    
    results = {}
    if not os.path.exists(splits_file):
        return results
        
    splits = joblib.load(splits_file)
    X_test, y_test = splits["X_test"], splits["y_test"]
    
    if os.path.exists(ens_file):
        ens = joblib.load(ens_file)
        y_proba_ens = ens.predict_proba(X_test)[:, 1]
        fpr_ens, tpr_ens, _ = roc_curve(y_test, y_proba_ens)
        results["Voting Ensemble"] = {
            "fpr": fpr_ens.tolist(),
            "tpr": tpr_ens.tolist(),
            "auc": float(auc(fpr_ens, tpr_ens))
        }
        
    if os.path.exists(rf_file):
        rf = joblib.load(rf_file)
        y_proba_rf = rf.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
        results["Random Forest"] = {
            "fpr": fpr_rf.tolist(),
            "tpr": tpr_rf.tolist(),
            "auc": float(auc(fpr_rf, tpr_rf))
        }
        
    return results


def get_confusion_matrix_details(model_name: str = "Voting Ensemble (RF+ET+KNN)", project_root: str = None) -> dict:
    """
    Extracts and calculates TN, FP, FN, TP, Specificity, Sensitivity, and Precision
    for any specified model from the classification report.
    """
    reports = load_classification_reports(project_root)
    if model_name not in reports:
        return {}
        
    data = reports[model_name]
    cm = data.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    false_sanction_rate = fp / (tp + fp) if (tp + fp) > 0 else 0  # Bad loans approved
    
    return {
        "model_name": model_name,
        "matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "false_sanction_rate": false_sanction_rate,
        "report": data.get("report", {})
    }


def get_cv_scores_df(project_root: str = None) -> pd.DataFrame:
    """Extracts 5-fold cross-validation scores for all models into a tidy DataFrame."""
    reports = load_classification_reports(project_root)
    records = []
    for model_name, data in reports.items():
        if "cv_scores" in data and "test_accuracy" in data["cv_scores"]:
            for fold_idx, acc in enumerate(data["cv_scores"]["test_accuracy"]):
                f1 = data["cv_scores"]["test_f1"][fold_idx]
                records.append({
                    "Model": model_name,
                    "Fold": f1_fold if (f1_fold := f"Fold {fold_idx + 1}") else fold_idx + 1,
                    "CV_Accuracy": acc,
                    "CV_F1": f1
                })
    return pd.DataFrame(records)


def generate_evaluation_artifacts(output_dir: str):
    """
    Reads saved models and test splits to generate static charts in visualizations/
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    reports_file = os.path.join(output_dir, "reports", "classification_reports.json")
    metrics_file = os.path.join(output_dir, "reports", "model_metrics.csv")
    splits_file = os.path.join(output_dir, "models", "test_splits.joblib")
    rf_file = os.path.join(output_dir, "models", "random_forest_pipeline.joblib")
    
    if not (os.path.exists(reports_file) and os.path.exists(splits_file)):
        print("Required artifacts not found for offline chart generation.")
        return
        
    metrics_df = pd.read_csv(metrics_file)
    splits = joblib.load(splits_file)
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    
    # 1. Model Accuracy & F1 Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    bar_df = metrics_df.sort_values(by="Accuracy", ascending=True)
    y_pos = np.arange(len(bar_df))
    plt.barh(y_pos - 0.2, bar_df["Accuracy"], height=0.4, label="Accuracy", color="#1E88E5")
    plt.barh(y_pos + 0.2, bar_df["F1_Score"], height=0.4, label="F1 Score", color="#004D40")
    plt.yticks(y_pos, bar_df["Model"])
    plt.xlabel("Score")
    plt.title("Model Performance Benchmark: Accuracy vs F1-Score")
    plt.legend(loc="lower right")
    plt.xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "model_comparison_bar.png"), dpi=200)
    plt.close()
    
    # 2. Confusion Matrix for Ensemble
    with open(reports_file, "r") as f:
        rep_data = json.load(f)
        
    cm = np.array(rep_data["Voting Ensemble (RF+ET+KNN)"]["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Rejected (0)", "Approved (1)"],
                yticklabels=["Rejected (0)", "Approved (1)"])
    plt.title("Voting Ensemble Confusion Matrix (Held-out Test)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "ensemble_confusion_matrix.png"), dpi=200)
    plt.close()
    
    # 3. Feature Importance (Random Forest from pipeline)
    if os.path.exists(rf_file):
        rf_pipeline = joblib.load(rf_file)
        preprocessor = rf_pipeline.named_steps["preprocessor"]
        feature_names = get_feature_names(preprocessor)
        rf_clf = rf_pipeline.named_steps["classifier"]
        importances = rf_clf.feature_importances_
        
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=False).head(12)
        
        plt.figure(figsize=(9, 5))
        sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
        plt.title("Top Feature Importances (Random Forest Classifier)")
        plt.xlabel("Gini Importance")
        plt.ylabel("Engineered Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=200)
        plt.close()
        
    print(f"Generated evaluation visual artifacts in: {vis_dir}")


if __name__ == "__main__":
    project_root = get_project_root()
    generate_evaluation_artifacts(project_root)
