"""
Evaluation Metrics and Visualizations Helper Module.
Generates evaluation figures, ROC curves, confusion matrices, and feature importances.
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


def generate_evaluation_artifacts(output_dir: str):
    """
    Reads saved models and test splits to generate charts in visualizations/
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    generate_evaluation_artifacts(project_root)
