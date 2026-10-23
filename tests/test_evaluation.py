"""
Automated Unit Tests for Evaluation & ML Visualization Helpers.
Verifies model metrics loading, confusion matrix calculation, feature importances, and ROC data.
"""

import pytest
import pandas as pd
from src.evaluation import (
    load_model_metrics,
    load_classification_reports,
    get_feature_importances,
    get_roc_data,
    get_confusion_matrix_details,
    get_cv_scores_df
)


def test_load_model_metrics():
    df = load_model_metrics()
    assert not df.empty, "Model metrics dataframe should not be empty"
    assert "Model" in df.columns
    assert "Accuracy" in df.columns
    assert "F1_Score" in df.columns
    assert len(df) >= 9, "Should contain at least 9 baseline models + ensemble"


def test_load_classification_reports():
    reports = load_classification_reports()
    assert isinstance(reports, dict)
    assert "Voting Ensemble (RF+ET+KNN)" in reports
    ens = reports["Voting Ensemble (RF+ET+KNN)"]
    assert "confusion_matrix" in ens
    assert "report" in ens


def test_confusion_matrix_details_calculation():
    details = get_confusion_matrix_details("Voting Ensemble (RF+ET+KNN)")
    assert details, "Details dict should not be empty"
    assert details["tn"] > 0
    assert details["tp"] > 0
    assert 0.0 <= details["accuracy"] <= 1.0
    assert 0.0 <= details["sensitivity"] <= 1.0
    assert 0.0 <= details["specificity"] <= 1.0
    assert 0.0 <= details["precision"] <= 1.0
    assert details["total"] == (details["tn"] + details["fp"] + details["fn"] + details["tp"])


def test_get_feature_importances():
    feat_df = get_feature_importances()
    assert not feat_df.empty, "Feature importances DataFrame should not be empty"
    assert "Feature" in feat_df.columns
    assert "Importance" in feat_df.columns
    assert feat_df["Importance"].sum() == pytest.approx(1.0, abs=1e-3)
    # Credit History should be the top predictive feature
    top_feature = feat_df.iloc[0]["Feature"]
    assert "Credit_History" in top_feature


def test_get_roc_data():
    roc_data = get_roc_data()
    assert "Voting Ensemble" in roc_data or "Random Forest" in roc_data
    if "Voting Ensemble" in roc_data:
        ens_roc = roc_data["Voting Ensemble"]
        assert "fpr" in ens_roc
        assert "tpr" in ens_roc
        assert 0.85 <= ens_roc["auc"] <= 1.0


def test_get_cv_scores_df():
    cv_df = get_cv_scores_df()
    assert not cv_df.empty
    assert "Model" in cv_df.columns
    assert "CV_Accuracy" in cv_df.columns
    assert "CV_F1" in cv_df.columns
