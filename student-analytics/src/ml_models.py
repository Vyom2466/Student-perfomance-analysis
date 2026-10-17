from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import os
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_loader import CATEGORICAL_COLUMNS, TARGET_COLUMN, add_performance_columns
from .preprocessing import (
    MODELING_NUMERIC_FEATURES,
    build_preprocessor,
    build_risk_target,
    clean_for_modeling,
    get_feature_columns,
)


@dataclass
class TrainingResult:
    regression_results: pd.DataFrame
    classification_results: pd.DataFrame
    best_regression_name: str
    best_classification_name: str
    regression_model: Pipeline
    classification_model: Pipeline
    feature_importance: pd.DataFrame
    prediction_frame: pd.DataFrame
    clustering: dict[str, Any]
    explanation: str


def train_all_models(df: pd.DataFrame) -> TrainingResult:
    clean_df = clean_for_modeling(df)
    if len(clean_df) < 30:
        raise ValueError("At least 30 valid records are required for model training.")

    features = get_feature_columns(clean_df)
    X = clean_df[features]
    y_reg = clean_df[TARGET_COLUMN]
    y_cls = build_risk_target(y_reg)

    stratify = y_cls if pd.Series(y_cls).nunique() > 1 else None
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X,
        y_reg,
        y_cls,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    reg_results, reg_models = _train_regression_candidates(
        X_train, X_test, y_train_reg, y_test_reg
    )
    cls_results, cls_models = _train_classification_candidates(
        X_train, X_test, y_train_cls, y_test_cls
    )

    best_reg_name = reg_results.sort_values(["MAE", "RMSE"], ascending=True).iloc[0][
        "Model"
    ]
    best_cls_name = cls_results.sort_values(
        ["F1 Score", "Recall", "Precision"], ascending=[False, False, False]
    ).iloc[0]["Model"]

    best_reg_model = reg_models[best_reg_name]
    best_cls_model = cls_models[best_cls_name]
    importance = get_feature_importance(best_reg_model, best_reg_name)
    predictions = build_prediction_frame(clean_df, best_reg_model, best_cls_model)
    clustering = segment_students(clean_df)

    explanation = (
        f"{best_reg_name} is selected for numerical score prediction because it has "
        f"the lowest MAE on the test set. {best_cls_name} is selected for at-risk "
        "prediction using F1 score first, with recall considered because missing an "
        "at-risk student is more serious than a false alert."
    )

    return TrainingResult(
        regression_results=reg_results,
        classification_results=cls_results,
        best_regression_name=str(best_reg_name),
        best_classification_name=str(best_cls_name),
        regression_model=best_reg_model,
        classification_model=best_cls_model,
        feature_importance=importance,
        prediction_frame=predictions,
        clustering=clustering,
        explanation=explanation,
    )


def predict_single_student(
    model_result: TrainingResult, student_record: dict[str, Any]
) -> dict[str, Any]:
    X = pd.DataFrame([student_record])
    predicted_score = float(model_result.regression_model.predict(X)[0])
    predicted_score = float(np.clip(predicted_score, 0, 100))
    predicted_risk = int(model_result.classification_model.predict(X)[0])
    confidence = None
    if hasattr(model_result.classification_model, "predict_proba"):
        probabilities = model_result.classification_model.predict_proba(X)[0]
        confidence = float(max(probabilities))

    category = _category_from_score(predicted_score)
    risk_level = _risk_level(predicted_score, predicted_risk)
    return {
        "predicted_percentage": round(predicted_score, 2),
        "performance_category": category,
        "risk_level": risk_level,
        "risk_flag": predicted_risk,
        "confidence": confidence,
    }


def _train_regression_candidates(X_train, X_test, y_train, y_test):
    candidates = {
        "Linear Regression": Pipeline(
            [("preprocessor", build_preprocessor(scale_numeric=True)), ("model", LinearRegression())]
        ),
        "Random Forest Regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=3)),
            ]
        ),
        "Gradient Boosting Regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
    }
    rows = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": sqrt(mean_squared_error(y_test, predictions)),
                "R2": r2_score(y_test, predictions),
            }
        )
        fitted[name] = model
    return pd.DataFrame(rows).round(4), fitted


def _train_classification_candidates(X_train, X_test, y_train, y_test):
    if len(np.unique(y_train)) < 2:
        raise ValueError("At-risk model needs at least two risk classes in the dataset.")

    candidates = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest Classifier": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced", min_samples_leaf=3)),
            ]
        ),
        "Gradient Boosting Classifier": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }
    rows = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1 Score": f1_score(y_test, predictions, zero_division=0),
                "Confusion Matrix": confusion_matrix(y_test, predictions).tolist(),
            }
        )
        fitted[name] = model
    return pd.DataFrame(rows).round(4), fitted


def build_prediction_frame(
    df: pd.DataFrame, regression_model: Pipeline, classification_model: Pipeline
) -> pd.DataFrame:
    features = get_feature_columns(df)
    result = add_performance_columns(df).copy()
    result["predicted_percentage"] = np.clip(regression_model.predict(result[features]), 0, 100)
    result["predicted_risk_flag"] = classification_model.predict(result[features])
    result["predicted_risk_level"] = result.apply(
        lambda row: _risk_level(row["predicted_percentage"], row["predicted_risk_flag"]),
        axis=1,
    )
    result["predicted_category"] = result["predicted_percentage"].apply(_category_from_score)
    if hasattr(classification_model, "predict_proba"):
        result["risk_probability"] = classification_model.predict_proba(result[features])[:, 1]
    else:
        result["risk_probability"] = np.nan
    result["predicted_percentage"] = result["predicted_percentage"].round(2)
    return result


def get_feature_importance(model: Pipeline, model_name: str) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    clean_names = [name.replace("num__", "").replace("cat__", "") for name in names]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    return (
        pd.DataFrame({"Feature": clean_names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )


def segment_students(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df[MODELING_NUMERIC_FEATURES].copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median())
    if len(numeric) < 40:
        return {"enabled": False, "reason": "Not enough records for stable segmentation."}

    preprocessor = build_preprocessor(scale_numeric=True)
    X_scaled = preprocessor.fit_transform(df[MODELING_NUMERIC_FEATURES + CATEGORICAL_COLUMNS])

    scores = []
    models = {}
    upper_k = min(5, len(df) - 1)
    for k in range(2, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append({"Clusters": k, "Silhouette Score": round(float(score), 4)})
        models[k] = (model, labels)

    best_k = max(scores, key=lambda item: item["Silhouette Score"])["Clusters"]
    best_model, labels = models[best_k]
    segmented = df.copy()
    segmented["segment"] = labels + 1
    profile = (
        segmented.groupby("segment")[
            ["final_percentage", "attendance", "study_hours", "assignment_score", "internal_marks"]
        ]
        .mean()
        .round(2)
        .reset_index()
    )
    return {
        "enabled": True,
        "best_k": best_k,
        "scores": pd.DataFrame(scores),
        "profile": profile,
        "labels": labels,
        "model": best_model,
    }


def _category_from_score(score: float) -> str:
    if score < 50:
        return "At Risk"
    if score < 75:
        return "Average"
    return "High"


def _risk_level(score: float, risk_flag: int) -> str:
    if risk_flag == 1 or score < 50:
        return "High Risk"
    if score < 60:
        return "Moderate Risk"
    return "Low Risk"
