from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import CATEGORICAL_COLUMNS, TARGET_COLUMN


MODELING_NUMERIC_FEATURES = [
    "age",
    "semester",
    "study_hours",
    "attendance",
    "assignment_score",
    "internal_marks",
    "previous_percentage",
    "practical_score",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = [col for col in MODELING_NUMERIC_FEATURES if col in df.columns]
    categorical = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    return numeric + categorical


def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = result.drop_duplicates()
    for column in MODELING_NUMERIC_FEATURES + [TARGET_COLUMN]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("string").str.strip().fillna("Unknown")
            result.loc[result[column].eq(""), column] = "Unknown"

    if TARGET_COLUMN in result.columns:
        result = result[result[TARGET_COLUMN].notna()]
        result[TARGET_COLUMN] = result[TARGET_COLUMN].clip(0, 100)
    return result


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, MODELING_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def split_features(
    df: pd.DataFrame, target: str = TARGET_COLUMN, test_size: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    clean_df = clean_for_modeling(df)
    features = get_feature_columns(clean_df)
    X = clean_df[features]
    y = clean_df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def build_risk_target(final_percentage: pd.Series, threshold: float = 50) -> pd.Series:
    return np.where(final_percentage < threshold, 1, 0)
