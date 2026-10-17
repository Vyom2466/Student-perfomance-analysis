from __future__ import annotations

import pandas as pd

from .data_loader import NUMERIC_COLUMNS, TARGET_COLUMN


def dataset_overview(df: pd.DataFrame) -> dict[str, object]:
    return {
        "records": len(df),
        "features": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": df.dtypes.astype(str).to_dict(),
    }


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [col for col in NUMERIC_COLUMNS if col in df.columns]
    return df[numeric].describe().T.round(2)


def group_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups: dict[str, pd.DataFrame] = {}
    for column in ["gender", "semester", "internet_access", "extracurricular"]:
        if column in df.columns:
            groups[column] = (
                df.groupby(column, dropna=False)[TARGET_COLUMN]
                .agg(["count", "mean", "median", "std"])
                .round(2)
                .reset_index()
            )
    return groups


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [col for col in NUMERIC_COLUMNS if col in df.columns]
    corr = df[numeric].corr(numeric_only=True)[[TARGET_COLUMN]].drop(TARGET_COLUMN)
    return corr.rename(columns={TARGET_COLUMN: "Correlation with Final %"}).sort_values(
        "Correlation with Final %", ascending=False
    ).round(3)


def dashboard_kpis(df: pd.DataFrame, prediction_frame: pd.DataFrame | None = None) -> dict[str, object]:
    total = len(df)
    at_risk = int((df[TARGET_COLUMN] < 50).sum())
    high = int((df[TARGET_COLUMN] >= 75).sum())
    if prediction_frame is not None and "predicted_risk_level" in prediction_frame:
        at_risk = int(
            prediction_frame["predicted_risk_level"].isin(["High Risk", "Moderate Risk"]).sum()
        )
        high = int((prediction_frame["predicted_category"] == "High").sum())
    return {
        "Total Students": total,
        "Average Performance": f"{df[TARGET_COLUMN].mean():.1f}%",
        "Average Attendance": f"{df['attendance'].mean():.1f}%",
        "Pass Rate": f"{(df[TARGET_COLUMN].ge(50).mean() * 100):.1f}%",
        "At-Risk Students": at_risk,
        "High-Performing Students": high,
    }


def at_risk_students(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "student_id",
        "attendance",
        "study_hours",
        "assignment_score",
        "internal_marks",
        "previous_percentage",
        "predicted_percentage",
        "predicted_category",
        "predicted_risk_level",
        "risk_probability",
    ]
    available = [col for col in columns if col in prediction_frame.columns]
    risk_df = prediction_frame[
        prediction_frame["predicted_risk_level"].isin(["High Risk", "Moderate Risk"])
    ][available]
    sort_column = "risk_probability" if "risk_probability" in risk_df else "predicted_percentage"
    return risk_df.sort_values(sort_column, ascending=False if sort_column == "risk_probability" else True)

