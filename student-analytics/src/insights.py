from __future__ import annotations

import pandas as pd

from .analysis import correlation_table
from .data_loader import TARGET_COLUMN


def generate_key_insights(
    df: pd.DataFrame,
    prediction_frame: pd.DataFrame | None = None,
    feature_importance: pd.DataFrame | None = None,
    clustering: dict | None = None,
) -> list[str]:
    insights: list[str] = []

    avg_performance = df[TARGET_COLUMN].mean()
    pass_rate = df[TARGET_COLUMN].ge(50).mean() * 100
    high_share = df[TARGET_COLUMN].ge(75).mean() * 100
    insights.append(
        f"Average final performance is {avg_performance:.1f}% with a pass rate of {pass_rate:.1f}%."
    )
    insights.append(f"{high_share:.1f}% of students are currently in the high-performing range.")

    low_attendance_share = df["attendance"].lt(75).mean() * 100
    insights.append(
        f"{low_attendance_share:.1f}% of students have attendance below 75%, which may need monitoring."
    )

    low_study_share = df["study_hours"].lt(3).mean() * 100
    insights.append(
        f"{low_study_share:.1f}% of students report fewer than 3 study hours per day."
    )

    corr = correlation_table(df)
    if not corr.empty:
        top = corr.iloc[0]
        insights.append(
            f"The strongest positive association with final performance is {top.name} "
            f"(correlation {top['Correlation with Final %']:.2f}); this is not proof of causation."
        )

    if prediction_frame is not None:
        predicted_risk = prediction_frame["predicted_risk_level"].isin(["High Risk", "Moderate Risk"]).sum()
        insights.append(
            f"The selected model flags {predicted_risk} student(s) for possible academic attention."
        )

    if feature_importance is not None and not feature_importance.empty:
        important = ", ".join(feature_importance["Feature"].head(3).tolist())
        insights.append(f"The most influential model features are currently: {important}.")

    if clustering and clustering.get("enabled"):
        best_k = clustering["best_k"]
        insights.append(f"Student segmentation is most coherent with {best_k} clusters by silhouette score.")

    return insights[:8]

