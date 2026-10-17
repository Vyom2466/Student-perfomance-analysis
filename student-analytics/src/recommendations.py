from __future__ import annotations

from typing import Any

import pandas as pd


def personalized_suggestions(student: dict[str, Any], prediction: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []

    if float(student.get("attendance", 0)) < 75:
        suggestions.append("Improve attendance through regular class participation and early follow-up after missed lectures.")
    if float(student.get("study_hours", 0)) < 3:
        suggestions.append("Build a consistent study routine of at least 3 focused hours per day during active academic weeks.")
    if float(student.get("assignment_score", 0)) < 60:
        suggestions.append("Prioritize assignments because they show weak concept application and can improve with guided practice.")
    if float(student.get("internal_marks", 0)) < 60:
        suggestions.append("Plan targeted revision for internal assessment topics and review previous test mistakes.")
    if float(student.get("practical_score", 0)) < 60:
        suggestions.append("Schedule extra practical/lab practice to strengthen hands-on understanding.")
    if float(student.get("previous_percentage", 0)) < 55:
        suggestions.append("Provide mentoring support because previous academic performance indicates a weaker base.")

    if not suggestions and prediction.get("risk_level") == "Low Risk":
        suggestions.append("Maintain current habits and use weekly self-review to protect existing strengths.")
        suggestions.append("Choose one weaker subject area for incremental improvement rather than changing the whole routine.")

    return suggestions[:4]


def class_level_recommendations(df: pd.DataFrame, prediction_frame: pd.DataFrame | None = None) -> list[str]:
    recommendations: list[str] = []
    avg_attendance = df["attendance"].mean()
    avg_study_hours = df["study_hours"].mean()
    avg_assignment = df["assignment_score"].mean()
    avg_internal = df["internal_marks"].mean()

    if avg_attendance < 75:
        recommendations.append("Strengthen attendance monitoring and contact students whose attendance is below 75%.")
    if avg_study_hours < 3:
        recommendations.append("Introduce weekly study-planning check-ins because average study hours are low.")
    if avg_assignment < 65:
        recommendations.append("Run assignment improvement sessions focused on common mistakes and submission quality.")
    if avg_internal < 65:
        recommendations.append("Arrange targeted revision before internal assessments for weaker topics.")

    if prediction_frame is not None:
        risk_share = prediction_frame["predicted_risk_level"].isin(["High Risk", "Moderate Risk"]).mean()
        if risk_share >= 0.25:
            recommendations.append("Create a priority mentoring list because a meaningful share of students are predicted to need support.")

    if not recommendations:
        recommendations.append("Maintain the current academic support plan and monitor moderate-risk students weekly.")
    return recommendations[:5]

