from __future__ import annotations

import altair as alt
import pandas as pd


COLOR_RANGE = ["#2454A6", "#00A6A6", "#F28E2B", "#D45087", "#59A14F", "#6F4E7C"]


def performance_distribution(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_bar(color=COLOR_RANGE[0])
        .encode(
            x=alt.X("final_percentage:Q", bin=alt.Bin(maxbins=24), title="Final Percentage"),
            y=alt.Y("count():Q", title="Students"),
            tooltip=[alt.Tooltip("count():Q", title="Students")],
        )
        .properties(title="Performance Distribution")
    )


def attendance_distribution(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_bar(color=COLOR_RANGE[1])
        .encode(
            x=alt.X("attendance:Q", bin=alt.Bin(maxbins=20), title="Attendance (%)"),
            y=alt.Y("count():Q", title="Students"),
            tooltip=[alt.Tooltip("count():Q", title="Students")],
        )
        .properties(title="Attendance Distribution")
    )


def study_hours_distribution(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_bar(color=COLOR_RANGE[2])
        .encode(
            x=alt.X("study_hours:Q", bin=alt.Bin(maxbins=18), title="Study Hours per Day"),
            y=alt.Y("count():Q", title="Students"),
            tooltip=[alt.Tooltip("count():Q", title="Students")],
        )
        .properties(title="Study Hours Distribution")
    )


def category_distribution(df: pd.DataFrame):
    counts = (
        df["performance_category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Students")
    )
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("Category:N", sort=None),
            y="Students:Q",
            color=alt.Color("Category:N", scale=alt.Scale(range=COLOR_RANGE), legend=None),
            tooltip=["Category:N", "Students:Q"],
        )
        .properties(title="Performance Category Distribution")
    )


def relationship_scatter(df: pd.DataFrame, x: str, title: str):
    return (
        alt.Chart(df)
        .mark_circle(size=64, opacity=0.75)
        .encode(
            x=alt.X(f"{x}:Q", title=x.replace("_", " ").title()),
            y=alt.Y("final_percentage:Q", title="Final Percentage"),
            color=alt.Color(
                "performance_category:N",
                scale=alt.Scale(range=COLOR_RANGE),
                title="Category",
            ),
            tooltip=[
                "student_id:N",
                alt.Tooltip(f"{x}:Q", format=".1f"),
                alt.Tooltip("final_percentage:Q", format=".1f"),
                "performance_category:N",
            ],
        )
        .properties(title=title)
    )


def model_comparison(results: pd.DataFrame, metric: str, title: str):
    ascending = metric in ["MAE", "RMSE"]
    sorted_results = results.sort_values(metric, ascending=ascending)
    return (
        alt.Chart(sorted_results)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=sorted_results["Model"].tolist(), title="Model"),
            y=alt.Y(f"{metric}:Q", title=metric),
            color=alt.Color("Model:N", scale=alt.Scale(range=COLOR_RANGE), legend=None),
            tooltip=["Model:N", alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
        .properties(title=title)
    )


def feature_importance_chart(importance: pd.DataFrame):
    return (
        alt.Chart(importance.sort_values("Importance"))
        .mark_bar(color=COLOR_RANGE[3])
        .encode(
            x=alt.X("Importance:Q", title="Importance"),
            y=alt.Y("Feature:N", sort=None, title="Feature"),
            tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".4f")],
        )
        .properties(title="Model Feature Importance")
    )


def prediction_results_chart(prediction_frame: pd.DataFrame):
    counts = (
        prediction_frame["predicted_risk_level"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Students")
    )
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("Risk Level:N", sort=None),
            y="Students:Q",
            color=alt.Color("Risk Level:N", scale=alt.Scale(range=COLOR_RANGE), legend=None),
            tooltip=["Risk Level:N", "Students:Q"],
        )
        .properties(title="Predicted Risk Levels")
    )

