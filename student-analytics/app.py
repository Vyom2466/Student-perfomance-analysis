from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis import (
    at_risk_students,
    correlation_table,
    dashboard_kpis,
    dataset_overview,
    descriptive_statistics,
    group_analysis,
)
from src.data_loader import (
    REQUIRED_COLUMNS,
    load_sample_data,
    read_csv_safely,
    required_columns_text,
    validate_dataset,
)
from src.insights import generate_key_insights
from src.ml_models import predict_single_student, train_all_models
from src.recommendations import class_level_recommendations, personalized_suggestions
from src.visualization import (
    attendance_distribution,
    category_distribution,
    feature_importance_chart,
    model_comparison,
    performance_distribution,
    prediction_results_chart,
    relationship_scatter,
    study_hours_distribution,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "students.csv"


st.set_page_config(
    page_title="Student Analytics & Performance Prediction",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    h1, h2, h3 {letter-spacing: 0;}
    .dashboard-hero {
        border: 1px solid rgba(120, 170, 210, 0.26);
        border-radius: 8px;
        padding: 30px 24px 22px 24px;
        margin-bottom: 18px;
        background:
            linear-gradient(135deg, rgba(36, 84, 166, 0.28), rgba(0, 166, 166, 0.12)),
            rgba(14, 20, 30, 0.82);
    }
    .dashboard-title {
        margin: 0;
        color: #F8FAFC;
        font-size: clamp(2rem, 3.5vw, 2.85rem);
        line-height: 1.18;
        font-weight: 800;
    }
    .dashboard-subtitle {
        margin-top: 10px;
        color: #B9C3D3;
        max-width: 880px;
        font-size: 1.02rem;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
        gap: 12px;
        margin: 8px 0 18px 0;
    }
    .kpi-card {
        position: relative;
        overflow: hidden;
        min-height: 122px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 16px 16px 14px 16px;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 5px;
        background: var(--accent);
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #4B5A6C;
        margin-bottom: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-value {
        font-size: 1.58rem;
        line-height: 1.1;
        font-weight: 800;
        color: #0E1B2A;
    }
    .kpi-help {
        margin-top: 9px;
        font-size: 0.78rem;
        color: #64748B;
    }
    .status-note {
        display: grid;
        grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr) minmax(220px, 1fr);
        gap: 12px;
        padding: 14px;
        background: rgba(248, 250, 252, 0.96);
        color: #102033;
        border: 1px solid rgba(36, 84, 166, 0.20);
        border-left: 5px solid #00A6A6;
        border-radius: 8px;
        margin: 4px 0 22px 0;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.14);
    }
    .model-chip {
        border-radius: 8px;
        padding: 10px 12px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
    }
    .model-label {
        color: #64748B;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .model-value {
        color: #0F172A;
        font-weight: 800;
        font-size: 1rem;
        line-height: 1.25;
    }
    .model-note {
        color: #475569;
        font-size: 0.82rem;
        margin-top: 4px;
    }
    @media (max-width: 900px) {
        .status-note {grid-template-columns: 1fr;}
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 12px 14px;
        background: rgba(255,255,255,0.02);
        color: #F8FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_sample_data() -> pd.DataFrame:
    return load_sample_data(SAMPLE_DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_trained_models(df: pd.DataFrame):
    return train_all_models(df)


def main() -> None:
    st.sidebar.title("Student Analytics")
    st.sidebar.caption("Performance prediction and academic decision support")

    df, source_name = load_current_dataset()
    page = st.sidebar.radio(
        "Pages",
        [
            "Dashboard",
            "Data Explorer",
            "Data Analysis",
            "ML Prediction",
            "Student Prediction",
            "At-Risk Students",
            "Model Insights",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Active dataset: {source_name}")
    st.sidebar.caption(f"{len(df)} student records")

    try:
        model_result = get_trained_models(df)
    except Exception as exc:
        model_result = None
        st.error(f"Model training could not be completed: {exc}")

    if page == "Dashboard":
        render_dashboard(df, model_result)
    elif page == "Data Explorer":
        render_data_explorer(df, source_name)
    elif page == "Data Analysis":
        render_data_analysis(df)
    elif page == "ML Prediction":
        render_ml_prediction(model_result)
    elif page == "Student Prediction":
        render_student_prediction(df, model_result)
    elif page == "At-Risk Students":
        render_at_risk(model_result)
    elif page == "Model Insights":
        render_model_insights(model_result)


def load_current_dataset() -> tuple[pd.DataFrame, str]:
    uploaded_file = st.sidebar.file_uploader("Upload student CSV", type=["csv"])
    if uploaded_file is None:
        sample = get_sample_data()
        validation = validate_dataset(sample)
        if not validation.ok:
            st.stop()
        return validation.dataframe, "Sample dataset"

    validation = read_csv_safely(uploaded_file)
    if not validation.ok:
        st.sidebar.error("Uploaded CSV is not valid.")
        for error in validation.errors:
            st.sidebar.error(error)
        st.sidebar.info("Required columns: " + required_columns_text())
        sample = get_sample_data()
        return sample, "Sample dataset"

    for warning in validation.warnings:
        st.sidebar.warning(warning)
    return validation.dataframe, uploaded_file.name


def render_dashboard(df: pd.DataFrame, model_result) -> None:
    st.markdown(
        """
        <div class="dashboard-hero">
            <h1 class="dashboard-title">Student Performance Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prediction_frame = model_result.prediction_frame if model_result else None
    render_kpi_cards(dashboard_kpis(df, prediction_frame))

    if model_result:
        best_row = model_result.regression_results[
            model_result.regression_results["Model"] == model_result.best_regression_name
        ].iloc[0]
        cls_row = model_result.classification_results[
            model_result.classification_results["Model"] == model_result.best_classification_name
        ].iloc[0]
        render_model_summary(model_result, best_row, cls_row)

    left, right = st.columns(2)
    with left:
        st.altair_chart(performance_distribution(df), use_container_width=True)
    with right:
        if model_result:
            st.altair_chart(prediction_results_chart(model_result.prediction_frame), use_container_width=True)
        else:
            st.altair_chart(category_distribution(df), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.altair_chart(
            relationship_scatter(df, "attendance", "Attendance vs Final Performance"),
            use_container_width=True,
        )
    with right:
        st.altair_chart(
            relationship_scatter(df, "study_hours", "Study Hours vs Final Performance"),
            use_container_width=True,
        )

    st.subheader("Key Insights")
    insights = generate_key_insights(
        df,
        prediction_frame=prediction_frame,
        feature_importance=model_result.feature_importance if model_result else None,
        clustering=model_result.clustering if model_result else None,
    )
    for insight in insights:
        st.write(f"- {insight}")

    st.subheader("Class-Level Recommendations")
    for recommendation in class_level_recommendations(df, prediction_frame):
        st.write(f"- {recommendation}")


def render_data_explorer(df: pd.DataFrame, source_name: str) -> None:
    st.title("Data Explorer")
    st.caption("Upload, validate, preview, and understand the active student dataset.")

    st.info("Required CSV columns: " + ", ".join(REQUIRED_COLUMNS))
    st.write(f"Active dataset: **{source_name}**")
    st.dataframe(df.head(30), use_container_width=True)

    overview = dataset_overview(df)
    left, middle, right = st.columns(3)
    left.metric("Records", overview["records"])
    middle.metric("Features", overview["features"])
    right.metric("Missing Values", overview["missing_values"])

    st.subheader("Data Types")
    st.dataframe(
        pd.DataFrame(overview["data_types"].items(), columns=["Column", "Data Type"]),
        use_container_width=True,
    )

    st.subheader("Missing Values by Column")
    missing = df.isna().sum().rename("Missing Values").reset_index().rename(columns={"index": "Column"})
    st.dataframe(missing, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(descriptive_statistics(df), use_container_width=True)


def render_data_analysis(df: pd.DataFrame) -> None:
    st.title("Data Analysis")
    st.caption("Exploratory analysis of academic and behavioral indicators.")

    left, right = st.columns(2)
    with left:
        st.altair_chart(attendance_distribution(df), use_container_width=True)
        st.altair_chart(study_hours_distribution(df), use_container_width=True)
    with right:
        st.altair_chart(category_distribution(df), use_container_width=True)
        st.altair_chart(
            relationship_scatter(df, "previous_percentage", "Previous vs Final Performance"),
            use_container_width=True,
        )

    st.subheader("Correlation Analysis")
    st.caption("Correlation shows association, not causation.")
    st.dataframe(correlation_table(df), use_container_width=True)

    st.subheader("Group-Based Analysis")
    for group_name, table in group_analysis(df).items():
        with st.expander(f"Final performance by {group_name}"):
            st.dataframe(table, use_container_width=True)


def render_ml_prediction(model_result) -> None:
    st.title("ML Prediction")
    st.caption("Candidate models are trained, evaluated, compared, and selected using appropriate metrics.")

    if model_result is None:
        st.warning("Models are not available for this dataset.")
        return

    st.subheader("Candidate Models -> Evaluation -> Best Model")
    st.write(model_result.explanation)

    left, right = st.columns(2)
    with left:
        st.markdown("**Regression: final percentage prediction**")
        st.dataframe(model_result.regression_results, use_container_width=True)
        st.altair_chart(
            model_comparison(model_result.regression_results, "MAE", "Regression Model Comparison"),
            use_container_width=True,
        )
    with right:
        st.markdown("**Classification: at-risk prediction**")
        display_cls = model_result.classification_results.drop(columns=["Confusion Matrix"], errors="ignore")
        st.dataframe(display_cls, use_container_width=True)
        st.altair_chart(
            model_comparison(model_result.classification_results, "F1 Score", "At-Risk Model Comparison"),
            use_container_width=True,
        )

    st.subheader("Final Selected Models")
    st.write(f"- Final percentage model: **{model_result.best_regression_name}**")
    st.write(f"- At-risk model: **{model_result.best_classification_name}**")


def render_student_prediction(df: pd.DataFrame, model_result) -> None:
    st.title("Student Prediction")
    st.caption("Enter an individual student's indicators to estimate performance and get targeted suggestions.")

    if model_result is None:
        st.warning("Prediction is unavailable because model training did not complete.")
        return

    med = df.median(numeric_only=True)
    with st.form("student_prediction_form"):
        left, middle, right = st.columns(3)
        with left:
            age = st.number_input("Age", min_value=15, max_value=35, value=int(med.get("age", 20)))
            semester = st.number_input("Semester", min_value=1, max_value=10, value=int(med.get("semester", 4)))
            gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique().tolist()))
        with middle:
            study_hours = st.number_input("Study hours per day", 0.0, 16.0, float(med.get("study_hours", 4.0)), 0.5)
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, float(med.get("attendance", 75.0)), 1.0)
            assignment_score = st.number_input("Assignment score", 0.0, 100.0, float(med.get("assignment_score", 65.0)), 1.0)
        with right:
            internal_marks = st.number_input("Internal marks", 0.0, 100.0, float(med.get("internal_marks", 65.0)), 1.0)
            previous_percentage = st.number_input("Previous percentage", 0.0, 100.0, float(med.get("previous_percentage", 65.0)), 1.0)
            practical_score = st.number_input("Practical score", 0.0, 100.0, float(med.get("practical_score", 65.0)), 1.0)

        left, right = st.columns(2)
        with left:
            extracurricular = st.selectbox("Extracurricular", sorted(df["extracurricular"].dropna().unique().tolist()))
        with right:
            internet_access = st.selectbox("Internet access", sorted(df["internet_access"].dropna().unique().tolist()))

        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        student = {
            "age": age,
            "semester": semester,
            "study_hours": study_hours,
            "attendance": attendance,
            "assignment_score": assignment_score,
            "internal_marks": internal_marks,
            "previous_percentage": previous_percentage,
            "practical_score": practical_score,
            "gender": gender,
            "extracurricular": extracurricular,
            "internet_access": internet_access,
        }
        prediction = predict_single_student(model_result, student)
        left, middle, right = st.columns(3)
        left.metric("Predicted Performance", f"{prediction['predicted_percentage']:.1f}%")
        middle.metric("Category", prediction["performance_category"])
        right.metric("Risk Level", prediction["risk_level"])
        if prediction["confidence"] is not None:
            st.caption(f"Classification confidence: {prediction['confidence'] * 100:.1f}%")

        st.subheader("Personalized Suggestions")
        for suggestion in personalized_suggestions(student, prediction):
            st.write(f"- {suggestion}")


def render_at_risk(model_result) -> None:
    st.title("At-Risk Students")
    st.caption("Predictions are decision-support signals, not guaranteed outcomes.")

    if model_result is None:
        st.warning("At-risk analysis is unavailable because model training did not complete.")
        return

    risk_df = at_risk_students(model_result.prediction_frame)
    if risk_df.empty:
        st.success("No students are currently flagged as high or moderate risk.")
        return

    selected_levels = st.multiselect(
        "Risk levels",
        ["High Risk", "Moderate Risk", "Low Risk"],
        default=["High Risk", "Moderate Risk"],
    )
    filtered = risk_df[risk_df["predicted_risk_level"].isin(selected_levels)]
    sort_by = st.selectbox(
        "Sort by",
        [col for col in ["risk_probability", "predicted_percentage", "attendance", "study_hours"] if col in filtered.columns],
    )
    ascending = sort_by != "risk_probability"
    st.dataframe(filtered.sort_values(sort_by, ascending=ascending), use_container_width=True)


def render_model_insights(model_result) -> None:
    st.title("Model Insights")
    st.caption("Interpretability and segmentation views for explaining the selected models.")

    if model_result is None:
        st.warning("Model insights are unavailable because model training did not complete.")
        return

    st.subheader("Feature Importance")
    st.caption("Feature importance helps explain the model; it is not proof of causation.")
    if model_result.feature_importance.empty:
        st.info("The selected model does not expose direct feature importance.")
    else:
        st.altair_chart(feature_importance_chart(model_result.feature_importance), use_container_width=True)
        st.dataframe(model_result.feature_importance, use_container_width=True)

    st.subheader("Confusion Matrix for At-Risk Model")
    best_row = model_result.classification_results[
        model_result.classification_results["Model"] == model_result.best_classification_name
    ].iloc[0]
    matrix = pd.DataFrame(best_row["Confusion Matrix"], index=["Actual Not Risk", "Actual Risk"], columns=["Predicted Not Risk", "Predicted Risk"])
    st.dataframe(matrix, use_container_width=True)

    st.subheader("Student Segmentation")
    clustering = model_result.clustering
    if clustering.get("enabled"):
        st.write(f"Best cluster count by silhouette score: **{clustering['best_k']}**")
        st.dataframe(clustering["scores"], use_container_width=True)
        st.dataframe(clustering["profile"], use_container_width=True)
    else:
        st.info(clustering.get("reason", "Segmentation is unavailable for this dataset."))


def render_kpi_cards(kpis: dict[str, object]) -> None:
    labels = {
        "Total Students": ("Class Size", "Validated student records", "#2454A6"),
        "Average Performance": ("Average Score", "Mean final percentage", "#00A6A6"),
        "Average Attendance": ("Attendance", "Class attendance average", "#F28E2B"),
        "Pass Rate": ("Pass Rate", "Students scoring 50% or above", "#59A14F"),
        "At-Risk Students": ("Needs Support", "High + moderate predicted risk", "#D45087"),
        "High-Performing Students": ("High Performers", "Predicted or actual 75%+", "#6F4E7C"),
    }
    html = ['<div class="kpi-grid">']
    for label, value in kpis.items():
        display_label, help_text, accent = labels.get(label, (label, "", "#2454A6"))
        html.append(
            (
                f'<div class="kpi-card" style="--accent:{accent}">'
                f'<div class="kpi-label">{display_label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-help">{help_text}</div>'
                "</div>"
            )
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_model_summary(model_result, best_row: pd.Series, cls_row: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="status-note">
            <div class="model-chip">
                <div class="model-label">Score Prediction</div>
                <div class="model-value">{model_result.best_regression_name}</div>
                <div class="model-note">Average error: {best_row['MAE']:.2f} percentage points</div>
            </div>
            <div class="model-chip">
                <div class="model-label">Risk Detection</div>
                <div class="model-value">{model_result.best_classification_name}</div>
                <div class="model-note">F1 score: {cls_row['F1 Score']:.2f}; recall: {cls_row['Recall']:.2f}</div>
            </div>
            <div class="model-chip">
                <div class="model-label">Decision Rule</div>
                <div class="model-value">Flag students needing attention</div>
                <div class="model-note">High risk below 50%; watchlist below 60%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
