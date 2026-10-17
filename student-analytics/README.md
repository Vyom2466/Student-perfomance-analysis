# Student Analytics & Performance Prediction System

A complete Streamlit-based Python prototype for analyzing student academic data, predicting final performance, identifying students who may need academic attention, and generating practical recommendations for teachers or academic coordinators.

This project is designed for a GTU SBT / IIT Spoken Tutorial style demonstration. It focuses on a clear end-to-end workflow:

Clean data -> meaningful analysis -> appropriate machine learning -> reliable prediction -> understandable insights -> actionable recommendations.

## Problem Statement

Academic coordinators often have attendance, internal marks, assignments, practical scores, and previous performance data, but these indicators are not always converted into early decision-support signals. This project uses student data to answer:

- What is happening with student performance?
- Which indicators are associated with better or weaker outcomes?
- What does the model predict for final performance?
- Which students may need academic attention?
- What actions should teachers take?

## Objectives

- Load either the provided sample dataset or a user-uploaded CSV file.
- Validate required columns and handle common data-quality issues.
- Explore student data through statistics and visualizations.
- Train suitable machine-learning models for performance prediction.
- Compare candidate models and select the best model using relevant metrics.
- Identify potentially at-risk students.
- Generate individual student predictions and personalized suggestions.
- Present the final results through a polished dashboard.

## Features

- Streamlit multi-page interface.
- CSV upload with validation and clear error messages.
- Realistic sample dataset with 360 student records.
- Dataset overview, missing values, duplicate checks, data types, and descriptive statistics.
- Group analysis by gender, semester, internet access, and extracurricular participation.
- Correlation analysis with a clear warning that correlation is not causation.
- Regression model comparison for numerical final percentage prediction.
- Classification model comparison for at-risk prediction.
- K-Means clustering for student segmentation when enough data is available.
- Feature importance for model interpretability.
- Dashboard KPI cards, prediction summary, insights, and teacher-level recommendations.
- At-risk student table with filtering and sorting.
- Individual student prediction form with 2-4 personalized suggestions.
- Automated tests for data generation, model training, KPIs, prediction, and suggestions.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Altair
- Python unittest

## Dataset Description

The included sample dataset is saved at `data/students.csv` and contains 360 records with these columns:

- `student_id`
- `age`
- `gender`
- `semester`
- `study_hours`
- `attendance`
- `assignment_score`
- `internal_marks`
- `previous_percentage`
- `practical_score`
- `extracurricular`
- `internet_access`
- `final_percentage`

The sample data is generated with realistic relationships. For example, final performance is influenced by previous percentage, attendance, assignments, internal marks, practical score, study hours, and internet access, with random variation added so the dataset does not look artificially perfect.

The app also supports uploading a custom CSV file with the same required columns.

## Data Preprocessing

The preprocessing pipeline is reusable and prevents data leakage by fitting preprocessing only on training data through scikit-learn pipelines.

It handles:

- Duplicate rows.
- Missing numerical values using median imputation.
- Missing categorical values using most-frequent imputation.
- Categorical encoding using one-hot encoding.
- Scaling for models that benefit from it, such as linear and logistic regression.
- Reproducible train/test split using `random_state=42`.
- Feature selection that excludes `student_id`, target columns, and generated labels from training.

## Machine Learning Methodology

The project intentionally uses a small number of well-justified algorithms instead of adding many algorithms without purpose.

### Regression Task

Goal: predict numerical `final_percentage`.

Candidate models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Evaluation metrics:

- MAE
- RMSE
- R2 score

Primary selection metric: MAE, because it is easy to explain and directly shows the average prediction error in percentage points.

### Classification Task

Goal: predict whether a student may be at risk.

At-risk criteria:

- Students with final percentage below 50 are treated as at-risk for training.
- The dashboard also marks moderate concern when predicted performance is below 60.

Candidate models:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Primary selection metric: F1 score, with recall considered carefully because missing a student who needs support can be more serious than a false alert.

### Clustering Task

Goal: segment students into meaningful groups for analysis and teacher action.

Selected algorithm:

- K-Means Clustering

Evaluation:

- Silhouette score for cluster quality.
- The app tests cluster counts from 2 to 5 and selects the most coherent grouping.

Clustering is included because student segmentation can help coordinators understand different academic profiles, such as high-attendance/high-performance groups or low-attendance/moderate-performance groups.

## Why These Algorithms Were Selected

### Linear Regression

What it solves: predicts a numerical final percentage.

Why appropriate: the dataset contains several numerical academic indicators that have approximately linear relationships with final performance.

Advantages:

- Easy to explain during viva.
- Strong baseline model.
- Provides interpretable coefficients.
- Works well when relationships are mostly linear.

Limitations:

- May miss complex non-linear relationships.
- Sensitive to outliers and strong feature interactions.

### Random Forest

What it solves: predicts final percentage and at-risk status using multiple decision trees.

Why appropriate: handles mixed feature relationships and can capture non-linear patterns.

Advantages:

- Good performance on tabular data.
- Handles feature interactions.
- Provides feature importance.
- Less sensitive to outliers than simple linear models.

Limitations:

- Less transparent than linear/logistic regression.
- Can overfit if not controlled.
- More difficult to explain in detail during viva.

### Gradient Boosting

What it solves: builds sequential decision trees to improve prediction errors.

Why appropriate: useful for tabular academic data where multiple indicators combine in non-linear ways.

Advantages:

- Often strong predictive performance.
- Captures non-linear relationships.
- Supports feature importance.

Limitations:

- More tuning-sensitive.
- Less simple to explain than linear models.
- Can overfit on small datasets.

### Logistic Regression

What it solves: predicts at-risk vs not-at-risk categories.

Why appropriate: it is a strong, interpretable baseline for binary classification.

Advantages:

- Easy to explain.
- Produces class probabilities.
- Works well with scaled numerical data and encoded categories.

Limitations:

- Assumes relatively simple decision boundaries.
- May underperform when risk patterns are highly non-linear.

### K-Means Clustering

What it solves: groups students with similar academic profiles.

Why appropriate: it helps create student segments for class-level insights.

Advantages:

- Simple and explainable.
- Useful for discovering broad student profiles.
- Silhouette score gives a measurable way to choose cluster count.

Limitations:

- Requires numerical scaling.
- Cluster meanings must be interpreted carefully.
- Does not predict performance by itself.

## Current Sample Dataset Model Results

With the included 360-record sample dataset, the measured results were:

### Regression

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 5.2136 | 6.3944 | 0.7006 |
| Random Forest Regressor | 5.8139 | 7.0856 | 0.6324 |
| Gradient Boosting Regressor | 5.7753 | 7.2075 | 0.6196 |

Best regression model: Linear Regression.

Reason: it achieved the lowest MAE, meaning it had the smallest average final-percentage prediction error on the test set.

### Classification

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8778 | 0.4167 | 0.5556 | 0.4762 |
| Random Forest Classifier | 0.8889 | 0.4444 | 0.4444 | 0.4444 |
| Gradient Boosting Classifier | 0.9333 | 0.8000 | 0.4444 | 0.5714 |

Best at-risk classification model: Gradient Boosting Classifier.

Reason: it achieved the best F1 score on the test set. For this task, F1 is more useful than accuracy alone because at-risk students are a smaller group and class imbalance can make accuracy misleading.

### Clustering

K-Means selected 2 clusters using silhouette score on the sample dataset.

## Dashboard Explanation

The dashboard is the main decision-support layer. It dynamically shows:

- Total students.
- Average performance.
- Average attendance.
- Pass rate.
- At-risk students.
- High-performing students.
- Performance distribution.
- Predicted risk levels.
- Attendance and study-hour relationships with final performance.
- Best model summary.
- Key calculated insights.
- Class-level recommendations.

The dashboard is designed to answer: what is happening, what the model predicts, who needs attention, and what should be done next.

## Prediction Workflow

1. The user uploads a CSV or uses the sample dataset.
2. The dataset is validated.
3. Data is cleaned and split into training and test sets.
4. Candidate regression models are trained and compared.
5. Candidate classification models are trained and compared.
6. The best regression model predicts final percentage.
7. The best classification model predicts at-risk status.
8. Results are shown in the dashboard, at-risk table, model insights page, and individual prediction form.

## Recommendation System

Recommendations are rule-based and connected to actual student indicators and model predictions.

Examples:

- Low attendance produces an attendance improvement suggestion.
- Low study hours produces a study-routine suggestion.
- Weak assignment scores produce assignment-focused guidance.
- Weak internal marks produce targeted revision guidance.
- Strong students receive maintenance and incremental improvement suggestions.

The app returns only the most relevant 2-4 suggestions for an individual student.

## Project Structure

```text
student-analytics/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- students.csv
|-- models/
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- analysis.py
|   |-- visualization.py
|   |-- ml_models.py
|   |-- insights.py
|   `-- recommendations.py
`-- tests/
    `-- test_basic.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## How to Test

Run:

```bash
python -m unittest discover -s tests
```

The tests validate:

- Sample dataset generation.
- Required column validation.
- Model training.
- Prediction output range.
- Dashboard KPI calculation.
- Personalized recommendation logic.

## Error Handling

The app handles:

- Missing files.
- Empty datasets.
- Invalid CSV files.
- Missing required columns.
- Invalid numeric values.
- Duplicate rows.
- Insufficient records.
- Model-training failures.
- Prediction failures.

When a user-uploaded CSV is invalid, the app shows clear errors and safely falls back to the sample dataset.

## Limitations

- The included dataset is synthetic, so model results demonstrate methodology rather than real institutional performance.
- Predictions are decision-support signals, not guaranteed outcomes.
- Correlation and feature importance do not prove causation.
- The model should be retrained and validated with real student data before real academic decisions.
- Sensitive student data should be handled with proper privacy and permission controls in a production system.

## Future Enhancements

- Add authentication for teachers and coordinators.
- Store model versions and training history.
- Add subject-wise marks and semester-wise progression.
- Export at-risk student reports to PDF or Excel.
- Add explainability using permutation importance or SHAP.
- Add deployment configuration for institutional use.
