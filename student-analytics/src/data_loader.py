from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "student_id",
    "age",
    "gender",
    "semester",
    "study_hours",
    "attendance",
    "assignment_score",
    "internal_marks",
    "previous_percentage",
    "practical_score",
    "extracurricular",
    "internet_access",
    "final_percentage",
]

NUMERIC_COLUMNS = [
    "age",
    "semester",
    "study_hours",
    "attendance",
    "assignment_score",
    "internal_marks",
    "previous_percentage",
    "practical_score",
    "final_percentage",
]

CATEGORICAL_COLUMNS = ["gender", "extracurricular", "internet_access"]
TARGET_COLUMN = "final_percentage"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    dataframe: pd.DataFrame | None = None


def generate_sample_dataset(records: int = 360, random_state: int = 42) -> pd.DataFrame:
    """Create realistic synthetic student data with useful predictive signal."""
    rng = np.random.default_rng(random_state)

    student_id = [f"STU{1001 + i}" for i in range(records)]
    age = rng.integers(18, 24, size=records)
    gender = rng.choice(["Female", "Male", "Other"], size=records, p=[0.47, 0.5, 0.03])
    semester = rng.integers(1, 9, size=records)
    internet_access = rng.choice(["Yes", "No"], size=records, p=[0.83, 0.17])
    extracurricular = rng.choice(["Yes", "No"], size=records, p=[0.38, 0.62])

    previous_percentage = np.clip(rng.normal(66, 13, records), 32, 95)
    attendance = np.clip(
        52 + 0.22 * previous_percentage + rng.normal(0, 10, records), 35, 100
    )
    study_hours = np.clip(
        1.0
        + 0.045 * previous_percentage
        + np.where(internet_access == "Yes", 0.45, -0.25)
        + rng.normal(0, 1.15, records),
        0.5,
        8.5,
    )
    assignment_score = np.clip(
        0.47 * previous_percentage + 4.0 * study_hours + rng.normal(0, 8, records),
        25,
        100,
    )
    internal_marks = np.clip(
        0.45 * previous_percentage
        + 0.22 * attendance
        + 1.7 * study_hours
        + rng.normal(0, 7, records),
        20,
        100,
    )
    practical_score = np.clip(
        0.42 * previous_percentage
        + 0.22 * assignment_score
        + np.where(extracurricular == "Yes", 2.5, 0)
        + rng.normal(0, 8, records),
        25,
        100,
    )
    final_percentage = np.clip(
        0.27 * previous_percentage
        + 0.18 * attendance
        + 0.18 * assignment_score
        + 0.2 * internal_marks
        + 0.12 * practical_score
        + 1.8 * study_hours
        + np.where(internet_access == "Yes", 1.5, -1.2)
        + rng.normal(0, 5.5, records)
        + 3,
        25,
        100,
    )

    df = pd.DataFrame(
        {
            "student_id": student_id,
            "age": age,
            "gender": gender,
            "semester": semester,
            "study_hours": np.round(study_hours, 1),
            "attendance": np.round(attendance, 1),
            "assignment_score": np.round(assignment_score, 1),
            "internal_marks": np.round(internal_marks, 1),
            "previous_percentage": np.round(previous_percentage, 1),
            "practical_score": np.round(practical_score, 1),
            "extracurricular": extracurricular,
            "internet_access": internet_access,
            "final_percentage": np.round(final_percentage, 1),
        }
    )
    return add_performance_columns(df)


def save_sample_dataset(path: str | Path, records: int = 360) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_dataset(records=records).to_csv(output, index=False)
    return output


def load_sample_data(path: str | Path = "data/students.csv") -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        save_sample_dataset(csv_path)
    return add_performance_columns(pd.read_csv(csv_path))


def read_csv_safely(file_or_path) -> ValidationResult:
    try:
        df = pd.read_csv(file_or_path)
    except pd.errors.EmptyDataError:
        return ValidationResult(False, ["The CSV file is empty."], [])
    except Exception as exc:
        return ValidationResult(False, [f"Could not read CSV file: {exc}"], [])
    return validate_dataset(df)


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        return ValidationResult(False, ["Dataset is empty."], [])

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns))
        return ValidationResult(False, errors, warnings)

    cleaned = df.copy()
    duplicate_count = int(cleaned.duplicated().sum())
    if duplicate_count:
        warnings.append(f"Removed {duplicate_count} duplicate row(s).")
        cleaned = cleaned.drop_duplicates()

    for column in NUMERIC_COLUMNS:
        before_missing = int(cleaned[column].isna().sum())
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        after_missing = int(cleaned[column].isna().sum())
        introduced = after_missing - before_missing
        if introduced > 0:
            warnings.append(
                f"{introduced} invalid value(s) in '{column}' were treated as missing."
            )

    if len(cleaned) < 30:
        errors.append("At least 30 valid records are required for model training.")

    _validate_ranges(cleaned, errors, warnings)

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip().fillna("Unknown")
        cleaned.loc[cleaned[column].eq(""), column] = "Unknown"

    cleaned["student_id"] = cleaned["student_id"].astype("string").str.strip()
    cleaned = add_performance_columns(cleaned)
    return ValidationResult(len(errors) == 0, errors, warnings, cleaned)


def add_performance_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if TARGET_COLUMN in result.columns:
        result["performance_category"] = pd.cut(
            result[TARGET_COLUMN],
            bins=[-np.inf, 50, 75, np.inf],
            labels=["At Risk", "Average", "High"],
        ).astype("object")
        result["actual_risk"] = np.where(result[TARGET_COLUMN] < 50, 1, 0)
    return result


def required_columns_text(columns: Iterable[str] = REQUIRED_COLUMNS) -> str:
    return ", ".join(columns)


def _validate_ranges(df: pd.DataFrame, errors: list[str], warnings: list[str]) -> None:
    range_rules = {
        "age": (15, 35),
        "semester": (1, 10),
        "study_hours": (0, 16),
        "attendance": (0, 100),
        "assignment_score": (0, 100),
        "internal_marks": (0, 100),
        "previous_percentage": (0, 100),
        "practical_score": (0, 100),
        "final_percentage": (0, 100),
    }
    for column, (low, high) in range_rules.items():
        invalid = df[column].notna() & ~df[column].between(low, high)
        invalid_count = int(invalid.sum())
        if invalid_count:
            warnings.append(
                f"{invalid_count} value(s) in '{column}' are outside the expected "
                f"{low}-{high} range and will be imputed or clipped during modeling."
            )


if __name__ == "__main__":
    save_sample_dataset(Path(__file__).resolve().parents[1] / "data" / "students.csv")
