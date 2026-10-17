from __future__ import annotations

import unittest

from src.analysis import dashboard_kpis
from src.data_loader import REQUIRED_COLUMNS, generate_sample_dataset, validate_dataset
from src.ml_models import predict_single_student, train_all_models
from src.recommendations import personalized_suggestions


class StudentAnalyticsTests(unittest.TestCase):
    def test_sample_dataset_is_valid(self):
        df = generate_sample_dataset(records=320)
        validation = validate_dataset(df)
        self.assertTrue(validation.ok)
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(df.columns))
        self.assertGreaterEqual(len(df), 300)

    def test_model_training_and_predictions_work(self):
        df = generate_sample_dataset(records=320)
        result = train_all_models(df)
        self.assertFalse(result.regression_results.empty)
        self.assertFalse(result.classification_results.empty)
        self.assertIn("predicted_percentage", result.prediction_frame.columns)
        self.assertTrue(result.prediction_frame["predicted_percentage"].between(0, 100).all())

    def test_dashboard_kpis_are_dynamic(self):
        df = generate_sample_dataset(records=320)
        result = train_all_models(df)
        kpis = dashboard_kpis(df, result.prediction_frame)
        self.assertEqual(kpis["Total Students"], 320)
        self.assertIn("Average Performance", kpis)
        self.assertIn("At-Risk Students", kpis)

    def test_personalized_prediction_and_suggestions(self):
        df = generate_sample_dataset(records=320)
        result = train_all_models(df)
        student = {
            "age": 19,
            "semester": 4,
            "study_hours": 1.5,
            "attendance": 58,
            "assignment_score": 52,
            "internal_marks": 55,
            "previous_percentage": 50,
            "practical_score": 57,
            "gender": "Female",
            "extracurricular": "No",
            "internet_access": "Yes",
        }
        prediction = predict_single_student(result, student)
        suggestions = personalized_suggestions(student, prediction)
        self.assertGreaterEqual(prediction["predicted_percentage"], 0)
        self.assertLessEqual(prediction["predicted_percentage"], 100)
        self.assertGreaterEqual(len(suggestions), 2)
        self.assertLessEqual(len(suggestions), 4)


if __name__ == "__main__":
    unittest.main()
