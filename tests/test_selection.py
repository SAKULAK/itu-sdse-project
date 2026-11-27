import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from bsops_the_final.deployment import selection

MODULE = "bsops_the_final.deployment.selection"


class FakeModelVersion:
    """Mimics MLflow's ModelVersion: dict(mv) yields its fields."""

    def __init__(self, current_stage, version, run_id):
        self.current_stage = current_stage
        self.version = version
        self.run_id = run_id

    def __iter__(self):
        return iter(vars(self).items())


class TestGetBestRun(unittest.TestCase):

    @patch(f"{MODULE}.mlflow")
    def test_returns_top_run_ordered_by_f1(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="42")
        mock_mlflow.search_runs.return_value = pd.DataFrame(
            [{"run_id": "best", "metrics.test_f1_score": 0.9}]
        )

        best = selection.get_best_run("my_experiment")

        self.assertEqual(best["run_id"], "best")
        mock_mlflow.search_runs.assert_called_once_with(
            experiment_ids=["42"],
            order_by=["metrics.test_f1_score DESC"],
            max_results=1,
        )


class TestGetProductionModel(unittest.TestCase):

    @patch(f"{MODULE}.client")
    def test_uses_production_alias(self, mock_client):
        mock_client.get_model_version_by_alias.return_value = SimpleNamespace(version="3", run_id="r3")

        uri, version, run_id = selection.get_production_model()

        self.assertEqual(uri, f"models:/{selection.model_name}@Production")
        self.assertEqual((version, run_id), ("3", "r3"))

    @patch(f"{MODULE}.client")
    def test_falls_back_to_production_stage(self, mock_client):
        mock_client.get_model_version_by_alias.side_effect = Exception("no alias")
        staging = FakeModelVersion("Staging", version="1", run_id="r1")
        prod = FakeModelVersion("Production", version="2", run_id="r2")
        mock_client.search_model_versions.return_value = [staging, prod]

        uri, version, run_id = selection.get_production_model()

        self.assertEqual(uri, f"models:/{selection.model_name}/2")
        self.assertEqual((version, run_id), ("2", "r2"))

    @patch(f"{MODULE}.client")
    def test_returns_nones_when_nothing_in_production(self, mock_client):
        mock_client.get_model_version_by_alias.side_effect = Exception("no alias")
        mock_client.search_model_versions.return_value = []

        self.assertEqual(selection.get_production_model(), (None, None, None))


class TestCompareWithProduction(unittest.TestCase):

    def _prod_run(self, metrics):
        return SimpleNamespace(data=SimpleNamespace(metrics=metrics))

    def test_no_production_promotes_trained_model(self):
        best = pd.Series({"run_id": "new", "metrics.test_f1_score": 0.5})
        self.assertEqual(selection.compare_with_production(best, None, None), "new")

    @patch(f"{MODULE}.mlflow")
    def test_better_model_is_selected(self, mock_mlflow):
        mock_mlflow.get_run.return_value = self._prod_run({"test_f1_score": 0.7})
        best = pd.Series({"run_id": "new", "metrics.test_f1_score": 0.9})

        run_id = selection.compare_with_production(best, "models:/x@Production", "old")

        self.assertEqual(run_id, "new")
        mock_mlflow.get_run.assert_called_once_with("old")

    @patch(f"{MODULE}.mlflow")
    def test_worse_model_is_not_selected(self, mock_mlflow):
        mock_mlflow.get_run.return_value = self._prod_run({"test_f1_score": 0.95})
        best = pd.Series({"run_id": "new", "metrics.test_f1_score": 0.9})

        self.assertIsNone(selection.compare_with_production(best, "models:/x@Production", "old"))

    @patch(f"{MODULE}.mlflow")
    def test_equal_score_keeps_production(self, mock_mlflow):
        mock_mlflow.get_run.return_value = self._prod_run({"test_f1_score": 0.9})
        best = pd.Series({"run_id": "new", "metrics.test_f1_score": 0.9})

        self.assertIsNone(selection.compare_with_production(best, "models:/x@Production", "old"))

    @patch(f"{MODULE}.mlflow")
    def test_falls_back_to_f1_score_metric(self, mock_mlflow):
        mock_mlflow.get_run.return_value = self._prod_run({"f1_score": 0.6})
        best = pd.Series({"run_id": "new", "metrics.f1_score": 0.8})

        self.assertEqual(selection.compare_with_production(best, "models:/x@Production", "old"), "new")


if __name__ == "__main__":
    unittest.main()
