import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bsops_the_final.deployment import pipeline

MODULE = "bsops_the_final.deployment.pipeline"


class TestPipelineMain(unittest.TestCase):

    def test_runs_steps_in_order_and_passes_results_along(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "model_results.json"
            results_path.write_text(json.dumps({"classification_report": {"accuracy": 0.9}}))

            with patch(f"{MODULE}.artifact_dir", str(Path(tmp) / "artifacts")), \
                 patch(f"{MODULE}.mlflow_state_dir", str(Path(tmp) / "mlflow-state")), \
                 patch(f"{MODULE}.model_results_path", str(results_path)), \
                 patch(f"{MODULE}.setup_mlflow_tracking") as setup, \
                 patch(f"{MODULE}.mlflow"), \
                 patch(f"{MODULE}.get_best_run", return_value="best-run") as get_best, \
                 patch(f"{MODULE}.get_production_model", return_value=("uri", "1", "prod-run")), \
                 patch(f"{MODULE}.compare_with_production", return_value="run-x") as compare, \
                 patch(f"{MODULE}.promote_model", return_value="2") as promote, \
                 patch(f"{MODULE}.export_model") as export:
                pipeline.main()

                self.assertTrue((Path(tmp) / "artifacts").is_dir())
                self.assertTrue((Path(tmp) / "mlflow-state").is_dir())

        setup.assert_called_once_with(pipeline.experiment_name)
        get_best.assert_called_once_with(pipeline.experiment_name)
        compare.assert_called_once_with("best-run", "uri", "prod-run")
        promote.assert_called_once_with("run-x")
        export.assert_called_once_with("2")


if __name__ == "__main__":
    unittest.main()
