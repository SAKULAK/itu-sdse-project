import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from bsops_the_final.deployment import promote

MODULE = "bsops_the_final.deployment.promote"


class FakeModelVersion:
    """Mimics MLflow's ModelVersion: dict(mv) yields its fields."""

    def __init__(self, current_stage, version="1", run_id=None):
        self.current_stage = current_stage
        self.version = version
        self.run_id = run_id

    def __iter__(self):
        return iter(vars(self).items())


def model_version(stage):
    return FakeModelVersion(stage)


@patch(f"{MODULE}.time.sleep")
class TestWaitForDeployment(unittest.TestCase):

    @patch(f"{MODULE}.client")
    def test_returns_true_when_stage_reached(self, mock_client, mock_sleep):
        mock_client.get_model_version.side_effect = [model_version("None"), model_version("Staging")]

        self.assertTrue(promote.wait_for_deployment("m", "1", "Staging"))
        self.assertEqual(mock_client.get_model_version.call_count, 2)

    @patch(f"{MODULE}.client")
    def test_returns_false_after_max_attempts(self, mock_client, mock_sleep):
        mock_client.get_model_version.return_value = model_version("None")

        self.assertFalse(promote.wait_for_deployment("m", "1", "Production", max_attempts=3))
        self.assertEqual(mock_client.get_model_version.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 3)


@patch(f"{MODULE}.wait_for_deployment", return_value=True)
class TestPromoteModel(unittest.TestCase):

    @patch(f"{MODULE}.mlflow")
    @patch(f"{MODULE}.client")
    def test_promotes_existing_version_to_production(self, mock_client, mock_mlflow, mock_wait):
        mock_client.search_model_versions.return_value = [
            SimpleNamespace(run_id="other", version="1"),
            SimpleNamespace(run_id="run-a", version="2"),
        ]
        mock_client.get_model_version.return_value = model_version("None")

        version = promote.promote_model("run-a")

        self.assertEqual(version, "2")
        mock_mlflow.register_model.assert_not_called()
        stages = [c.kwargs["stage"] for c in mock_client.transition_model_version_stage.call_args_list]
        self.assertEqual(stages, ["Staging", "Production"])
        mock_client.set_registered_model_alias.assert_called_once_with(
            name=promote.model_name, alias="Production", version="2"
        )
        mock_client.set_model_version_tag.assert_called_once_with(
            promote.model_name, "2", "deployed_to", "production"
        )

    @patch(f"{MODULE}.mlflow")
    @patch(f"{MODULE}.client")
    def test_registers_run_if_not_registered(self, mock_client, mock_mlflow, mock_wait):
        mock_client.search_model_versions.return_value = []
        mock_mlflow.register_model.return_value = SimpleNamespace(version="5")
        mock_client.get_model_version.return_value = model_version("None")

        version = promote.promote_model("run-b")

        self.assertEqual(version, "5")
        mock_mlflow.register_model.assert_called_once_with(
            f"runs:/run-b/{promote.artifact_path}", name=promote.model_name
        )

    @patch(f"{MODULE}.mlflow")
    @patch(f"{MODULE}.client")
    def test_skips_staging_transition_if_already_staged(self, mock_client, mock_mlflow, mock_wait):
        mock_client.search_model_versions.return_value = [SimpleNamespace(run_id="run-a", version="2")]
        mock_client.get_model_version.return_value = model_version("Staging")

        promote.promote_model("run-a")

        mock_client.transition_model_version_stage.assert_called_once()
        self.assertEqual(
            mock_client.transition_model_version_stage.call_args.kwargs["stage"], "Production"
        )
        mock_wait.assert_has_calls([call(promote.model_name, "2", "Production")])


if __name__ == "__main__":
    unittest.main()
