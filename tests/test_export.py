import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from bsops_the_final.deployment import export

MODULE = "bsops_the_final.deployment.export"


class TestResolveModelUri(unittest.TestCase):

    @patch(f"{MODULE}.client")
    def test_prefers_production_alias(self, mock_client):
        mock_client.get_model_version_by_alias.return_value = SimpleNamespace(
            version="3", source="s3://prod-model"
        )

        uri, description = export.resolve_model_uri()

        self.assertEqual(uri, "s3://prod-model")
        self.assertIn("Production alias", description)

    @patch(f"{MODULE}.client")
    def test_falls_back_to_newest_registered_version(self, mock_client):
        mock_client.get_model_version_by_alias.side_effect = Exception("no alias")
        mock_client.search_model_versions.return_value = [
            SimpleNamespace(version="2", source="src-2"),
            SimpleNamespace(version="10", source="src-10"),
            SimpleNamespace(version="9", source="src-9"),
        ]

        uri, description = export.resolve_model_uri()

        # Compared as ints, so "10" beats "9"
        self.assertEqual(uri, "src-10")
        self.assertIn("version 10", description)

    @patch(f"{MODULE}.mlflow")
    @patch(f"{MODULE}.client")
    def test_falls_back_to_latest_run(self, mock_client, mock_mlflow):
        mock_client.get_model_version_by_alias.side_effect = Exception("no alias")
        mock_client.search_model_versions.return_value = []
        mock_mlflow.search_runs.return_value = pd.DataFrame([{"run_id": "abc"}])

        uri, _ = export.resolve_model_uri()

        self.assertEqual(uri, f"runs:/abc/{export.artifact_path}")

    @patch(f"{MODULE}.mlflow")
    @patch(f"{MODULE}.client")
    def test_raises_when_nothing_to_export(self, mock_client, mock_mlflow):
        mock_client.get_model_version_by_alias.side_effect = Exception("no alias")
        mock_client.search_model_versions.return_value = []
        mock_mlflow.search_runs.return_value = pd.DataFrame()

        with self.assertRaises(RuntimeError):
            export.resolve_model_uri()


def fake_download(artifact_uri, dst_path):
    """Pretend to download a model by writing an MLmodel file into dst_path."""
    model_dir = Path(dst_path) / "model_pyfunc"
    model_dir.mkdir()
    (model_dir / "MLmodel").write_text("flavors: {}")
    return str(model_dir)


class TestExportModel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.export_dir = str(Path(self.tmp.name) / "models" / "model")
        patcher = patch(f"{MODULE}.export_dir", self.export_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    @patch(f"{MODULE}.mlflow")
    @patch(f"{MODULE}.resolve_model_uri", return_value=("runs:/abc/model_pyfunc", "test"))
    @patch(f"{MODULE}.client")
    def test_exports_model_and_writes_production_version(self, mock_client, _resolve, mock_mlflow):
        mock_mlflow.artifacts.download_artifacts.side_effect = fake_download
        mock_client.get_model_version_by_alias.return_value = SimpleNamespace(version="4", run_id="r4")
        # 2024-01-01T00:00:00Z in milliseconds
        mock_client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(start_time=1704067200000))

        info = export.export_model(version=1)

        self.assertEqual(info, {"model_version": "4", "trained_at": "2024-01-01T00:00:00+00:00"})
        self.assertTrue((Path(self.export_dir) / "MLmodel").exists())
        with open(Path(self.export_dir) / "version.json") as f:
            self.assertEqual(json.load(f), info)

    @patch(f"{MODULE}.mlflow")
    @patch(f"{MODULE}.resolve_model_uri", return_value=("runs:/abc/model_pyfunc", "test"))
    @patch(f"{MODULE}.client")
    def test_uses_fallback_version_without_production_alias(self, mock_client, _resolve, mock_mlflow):
        mock_mlflow.artifacts.download_artifacts.side_effect = fake_download
        mock_client.get_model_version_by_alias.side_effect = Exception("no alias")

        info = export.export_model(version=7)

        self.assertEqual(info, {"model_version": "7", "trained_at": None})

    @patch(f"{MODULE}.mlflow")
    @patch(f"{MODULE}.resolve_model_uri", return_value=("runs:/abc/model_pyfunc", "test"))
    @patch(f"{MODULE}.client")
    def test_replaces_previous_export(self, mock_client, _resolve, mock_mlflow):
        mock_mlflow.artifacts.download_artifacts.side_effect = fake_download
        mock_client.get_model_version_by_alias.side_effect = Exception("no alias")
        Path(self.export_dir).mkdir(parents=True)
        (Path(self.export_dir) / "stale.txt").write_text("old")

        export.export_model(version=1)

        self.assertFalse((Path(self.export_dir) / "stale.txt").exists())
        self.assertTrue((Path(self.export_dir) / "MLmodel").exists())


if __name__ == "__main__":
    unittest.main()
