"""Export the serving model.

The serving image needs the model as a plain directory (no MLflow server involved at
serving time), so we download the winning pyfunc model to `models/model`. This is the
directory that later gets baked into the Docker image.

We also write the model version and its training timestamp into the export, so the
`/version` endpoint of the service can report them.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import mlflow

from ..cfg import artifact_path, experiment_name, export_dir, model_name
from ..tracking import client


def resolve_model_uri():
    """Resolve which model to serve: Production alias, else the newest
    registered version, else the latest run's artifact."""
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        source = getattr(mv, "source", None) or mv.storage_location
        return source, f"{model_name} version {mv.version} (Production alias)"
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            mv = max(versions, key=lambda v: int(v.version))
            source = getattr(mv, "source", None) or mv.storage_location
            return source, f"{model_name} version {mv.version} (newest registered; no Production alias)"
    except Exception:
        pass

    runs = mlflow.search_runs(experiment_names=[experiment_name], max_results=1, order_by=["start_time DESC"])
    if runs.empty:
        raise RuntimeError(f"No registered {model_name} model and no runs in experiment {experiment_name} - nothing to export")
    return f"runs:/{runs.iloc[0].run_id}/{artifact_path}", "latest run artifact (nothing registered)"


def export_model(version):
    """Download the model to `export_dir` and write version.json next to it.

    `version` is the fallback version reported if there is no Production alias.
    """
    model_uri, description = resolve_model_uri()
    print(f"Exporting: {description}")
    print(f"  Source: {model_uri}")

    Path(export_dir).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(export_dir).parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        if Path(export_dir).exists():
            shutil.rmtree(export_dir)
        shutil.move(str(download_dir), export_dir)
    print(f"  Exported to: {Path(export_dir).resolve()}")

    # Bake the version info into the export for the /version endpoint
    try:
        prod_mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        trained_at = datetime.fromtimestamp(client.get_run(prod_mv.run_id).info.start_time / 1000, tz=timezone.utc).isoformat()
        version_info = {"model_version": str(prod_mv.version), "trained_at": trained_at}
    except Exception:
        version_info = {"model_version": str(version), "trained_at": None}

    with open(Path(export_dir) / "version.json", "w") as f:
        json.dump(version_info, f, indent=2)
    print("Version info:", version_info)
    return version_info
