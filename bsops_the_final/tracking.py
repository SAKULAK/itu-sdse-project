import os

import mlflow
from mlflow.tracking import MlflowClient

from .cfg import mlflow_artifact_location, mlflow_state_dir, mlflow_tracking_uri


def setup_mlflow_tracking(experiment_name=None):
    os.makedirs(mlflow_state_dir, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if experiment_name is None:
        return
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location)
    mlflow.set_experiment(experiment_name)


# Shared registry client. The tracking URI is passed explicitly so the client does not
# depend on setup_mlflow_tracking() having run before this module was imported.
client = MlflowClient(tracking_uri=mlflow_tracking_uri)
