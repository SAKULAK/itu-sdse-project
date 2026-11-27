import os

import mlflow
from mlflow.tracking import MlflowClient

from .cfg import mlflow_artifact_location, mlflow_state_dir, mlflow_tracking_uri, experiment_name

def setup_mlflow_tracking(mlflow_artifact_location=mlflow_artifact_location, 
                          mlflow_state_dir=mlflow_state_dir, 
                          mlflow_tracking_uri=mlflow_tracking_uri, 
                          experiment_name=experiment_name):
    os.makedirs(mlflow_state_dir, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if experiment_name is None:
        return
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location)
    mlflow.set_experiment(experiment_name)

client = MlflowClient(tracking_uri=mlflow_tracking_uri)