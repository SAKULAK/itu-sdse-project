"""Central configuration for the vial image classification project.

Only values and a small helper live here, no pipeline logic. The object-store bucket is public, so no .env file is needed.
All paths are relative to the project root: run everything from there.
"""
from pathlib import Path

import torch
import os
import mlflow

max_date = "2024-01-31"
min_date = "2024-01-01"

# Training configuration
num_epochs = 5
batch_size = 32
learning_rate = 0.0003
threshold = 0.5
early_stop_patience = 2
scheduler_patience = 3
random_state = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image specifications
image_size = (64, 64)
original_image_size = (1600, 768)
num_classes = 2
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]

# Paths (relative to the project root)
artifact_dir = "artifacts"
mlflow_state_dir = "mlflow-state"
image_data_dir = "./data/images/raw"
metadata_path = "./data/images/metadata.csv"
test_data_dir = "./data/test"
pytorch_model_path = artifact_dir + "/vial_cnn_model.pth"
model_results_path = artifact_dir + "/model_results.json"
drift_baseline_stats_path = artifact_dir + "/drift_baseline_stats.json"
class_names_path = artifact_dir + "/class_names.json"
image_statistics_path = artifact_dir + "/image_statistics.json"
inference_log_path = artifact_dir + "/inference_log.json"
test_metadata_path = "./data/test/metadata.csv"

# MLflow - we log to a local sqlite db so no tracking server has to be running
experiment_name = "vial_image_classification"
model_name = "cnn_model"
mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"
mlflow_artifact_location = "./mlflow-state/artifacts"
test_metadata_path = "./data/test/metadata.csv"

# MLflow - we log to a local sqlite db so no tracking server has to be running

# MLflow logs to a local sqlite db, so no tracking server has to be running
experiment_name = "vial_image_classification"
model_name = "cnn_model"
mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"
mlflow_artifact_location = "./mlflow-state/artifacts"

# Deployment
artifact_path = "model_pyfunc"
model_version = 1  # fallback, the real version is looked up from the registry

drift_default_days = 7
drift_ks_pvalue_threshold = 0.05

# Where the exported servable model goes (this is what gets baked into the Docker image)
export_dir = "models/model"
model_path = export_dir

# Serving ports
port = 5001
upstream_port = 5002
upstream_host = "127.0.0.1"

# Where the endpoint lives in production
PROD_ENDPOINT = "https://mlops-itu-jtk.ondigitalocean.app"

# S3 Configuration (Hetzner Object Storage)
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"
# Jupyter starts us in notebooks/ when the notebook is opened from there -
# everything below assumes we are running from the project root
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlflow-state", exist_ok=True)
os.makedirs("mlflow-state/artifacts", exist_ok=True)
os.makedirs("data/images/raw", exist_ok=True)
print("Created artifacts directory")


def setup_mlflow_tracking(experiment_name=None):
    """Point MLflow at the local state dir and create the experiment if needed."""
    os.makedirs(mlflow_state_dir, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location)
    mlflow.set_experiment(experiment_name)

if __name__ == "__main__":
    setup_mlflow_tracking(experiment_name)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
