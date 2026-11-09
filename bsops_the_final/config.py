"""Central configuration for the vial image classification project.

Only values and a small helper live here, no pipeline logic. The object-store bucket is public, so no .env file is needed.
All paths are relative to the project root: run everything from there.
"""
from pathlib import Path

import torch

# Training configuration
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 2
SCHEDULER_PATIENCE = 3
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image specifications
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 2
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "good", 1: "defective"}

# Paths (relative to the project root)
ARTIFACT_DIR = "artifacts"
MLFLOW_STATE_DIR = "mlflow-state"
IMAGE_DATA_DIR = "./data/images/raw"
METADATA_PATH = "./data/images/metadata.csv"
TEST_DATA_DIR = "./data/test"
TEST_METADATA_PATH = "./data/test/metadata.csv"

PYTORCH_MODEL_PATH = ARTIFACT_DIR + "/vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACT_DIR + "/model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACT_DIR + "/drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACT_DIR + "/class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACT_DIR + "/image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACT_DIR + "/inference_log.json"

# Object store (public bucket, unsigned access)
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"
S3_TRAIN_PREFIX = "images/raw/"
S3_TEST_PREFIX = "test/"

# MLflow logs to a local sqlite db, so no tracking server has to be running
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_STATE_DIR}/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = "./mlflow-state/artifacts"


def ensure_dirs():
    """Create the directories the pipeline writes to."""
    for directory in (
        ARTIFACT_DIR,
        MLFLOW_STATE_DIR,
        MLFLOW_ARTIFACT_LOCATION,
        IMAGE_DATA_DIR,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)