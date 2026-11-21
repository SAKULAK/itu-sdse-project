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
