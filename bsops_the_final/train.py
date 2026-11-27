import json
import os

import torch
import numpy as np
import mlflow

from bsops_the_final.model import SimpleCNN, train_cnn_model, evaluate_cnn_model, compute_baseline_statistics, tune_threshold, SimpleLoggingWrapper
from bsops_the_final.cfg import  (
    image_data_dir,
    metadata_path,
    num_classes,
    experiment_name,
    learning_rate,
    num_epochs, 
    batch_size, 
    random_state,
    threshold, 
    drift_baseline_stats_path, 
    pytorch_model_path, 
    model_results_path, 
    artifact_path,
    artifact_dir,
    mlflow_artifact_location, 
    mlflow_state_dir, 
    mlflow_tracking_uri,
    inference_log_path,
    image_size,
    transform_mean,
    transform_std
)
from bsops_the_final.data import split_dataset, create_dataloaders, load_image_dataset
from bsops_the_final.tracking import setup_mlflow_tracking
    
os.makedirs(artifact_dir, exist_ok=True)
os.makedirs(mlflow_state_dir, exist_ok=True)
os.makedirs(mlflow_artifact_location, exist_ok=True)
os.makedirs(image_data_dir, exist_ok=True)

image_dataset = load_image_dataset(image_data_dir, metadata_path)
train_dataset, val_dataset, test_dataset = split_dataset(image_dataset)
train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
setup_mlflow_tracking(mlflow_artifact_location, mlflow_state_dir, mlflow_tracking_uri, experiment_name=experiment_name)

torch.manual_seed(random_state)
np.random.seed(random_state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = mlflow.start_run(run_name="training_notebook")

# Log parameters to MLflow
mlflow.log_param("model_architecture", "SimpleCNN")
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("num_epochs", num_epochs)
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("initial_threshold", threshold)
mlflow.log_param("num_classes", num_classes)
mlflow.log_param("image_size", image_size)
mlflow.log_param("random_seed", random_state)

# Log model architecture details
model_params = sum(p.numel() for p in SimpleCNN().parameters())
trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
mlflow.log_param("total_parameters", model_params)
mlflow.log_param("trainable_parameters", trainable_params)

cnn_model, training_history = train_cnn_model(
    train_loader, val_loader, num_epochs=num_epochs, 
    learning_rate=learning_rate, threshold=threshold, 
    device=device, random_seed=random_state, verbose=True
)

# Tune threshold on validation set
optimal_threshold = tune_threshold(cnn_model, val_loader, device=device)
mlflow.log_param("optimal_threshold", optimal_threshold)

# Evaluate on test set with optimal threshold
test_results = evaluate_cnn_model(cnn_model, test_loader, device=device, threshold=optimal_threshold)

baseline_stats = compute_baseline_statistics(cnn_model, val_loader, threshold=optimal_threshold)

with open(drift_baseline_stats_path, 'w') as f:
    json.dump(baseline_stats, f, indent=2)

# Log baseline statistics to MLflow
mlflow.log_metric("baseline_image_mean_r", baseline_stats['image_statistics']['mean_r'])
mlflow.log_metric("baseline_image_mean_g", baseline_stats['image_statistics']['mean_g'])
mlflow.log_metric("baseline_image_mean_b", baseline_stats['image_statistics']['mean_b'])
mlflow.log_metric("baseline_image_std_r", baseline_stats['image_statistics']['std_r'])
mlflow.log_metric("baseline_image_std_g", baseline_stats['image_statistics']['std_g'])
mlflow.log_metric("baseline_image_std_b", baseline_stats['image_statistics']['std_b'])
mlflow.log_metric("baseline_class_0_count", baseline_stats['class_distribution']['class_0_count'])
mlflow.log_metric("baseline_class_1_count", baseline_stats['class_distribution']['class_1_count'])
mlflow.log_metric("baseline_confidence_mean", baseline_stats['confidence_statistics']['mean'])
mlflow.log_metric("baseline_confidence_std", baseline_stats['confidence_statistics']['std'])
mlflow.log_artifact(drift_baseline_stats_path, "drift_detection")


# Save model state dict
torch.save({
    'model_state_dict': cnn_model.state_dict(),
    'architecture': SimpleCNN.__name__,
    'input_shape': (3, image_size[0], image_size[1]),
    'output_shape': (1,),
    'threshold': optimal_threshold,
    'num_classes': num_classes,
}, pytorch_model_path)

# Save model results
with open(model_results_path, "w") as results_file:
    json.dump(test_results, results_file, indent=2)

# Log the pyfunc model with the SimpleLoggingWrapper as the model artifact
wrapper = SimpleLoggingWrapper(log_path = inference_log_path, image_size = image_size, transform_mean = transform_mean, transform_std = transform_std, threshold = optimal_threshold, model=cnn_model)
mlflow.pyfunc.log_model(artifact_path, python_model=wrapper)

mlflow.end_run()
