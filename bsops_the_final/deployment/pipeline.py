"""Deployment pipeline: select the best model, promote it if it beats production,
and export it to `models/model` for the serving image.

Run from the project root:  python -m bsops_the_final.deployment
"""

import json
import os
from pprint import pprint
import warnings

import mlflow

from ..cfg import artifact_dir, experiment_name, mlflow_state_dir, model_results_path
from ..tracking import setup_mlflow_tracking
from .export import export_model
from .promote import promote_model
from .selection import compare_with_production, get_best_run, get_production_model


def main():
    warnings.filterwarnings('ignore')

    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(mlflow_state_dir, exist_ok=True)

    setup_mlflow_tracking(experiment_name)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # MODEL SELECTION
    experiment_best = get_best_run(experiment_name)

    with open(model_results_path, "r") as f:
        model_results = json.load(f)
    pprint(model_results["classification_report"])

    # Get production model
    production_model_uri, prod_model_version, prod_model_run_id = get_production_model()

    # Compare prod and best trained model
    run_id = compare_with_production(experiment_best, production_model_uri, prod_model_run_id)

    # Promote best model
    version = promote_model(run_id)

    # Export the serving model
    export_model(version)


if __name__ == "__main__":
    main()
