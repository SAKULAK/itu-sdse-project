"""Model selection: find the best trained run and compare it with production."""

import mlflow

from ..cfg import model_name
from ..tracking import client


def get_best_run(experiment_name):
    """The best run of the experiment is the one with the highest test F1-score."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_ids = [experiment.experiment_id]

    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]
    print("Best run:", experiment_best["run_id"])
    print("Test F1-score:", experiment_best.get("metrics.test_f1_score"))
    return experiment_best


def get_production_model():
    """Fetch the current production model, by alias or by stage.

    Returns (model_uri, version, run_id) or (None, None, None) if there is
    no model in production yet.
    """
    # First, try to get the model with the alias "Production"
    try:
        model = client.get_model_version_by_alias(name=model_name, alias="Production")
        if model:
            print('Production model name: ', model_name)
            print('Production model version:', model.version)
            print('Production model run id:', model.run_id)
            return f"models:/{model_name}@Production", model.version, model.run_id
    except Exception:
        pass

    # Fall back to checking stage="Production"
    for model in client.search_model_versions(f"name='{model_name}'"):
        if dict(model).get('current_stage') == 'Production':
            print('Production model name: ', model_name)
            print('Production model version:', model.version)
            print('Production model run id:', model.run_id)
            return f"models:/{model_name}/{model.version}", model.version, model.run_id

    print('No model in production')
    return None, None, None


def compare_with_production(experiment_best, production_model_uri, prod_model_run_id):
    """Compare prod and best trained model.

    Returns the run_id to promote, or None if the production model is better.
    """
    train_model_score = experiment_best.get("metrics.test_f1_score")
    if train_model_score is None:
        train_model_score = experiment_best.get("metrics.f1_score")

    model_status = {}
    run_id = None

    if production_model_uri is not None:
        prod_run = mlflow.get_run(prod_model_run_id)
        prod_model_score = prod_run.data.metrics.get("test_f1_score") or prod_run.data.metrics.get("f1_score")

        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score

        print("Comparing models:")
        print(f"  Current best (training): {train_model_score}")
        print(f"  Production: {prod_model_score}")

        if train_model_score > prod_model_score:
            print("New model is better. Promoting the new model.")
            run_id = experiment_best["run_id"]
        else:
            print("Current production model is better. Keeping it.")
    else:
        print("No model in production. The trained model will be promoted.")
        run_id = experiment_best["run_id"]

    print(f"Selected run: {run_id}")
    return run_id
