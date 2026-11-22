"""Promote the selected model version through Staging to Production."""

import time

import mlflow

from ..cfg import artifact_path, model_name
from ..tracking import client


def wait_for_deployment(model_name, version, stage='Staging', max_attempts=20):
    status = False
    for attempt in range(max_attempts):
        model_version_details = dict(client.get_model_version(name=model_name, version=version))
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            print(f"Waiting for transition to {stage} (current: {model_version_details['current_stage']})")
            time.sleep(2)
    return status


def promote_model(run_id):
    """Register (if needed) and promote the model of `run_id`. Returns its version."""
    # Find the registered version that belongs to the winning run
    versions = [mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id]
    if not versions:
        print("Run was not registered yet, registering it now")
        model_details = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", name=model_name)
        version = model_details.version
    else:
        version = versions[0].version

    print(f"Promoting {model_name} version {version}")

    # Always via Staging
    model_version_details = dict(client.get_model_version(name=model_name, version=version))
    if model_version_details['current_stage'] != 'Staging':
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
        wait_for_deployment(model_name, version, 'Staging')

    # And to Production (we only get here if the comparison above decided we should)
    if run_id is not None:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        wait_for_deployment(model_name, version, 'Production')
        client.set_registered_model_alias(name=model_name, alias="Production", version=str(version))
        client.set_model_version_tag(model_name, version, "deployed_to", "production")
        print(f"{model_name} version {version} is now in Production")

    return version
