"""Smoke test against a running serving API (local by default).

Waits for /health, sends one test image to /infer as base64 (the portable contract -
the server can also take file paths, but those only work when the file is on the
server's machine), checks /version and /health, then runs the drift check both
locally and via the /drift endpoint.

Start the API first (python -m bsops_the_final.serving), then from the project root:
    python -m bsops_the_final.serving.smoke_test [BASE_URL]
"""

import base64
from pathlib import Path
from pprint import pprint
import sys
import time

import pandas as pd
import requests

from ..cfg import drift_default_days, inference_log_path, port
from .drift import detect_drift, load_baseline_stats, load_inference_logs


def wait_until_ready(base_url):
    # Wait for the scoring server to come up (loading torch takes a while)
    for attempt in range(120):
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            print(f"[{attempt}] /health -> {r.status_code} {r.text.strip()}")
            if r.status_code == 200:
                print("Model server is ready!")
                break
        except requests.RequestException:
            print(f"[{attempt}] waiting for the API server...")
        time.sleep(2)


def test_endpoint(base_url):
    test_metadata = pd.read_csv("data/test/metadata.csv")
    sample_file = test_metadata.iloc[0]["filename"]
    sample_label = int(test_metadata.iloc[0]["class"])
    print(f"Sample image: {sample_file} (label={sample_label})")

    with open(Path("data/test") / sample_file, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {"instances": [img_b64]}

    r = requests.post(f"{base_url}/infer", json=payload, timeout=60)
    print("POST /infer ->", r.status_code)
    pprint(r.json())

    print("GET /version ->", requests.get(f"{base_url}/version").json())
    print("GET /health  ->", requests.get(f"{base_url}/health").json())


def check_drift(base_url):
    logs = load_inference_logs(inference_log_path)
    print(f"Logged requests: {len(logs)}")
    pprint(logs[-3:])

    baseline = load_baseline_stats()
    drift_results = detect_drift(logs, days=drift_default_days, baseline_stats=baseline)
    print("\nDrift check results:")
    pprint(drift_results)

    if drift_results["overall_drift_detected"]:
        print("\n!! Drift detected - time to retrain and redeploy")
    else:
        print("\nNo drift detected")

    # The endpoint does the same check over HTTP
    print("\nGET /drift ->")
    pprint(requests.get(f"{base_url}/drift").json())


def main(base_url=f"http://localhost:{port}"):
    wait_until_ready(base_url)
    test_endpoint(base_url)
    check_drift(base_url)


if __name__ == "__main__":
    main(*sys.argv[1:2])
