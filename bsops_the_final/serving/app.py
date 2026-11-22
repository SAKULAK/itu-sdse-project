"""FastAPI wrapper in front of the MLflow scoring server.

- GET  /health      -> {"status": "ok"} once the scoring server is up, "loading" (503)
                       while it loads, "error" (503) if it died
- POST /invocations -> forwarded unchanged to the scoring server
- POST /infer       -> alias of /invocations
- GET  /version     -> the model version and training timestamp of the served model
- GET  /drift       -> drift detection (KS tests) over the requests logged by the scoring server

In production this app runs as the container's main process; the scoring server stays
a subprocess of it.

Run from the project root:  python -m bsops_the_final.serving
"""

import json
from pathlib import Path
import urllib.error
import urllib.request

# FastAPI/uvicorn are installed as MLflow dependencies (MLflow's own scoring
# server uses them)
from fastapi import FastAPI, Query, Request, Response
import uvicorn

from ..cfg import (
    drift_default_days,
    inference_log_path,
    model_path,
    port,
    upstream_host,
    upstream_port,
)
from .drift import detect_drift, load_baseline_stats, load_inference_logs
from .scoring_server import _start_scoring_server

# Set by main() once the scoring server subprocess has been started
scoring_proc = None

app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)


def _upstream_health():
    """Return the scoring server's /health status code, None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{upstream_host}:{upstream_port}/health",
            timeout=1.0,
        ) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


@app.get("/health")
def health():
    if scoring_proc is not None and scoring_proc.poll() is not None:
        return Response(content='{"status": "error"}', status_code=503, media_type="application/json")
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(content='{"status": "loading"}', status_code=503, media_type="application/json")


@app.get("/version")
def version():
    """Model version of the served model (watched by the monitoring platform)."""
    version_file = Path(model_path) / "version.json"
    if version_file.exists():
        with open(version_file) as f:
            return json.load(f)
    return {"model_version": "unknown", "trained_at": None}


@app.post("/invocations")
@app.post("/infer")
async def invocations(request: Request):
    body = await request.body()
    req = urllib.request.Request(
        f"http://{upstream_host}:{upstream_port}/invocations",
        data=body,
        headers={"Content-Type": request.headers.get("content-type", "application/json")},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return Response(
                content=resp.read(),
                status_code=resp.status,
                media_type=resp.headers.get("content-type", "application/json"),
            )
    except urllib.error.HTTPError as e:
        return Response(
            content=e.read(),
            status_code=e.code,
            media_type=e.headers.get("content-type", "application/json"),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Response(
            content=f'{{"status": "error", "detail": "{e}"}}',
            status_code=502,
            media_type="application/json",
        )


@app.get("/drift")
def drift(days: int = Query(default=drift_default_days, ge=1)):
    """Run drift detection over the requests logged by the scoring server."""
    logs = load_inference_logs(inference_log_path)
    baseline = load_baseline_stats()
    return detect_drift(logs, days=days, baseline_stats=baseline)


def main():
    global scoring_proc
    scoring_proc = _start_scoring_server()
    print("Scoring server starting on port", upstream_port)

    api_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    api_server = uvicorn.Server(api_config)
    print(f"API server starting on http://localhost:{port}")
    try:
        api_server.run()
    finally:
        if scoring_proc is not None and scoring_proc.poll() is None:
            scoring_proc.terminate()
            scoring_proc.wait(timeout=10)
            print("Scoring server stopped")
        print("API server stopped")


if __name__ == "__main__":
    main()
