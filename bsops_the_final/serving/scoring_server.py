"""The MLflow scoring server (the pyfunc model), run as a subprocess on an internal port."""

import atexit
import os
import subprocess
import sys

from ..cfg import model_path, upstream_host, upstream_port


def _start_scoring_server():
    env = os.environ.copy()
    # The scoring server execs `uvicorn` through PATH; make sure the
    # interpreter's bin directory is on it
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen([
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_path,
        "-p", str(upstream_port),
        "--workers", "1",
        "--host", upstream_host,
        "--no-conda",
    ], env=env)
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc
