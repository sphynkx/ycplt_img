"""ycplt_img configuration.

Settings come from, in priority order (highest first):

  1. A real process environment variable (systemd Environment=/EnvironmentFile=,
     or exported in the shell).
  2. A ".env" file in the project root, loaded via python-dotenv (see
     install/.env.example for the template).
  3. The hardcoded default below.

python-dotenv's load_dotenv() does not override variables already present in
os.environ, which is exactly the priority order above — no custom logic needed.

Note: when running under systemd with EnvironmentFile=.env in the unit,
systemd itself loads .env straight into the process environment, so priority
(1) already covers it there; load_dotenv() below is what makes the same .env
file work for a plain `python app.py` run too.
"""
import os

from dotenv import load_dotenv

load_dotenv()

HOST = os.environ.get("YCPLT_HOST", "0.0.0.0")
PORT = int(os.environ.get("YCPLT_PORT", "4011"))

# Model checkpoint. MODEL is just the filename (e.g. "sd-v1-4.ckpt"), looked
# up inside MODELS_DIR — this is the setting meant to change per deployment.
# YCPLT_MODEL_PATH is an optional escape hatch for a full path if the
# checkpoint doesn't live under MODELS_DIR.
MODELS_DIR = os.environ.get("YCPLT_MODELS_DIR", "models")
MODEL = os.environ.get("MODEL", "sd-v1-4.ckpt")
_explicit_model_path = os.environ.get("YCPLT_MODEL_PATH", "")
MODEL_PATH = _explicit_model_path if _explicit_model_path else os.path.join(MODELS_DIR, MODEL)

# Weight quantization type used on load (see GGML_TYPE_MAP in
# stable_diffusion_cpp, e.g. "q4_0", "q8_0", "f16", "default").
WTYPE = os.environ.get("YCPLT_WTYPE", "q4_0")

# Job queue
DB_PATH = os.environ.get("YCPLT_DB_PATH", "data/jobs.sqlite3")

# Defaults for jobs that don't specify these fields
DEFAULT_WIDTH = int(os.environ.get("YCPLT_DEFAULT_WIDTH", "512"))
DEFAULT_HEIGHT = int(os.environ.get("YCPLT_DEFAULT_HEIGHT", "512"))
DEFAULT_STEPS = int(os.environ.get("YCPLT_DEFAULT_STEPS", "20"))
DEFAULT_CFG_SCALE = float(os.environ.get("YCPLT_DEFAULT_CFG_SCALE", "7.5"))

# How often (seconds) the worker checks the queue for new jobs (internal loop —
# not to be confused with how often the main chat app polls this service).
WORKER_POLL_INTERVAL_SEC = float(os.environ.get("YCPLT_WORKER_POLL_INTERVAL_SEC", "1.0"))

# How many hours to keep finished (done/error) but unclaimed jobs before purging them.
JOB_TTL_HOURS = float(os.environ.get("YCPLT_JOB_TTL_HOURS", "24"))
