"""
ycplt_img entry point.

Run:
    python app.py

The model is loaded once at startup and stays resident in memory for the
whole lifetime of the daemon — jobs never trigger a reload. Processing runs
strictly sequentially in a background thread (srv/worker.py). The HTTP layer
(srv/server.py) is standard-library only, no FastAPI/Flask.

This file only wires things together: init the DB, load the model, start the
worker thread, start the HTTP server. All actual logic lives in conf/, db/,
srv/.
"""
import threading

from conf import config
from db import db
from srv import server, worker
from stable_diffusion_cpp import StableDiffusion


def main() -> None:
    db.init_db()
    reset = db.reset_stuck_processing()
    if reset:
        print(f"Reset {reset} job(s) stuck in 'processing' from a previous run.")

    print(f"Loading model: {config.MODEL_PATH} (wtype={config.WTYPE}) — this may take a while...")
    stable_diffusion = StableDiffusion(
        model_path=config.MODEL_PATH,
        wtype=config.WTYPE,
    )
    print("Model loaded, staying resident in memory.")

    worker_thread = threading.Thread(
        target=worker.run_worker, args=(stable_diffusion,), daemon=True
    )
    worker_thread.start()

    server.run_server(config.HOST, config.PORT)


if __name__ == "__main__":
    main()
