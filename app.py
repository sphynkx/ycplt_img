"""
ycplt_img entry point.

Run:
    python app.py

Models are loaded through conf/models.py's factory: the txt2img/img2img
model is loaded eagerly at startup (same "fail fast if misconfigured"
guarantee as before) and stays resident for the daemon's lifetime; other
slots (e.g. a dedicated inpainting model, if INPAINT_MODEL is set) load
lazily on first use instead. No job ever triggers a reload of a model
that's already loaded. Processing runs strictly sequentially in a
background thread (srv/worker.py). The HTTP layer (srv/server.py) is
standard-library only, no FastAPI/Flask.

This file only wires things together: init the DB, preload the default
model, start the worker thread, start the HTTP server. All actual logic
lives in conf/, db/, srv/.
"""
import threading

from conf import config, models
from db import db
from srv import server, worker


def main() -> None:
    db.init_db()
    reset = db.reset_stuck_processing()
    if reset:
        print(f"Reset {reset} job(s) stuck in 'processing' from a previous run.")

    models.preload_default()

    worker_thread = threading.Thread(target=worker.run_worker, daemon=True)
    worker_thread.start()

    server.run_server(config.HOST, config.PORT)


if __name__ == "__main__":
    main()
