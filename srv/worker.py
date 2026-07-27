"""Background worker: processes the queue strictly one job at a time, using
the model that was already loaded once at daemon startup (see app.py — the
model is passed in here, not reloaded between jobs).

Parallel job processing is intentionally not implemented: on a single
CPU-bound machine it buys no throughput (jobs would just split the same
cores), while doubling memory usage for the loaded model.
"""
import io
import time
import traceback

from conf import config
from db import db


def run_worker(stable_diffusion) -> None:
    """Infinite queue-processing loop."""
    while True:
        job = db.fetch_next_queued()
        if job is None:
            time.sleep(config.WORKER_POLL_INTERVAL_SEC)
            continue

        db.mark_processing(job["id"])
        print(f"[job {job['id']}] starting generation ({job['mode']}): {job['prompt'][:60]!r}")

        try:
            image_bytes = _generate(stable_diffusion, job)
            db.mark_done(job["id"], image_bytes)
            print(f"[job {job['id']}] done")
        except Exception as e:
            db.mark_error(job["id"], f"{e}\n{traceback.format_exc()}")
            print(f"[job {job['id']}] error: {e}")

        db.purge_expired(config.JOB_TTL_HOURS)


def _generate(stable_diffusion, job) -> bytes:
    from PIL import Image  # local import: only needed here

    kwargs = dict(
        prompt=job["prompt"],
        width=job["width"],
        height=job["height"],
        sample_steps=job["steps"],
        cfg_scale=job["cfg_scale"],
    )
    if job["negative_prompt"]:
        kwargs["negative_prompt"] = job["negative_prompt"]
    if job["seed"] is not None:
        kwargs["seed"] = job["seed"]

    if job["mode"] in ("img2img", "inpaint") and job["init_image"]:
        kwargs["init_image"] = Image.open(io.BytesIO(job["init_image"]))
        if job["strength"] is not None:
            kwargs["strength"] = job["strength"]
    if job["mode"] == "inpaint" and job["mask_image"]:
        kwargs["mask_image"] = Image.open(io.BytesIO(job["mask_image"]))

    output = stable_diffusion.generate_image(**kwargs)
    image = output[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
