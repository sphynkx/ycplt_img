"""Background worker: processes the queue strictly one job at a time. Which
model instance handles a given job is decided per-job by conf/models.py's
factory (get_model_for_mode) rather than a single fixed instance — see that
module for how modes map to checkpoints and why instances are cached rather
than reloaded.

Parallel job processing is intentionally not implemented: on a single
CPU-bound machine it buys no throughput (jobs would just split the same
cores), while doubling memory usage for whichever model(s) are loaded.
"""
import io
import time
import traceback

from conf import config, models
from db import db


def run_worker() -> None:
    """Infinite queue-processing loop."""
    while True:
        job = db.fetch_next_queued()
        if job is None:
            time.sleep(config.WORKER_POLL_INTERVAL_SEC)
            continue

        db.mark_processing(job["id"])
        print(f"[job {job['id']}] starting ({job['mode']}): {job['prompt'][:60]!r}")

        try:
            if job["mode"] == "caption":
                text = _caption(job)
                db.mark_done_text(job["id"], text)
            elif job["mode"] == "img2img" and job.get("remove_target"):
                image_bytes = _generate_removal_edit(job)
                db.mark_done(job["id"], image_bytes)
            else:
                image_bytes = _generate(job)
                db.mark_done(job["id"], image_bytes)
            print(f"[job {job['id']}] done")
        except Exception as e:
            db.mark_error(job["id"], f"{e}\n{traceback.format_exc()}")
            print(f"[job {job['id']}] error: {e}")

        db.purge_expired(config.JOB_TTL_HOURS)


def _caption(job) -> str:
    """mode="caption": answers a question about (or describes) job['init_image']
    using the vision model (conf/models.get_vision_model()) instead of
    generating pixels. job['prompt'] carries the user's question, reusing
    the same column generation jobs use for their prompt — no schema
    change needed beyond result_text (see db/db.py)."""
    import base64

    llm = models.get_vision_model()
    if llm is None:
        status = models.vision_status()
        raise RuntimeError(
            "vision model unavailable: "
            + (status["load_error"] or "not configured — see conf/config.VISION_MODEL_PATH")
        )
    if not job["init_image"]:
        raise RuntimeError("mode='caption' requires init_image")

    # The data URI's declared mime type is effectively decorative here —
    # the handler decodes the base64 bytes and detects the actual image
    # format from its contents, not from this label — so a fixed "image/png"
    # is fine regardless of what the original upload's real format was.
    data_uri = "data:image/png;base64," + base64.b64encode(job["init_image"]).decode("ascii")
    question = job["prompt"] or "Describe this image."

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    return response["choices"][0]["message"]["content"]


def _generate_removal_edit(job) -> bytes:
    """mode="img2img" with remove_target set (see routes/chat.py and
    utils/intent.get_removal_target_async on the main app side): plain
    img2img has no mechanism to execute a "remove X" instruction — it just
    partially re-renders the whole image guided by the prompt text, so the
    named object doesn't actually disappear, it just gets restyled.

    Instead: CLIPSeg (conf/segmentation.py) finds a mask for the named
    object from its English name alone — no manual mask needed — then the
    inpainting-tuned checkpoint (conf/models.py's "inpaint" slot) repaints
    just that region. A high strength is used deliberately: for removal we
    want the masked region fully replaced, not lightly retouched.
    """
    from PIL import Image
    from conf import segmentation

    if not job["init_image"]:
        raise RuntimeError("mode='img2img' with remove_target requires init_image")

    init_image = Image.open(io.BytesIO(job["init_image"]))
    mask = segmentation.get_mask(init_image, job["remove_target"])
    if mask is None:
        seg_status = segmentation.status()
        raise RuntimeError(
            "automatic object segmentation unavailable: "
            + (seg_status["load_error"] or "CLIPSeg not installed — see requirements.txt")
        )

    stable_diffusion = models.get_model_for_mode("inpaint")
    kwargs = dict(
        prompt=f"empty background, seamless, natural, no {job['remove_target']}",
        negative_prompt=job["remove_target"],
        init_image=init_image,
        mask_image=mask,
        strength=0.95,
        width=job["width"],
        height=job["height"],
        sample_steps=job["steps"],
        cfg_scale=job["cfg_scale"],
    )
    if job["seed"] is not None:
        kwargs["seed"] = job["seed"]

    output = stable_diffusion.generate_image(**kwargs)
    image = output[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _generate(job) -> bytes:
    from PIL import Image  # local import: only needed here

    stable_diffusion = models.get_model_for_mode(job["mode"])

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
