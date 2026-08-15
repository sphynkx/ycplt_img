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


def _fit_gen_size(width: int, height: int, max_dimension: int, multiple: int = 64) -> tuple:
    """Picks the (width, height) to actually generate at for a masked
    reconstruct_prompt edit — aligned with init_image/mask (the fix for
    the striped/scrambled-output corruption bug), but capped at
    max_dimension on the longer side (the fix for a SEPARATE, later
    crash: generating at a real, uncapped photo resolution pushed
    stable-diffusion.cpp's UNet compute buffer past 10GB and killed the
    worker process outright — see config.RECONSTRUCT_MAX_DIMENSION's own
    comment for the full story).

    Scales (width, height) down — never up — to fit within max_dimension,
    preserving aspect ratio, then rounds to the nearest multiple of
    `multiple` (SD1.x needs at least a multiple of 8; 64 matches this
    project's existing 512x512-style defaults). The final rounding step
    is nudged down by one multiple if it would otherwise land just over
    max_dimension, so the cap is a real, honored ceiling rather than a
    rough target.
    """
    scale = min(1.0, max_dimension / max(width, height))
    target_w = width * scale
    target_h = height * scale

    def _round(d: float) -> int:
        r = max(multiple, int(round(d / multiple)) * multiple)
        if r > max_dimension:
            r = max(multiple, r - multiple)
        return r

    return _round(target_w), _round(target_h)


def _generate_removal_edit(job) -> bytes:
    """mode="img2img" with remove_target set (see routes/chat.py and
    utils/intent.get_removal_target_async on the main app side): plain
    img2img has no mechanism to execute a "remove X" instruction — it just
    partially re-renders the whole image guided by the prompt text, so the
    named object doesn't actually disappear, it just gets restyled.

    CLIPSeg (conf/segmentation.py) finds a mask for the named object from
    its English name alone — no manual mask needed — dilated
    (config.REMOVE_MASK_DILATE_PX) to remove visible remnants of the
    object's soft edges. What repaints that mask depends on whether the
    job also carries a reconstruct_prompt (see utils/intent on the main
    app side — set only when the user's instruction described what
    should appear in the removed object's place, e.g. "remove the cat,
    recreate the perforated metal panel that was behind it", as opposed
    to a plain "remove the cat") AND config.RECONSTRUCT_ENABLED (an
    escape hatch — see that setting's own comment):

    - No reconstruct_prompt (the common case), or RECONSTRUCT_ENABLED
      turned off: LaMa (conf/models.get_lama_model()) — no text prompt at
      all, trained specifically to extend surrounding texture into a
      hole, which is the actual "just make it disappear" task. Real,
      repeated testing showed a text-prompt-driven diffusion model is the
      wrong tool for this specific case: asked to fill a masked region
      guided only by an "empty background" prompt, it still has to
      generate SOMETHING from that prompt, and in practice kept inventing
      content instead — a different cat in place of the removed one, or,
      on a base checkpoint, entirely unrelated scenery — regardless of
      mask dilation, strength, checkpoint choice, or VAE (see README.md's
      troubleshooting sections for the full history).
    - reconstruct_prompt set (and enabled): THIS is exactly the case a
      prompted diffusion model is the right tool for — the user has
      described specific, describable content (not "nothing"), which is
      precisely what INPAINT_MODEL's checkpoint (conf/models.py's
      "inpaint" slot) is trained to paint into a masked region, given a
      real prompt to work from. Full strength (1.0): for a
      removal-triggered reconstruction there's no reason to let any of
      the removed object's original pixels survive into the result.

      Width/height alignment (real, reported bug, fixed here): job["width"]/
      job["height"] reflect whatever the MAIN APP sent when it queued the
      job — this used to always be its img2img default of 512x512
      (utils/image_client.submit_job's own defaults), never the actual
      uploaded photo's real pixel dimensions, until the main app was also
      fixed to send them. init_image/mask, by contrast, are ALWAYS at the
      photo's true size (segmentation.get_mask resizes the mask up to
      init_image.size). Passing a generation width/height that doesn't
      match init_image/mask_image's own real size is harmless for a
      plain (unmasked) img2img edit — stable-diffusion.cpp just resizes
      the source internally and the whole picture still comes out as one
      coherent image — but it is NOT harmless for a MASKED edit: the mask
      needs to line up pixel-for-pixel with whatever resolution the model
      is actually generating at, or the region it paints comes back
      visibly misaligned/corrupted (a striped, scrambled-looking patch
      confined to roughly the mask's own shape, with everything outside
      it untouched) — this reproduced identically across three separate
      real tests here that were each, at the time, blamed on something
      else (a missing VAE; an upstream stable-diffusion.cpp masked-input
      bug, https://github.com/leejet/stable-diffusion.cpp/pull/926) —
      neither of which was the actual cause, since this same mismatch was
      already present in every one of those tests.

      Resolution cap (a SEPARATE real, reported bug, also fixed here):
      simply generating at init_image's own real size (rounded to a 64px
      multiple) fixed the alignment/corruption problem above, but
      introduced a new one — a real phone-camera-sized upload pushed
      stable-diffusion.cpp's UNet compute buffer past 10GB and crashed
      the worker process outright, since per-image working memory for
      SD1.x-family models scales worse than linearly with resolution.
      _fit_gen_size() resolves both problems together: it scales
      init_image/mask down (never up, and always preserving aspect
      ratio) to fit within config.RECONSTRUCT_MAX_DIMENSION on the longer
      side, THEN rounds to a 64px multiple (SD1.x needs at least a
      multiple of 8) — so generation always happens at a resolution that
      is simultaneously (a) exactly what init_image/mask were resized to,
      preserving the alignment fix, and (b) bounded regardless of how
      large the original upload is. The model's output is resized back up
      to the ORIGINAL photo size before returning it, so the delivered
      result still matches what was uploaded either way.
    """
    from PIL import Image
    from conf import segmentation

    if not job["init_image"]:
        raise RuntimeError("mode='img2img' with remove_target requires init_image")

    init_image = Image.open(io.BytesIO(job["init_image"])).convert("RGB")
    mask = segmentation.get_mask(init_image, job["remove_target"])
    if mask is None:
        seg_status = segmentation.status()
        raise RuntimeError(
            "automatic object segmentation unavailable: "
            + (seg_status["load_error"] or "CLIPSeg not installed — see requirements.txt")
        )

    reconstruct_prompt = job.get("reconstruct_prompt")
    if reconstruct_prompt and config.RECONSTRUCT_ENABLED:
        stable_diffusion = models.get_model_for_mode("inpaint")

        original_size = init_image.size
        gen_width, gen_height = _fit_gen_size(*original_size, config.RECONSTRUCT_MAX_DIMENSION)
        if (gen_width, gen_height) != original_size:
            gen_image = init_image.resize((gen_width, gen_height))
            gen_mask = mask.resize((gen_width, gen_height))
        else:
            gen_image = init_image
            gen_mask = mask

        kwargs = dict(
            prompt=reconstruct_prompt,
            negative_prompt=job["remove_target"],
            init_image=gen_image,
            mask_image=gen_mask,
            strength=1.0,
            width=gen_width,
            height=gen_height,
            sample_steps=job["steps"],
            cfg_scale=job["cfg_scale"],
        )
        if job["seed"] is not None:
            kwargs["seed"] = job["seed"]
        output = stable_diffusion.generate_image(**kwargs)
        image = output[0]
        if image.size != original_size:
            image = image.resize(original_size)
    else:
        lama = models.get_lama_model()
        if lama is None:
            lama_status = models.lama_status()
            raise RuntimeError(
                "object-removal inpainting (LaMa) unavailable: "
                + (lama_status["load_error"] or "torch not installed — see requirements.txt")
            )
        image = lama(init_image, mask)

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
