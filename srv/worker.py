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
            elif job["mode"] == "img2img" and job["init_image"] and models.get_kontext_model() is not None:
                # Experimental: once FLUX.1 Kontext is configured and
                # enabled (config.KONTEXT_ENABLED, conf/models.py's
                # get_kontext_model()), it takes over every general edit
                # instruction (anything that isn't a remove_target job) —
                # no main-app changes needed, since the prompt/init_image
                # a plain img2img job already carries is exactly what
                # Kontext needs too. See _generate_kontext_edit's own
                # docstring for why this exists (plain img2img has no way
                # to follow an arbitrary described transformation).
                image_bytes = _generate_kontext_edit(job)
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


def _fit_gen_size(
    width: int, height: int, max_dimension: int, multiple: int = 64, min_dimension: int = None
) -> tuple:
    """Picks the (width, height) to actually generate at for a masked
    reconstruct_prompt edit — aligned with init_image/mask (the fix for
    the striped/scrambled-output corruption bug), but capped at
    max_dimension on the longer side (the fix for a SEPARATE, later
    crash: generating at a real, uncapped photo resolution pushed
    stable-diffusion.cpp's UNet compute buffer past 10GB and killed the
    worker process outright — see config.RECONSTRUCT_MAX_DIMENSION's own
    comment for the full story).

    Scales (width, height) down to fit within max_dimension, preserving
    aspect ratio, then rounds to the nearest multiple of `multiple` (SD1.x
    needs at least a multiple of 8; 64 matches this project's existing
    512x512-style defaults). The final rounding step is nudged down by one
    multiple if it would otherwise land just over max_dimension, so the
    cap is a real, honored ceiling rather than a rough target.

    `min_dimension` (optional, default None — every existing caller keeps
    its old "never scale up" behavior unless it opts in): if given, and
    the LONGER side after the max_dimension step above is still smaller
    than this, scales back UP (never past max_dimension) to reach it
    instead. Added specifically for FLUX.1 Kontext (see
    _generate_kontext_edit and config.KONTEXT_MIN_DIMENSION's own
    comment) after real, reported testing: a small 256x198 source photo —
    which this function, before min_dimension existed, would pass through
    completely untouched, since it's already well under
    KONTEXT_MAX_DIMENSION — produced a Kontext edit that came back
    identical to the input, regardless of prompt language/content or
    quantization level. Root cause, confirmed by directly comparing
    against a 2048x1536 source (which the max_dimension step alone
    naturally scaled down to Kontext's own ~1-megapixel training
    resolution, and which DID produce a real, correct edit): FLUX/Kontext
    is a patch-based diffusion transformer trained at roughly 1024x1024-
    class resolutions — fed a 256x198 image, the latent grid is over 20x
    smaller than what it was trained on, too little spatial/token budget
    for the transformer to encode a meaningful edit at all, so it
    collapses toward reproducing the (strongly-conditioning) reference
    almost unchanged. Reconstruct_prompt (SD1.x-family) has never shown
    this specific failure mode in testing here, which is why this stays
    an opt-in parameter rather than changed default behavior for every
    caller.
    """
    long_side = max(width, height)
    target_long = min(max_dimension, long_side)  # existing scale-down-only cap
    if min_dimension is not None:
        # Scale back UP to min_dimension if the source (or the already-
        # capped target) is smaller than that — but min(...) here means
        # this never pushes target_long past max_dimension even if
        # min_dimension is misconfigured larger than it.
        target_long = max(target_long, min(min_dimension, max_dimension))
    scale = target_long / long_side
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


def _generate_kontext_edit(job) -> bytes:
    """mode="img2img" WITHOUT remove_target, once FLUX.1 Kontext is
    configured and enabled (config.KONTEXT_ENABLED, conf/models.py's
    get_kontext_model()) — the experimental, general-purpose "smarter
    fallback" for edit instructions plain img2img can't follow. Real,
    reported failure this exists to fix: "redraw this photo as if she
    were a man" came back essentially unchanged through plain img2img —
    a low-strength denoise-from-image guided by a text prompt describing
    the destination picture has no way to express "apply this specific
    semantic transformation" the way a real instruction-following edit
    model does.

    Takes over EVERY non-removal edit job automatically once configured —
    no main-app changes needed at all, since the prompt/init_image a
    plain img2img job already carries (see routes/chat.py's
    _handle_image_edit_request on the main app side) is exactly what
    Kontext needs too; this function only fires when
    get_kontext_model() actually returns a loaded model, so with
    KONTEXT_ENABLED left at its default (false) or the model files
    missing, every edit job falls straight through to the unchanged
    _generate() path below.

    Uses `ref_images` (NOT `init_image`/`strength`) — confirmed by
    reading stable-diffusion-cpp-python's own generate_image
    implementation: `ref_images` is the Kontext-specific image-
    conditioning channel (resized internally by stable-diffusion.cpp,
    distinct from init_image's strength-blended denoise-from-noised-
    latent img2img path, which Kontext doesn't use — passing an image via
    init_image instead would silently engage the wrong mechanism). Also
    uses `guidance` instead of `cfg_scale`: Flux is guidance-distilled
    rather than classifier-free-guided the SD1.x way, and
    stable-diffusion-cpp-python exposes these as genuinely separate
    parameters.

    Resolution: capped via the same _fit_gen_size helper
    _generate_removal_edit uses above (config.KONTEXT_MAX_DIMENSION
    instead of RECONSTRUCT_MAX_DIMENSION) — applied proactively here
    rather than waiting to rediscover the exact out-of-memory crash
    _generate_removal_edit's own history already went through, since a
    12B-parameter diffusion transformer's memory scaling is at least as
    steep as SD1.x's ~1B-parameter UNet.

    ALSO capped from BELOW via config.KONTEXT_MIN_DIMENSION (a real,
    reported bug, fixed here): a small 256x198 source photo produced an
    edit that came back identical to the input — confirmed to be a
    resolution problem, not a prompt/parameter one, by re-testing at
    2048x1536 (which _fit_gen_size's existing max_dimension step alone
    naturally scales down to Kontext's own ~1-megapixel training
    resolution) and getting a correct, real edit. FLUX/Kontext is a
    patch-based diffusion transformer trained at roughly 1024x1024-class
    resolutions; fed a much smaller image, the latent grid is too small
    for it to encode a meaningful edit at all, so it collapses toward
    reproducing the (strongly-conditioning) reference almost unchanged.
    _fit_gen_size's min_dimension parameter scales SMALL source photos
    back UP (never past KONTEXT_MAX_DIMENSION) to give the transformer a
    properly-sized latent grid to work with, same as it already scales
    large ones down. See config.KONTEXT_MIN_DIMENSION's own comment for
    the real compute-time cost this trades off against.
    """
    from PIL import Image

    if not job["init_image"]:
        raise RuntimeError("mode='img2img' Kontext edit requires init_image")

    kontext = models.get_kontext_model()
    if kontext is None:
        status = models.kontext_status()
        raise RuntimeError(
            "FLUX.1 Kontext unavailable: "
            + (status["load_error"] or "not configured — see README.md")
        )

    init_image = Image.open(io.BytesIO(job["init_image"])).convert("RGB")
    original_size = init_image.size
    gen_width, gen_height = _fit_gen_size(
        *original_size, config.KONTEXT_MAX_DIMENSION, min_dimension=config.KONTEXT_MIN_DIMENSION
    )
    ref_image = (
        init_image if (gen_width, gen_height) == original_size else init_image.resize((gen_width, gen_height))
    )

    kwargs = dict(
        prompt=job["prompt"],
        ref_images=[ref_image],
        width=gen_width,
        height=gen_height,
        guidance=config.KONTEXT_GUIDANCE,
        cfg_scale=config.KONTEXT_CFG_SCALE,
        sample_steps=job["steps"],
    )
    if job["negative_prompt"]:
        kwargs["negative_prompt"] = job["negative_prompt"]
    if job["seed"] is not None:
        kwargs["seed"] = job["seed"]

    output = kontext.generate_image(**kwargs)
    image = output[0]
    if image.size != original_size:
        image = image.resize(original_size)

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
