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
#
# Was "q4_0" here — a real, reported quality bug, not a model-choice
# problem: stable-diffusion.cpp's own docs (docs/sd.md's f32/f16/q8_0/
# q5_0/q5_1/q4_0/q4_1 comparison images) show q4_0 as visibly the worst
# of the quantized options, well below f16, which is close to
# indistinguishable from full f32. q4_0 exists to fit a model into very
# limited RAM — this deployment's whole design brief (README's "Intended
# for a dedicated machine with a modern CPU... and enough RAM/SSD for the
# model checkpoint") already assumes there's RAM to spare, so there was
# no real reason to pay q4_0's quality cost here. f16 roughly doubles
# resident memory per loaded checkpoint versus q4_0 (still half of f32)
# and, unlike q4_0, does NOT prevent LoRA from being used later (per
# stable-diffusion-cpp-python's own docs: "LoRAs will not work when
# using quantized models").
WTYPE = os.environ.get("YCPLT_WTYPE", "f16")

# Optional dedicated checkpoint for masked inpainting jobs — both
# mode="inpaint" (an explicit mask) and mode="img2img" with remove_target
# (CLIPSeg's automatic mask, see conf/segmentation.py) go through this
# slot. See conf/models.py. If INPAINT_MODEL is left unset, inpaint jobs
# fall back to the same MODEL_PATH/WTYPE as everything else; they'll
# still run, just with the "poor results without a model fine-tuned for
# inpainting" caveat stable-diffusion-cpp-python's own docs mention — in
# practice, for a remove_target job (a wide mask + strength=0.95, close
# to a full repaint of that region), a base checkpoint's poor result
# tends toward visible color-noise/static rather than a subtly-off
# repaint, since it has no idea how to fill a masked region coherently
# without the extra mask/masked-image input channels an inpainting-
# tuned checkpoint's UNet was actually trained with.
INPAINT_MODEL = os.environ.get("INPAINT_MODEL", "")
_explicit_inpaint_model_path = os.environ.get("YCPLT_INPAINT_MODEL_PATH", "")
if _explicit_inpaint_model_path:
    INPAINT_MODEL_PATH = _explicit_inpaint_model_path
elif INPAINT_MODEL:
    INPAINT_MODEL_PATH = os.path.join(MODELS_DIR, INPAINT_MODEL)
else:
    INPAINT_MODEL_PATH = MODEL_PATH  # no dedicated inpainting checkpoint configured

INPAINT_WTYPE = os.environ.get("YCPLT_INPAINT_WTYPE", WTYPE)

# Optional VAE override, loaded alongside MODEL_PATH/INPAINT_MODEL_PATH.
#
# Most SD1.x/SD2.x checkpoints bundle their own VAE and need no override
# (leave unset). BUT the recommended Realistic Vision V6.0 B1 pair this
# project's own .env.example points at by default is a "noVAE" release —
# its own Hugging Face model card says so explicitly: "For version 6.0 it
# is recommended to use with VAE (to improve generation quality and get
# rid of artifacts)". Skipping this is a real, confirmed failure mode,
# not a theoretical one: it produces a scrambled/striped color-noise
# patch over the masked region — visually similar to (and easily
# confused with) the separate upstream stable-diffusion.cpp masked-input
# bug (PR #926), but with a different, simpler cause and fix: load
# stabilityai/sd-vae-ft-mse-original (vae-ft-mse-840000-ema-pruned) here.
# See install/.env.example for the download command. SDXL checkpoints
# are the OTHER reason this setting exists (SDXL's original VAE produces
# NaN/black output in fp16 on some backends) — see README.md "Going
# further: SDXL" — but that's a separate case from the noVAE issue above.
VAE_PATH = os.environ.get("YCPLT_VAE_PATH", "")
INPAINT_VAE_PATH = os.environ.get("YCPLT_INPAINT_VAE_PATH", VAE_PATH)

# Optional vision/captioning model (moondream2, loaded via llama-cpp-python
# rather than stable-diffusion-cpp — a different backend entirely) for
# mode="caption" jobs: answering "what's in this image?" instead of
# generating/editing pixels. This is the graphics service, so this model
# lives here rather than in the main chat app — see conf/models.py and
# README.md "Understanding an uploaded image". Off by default: if these
# files aren't present, caption jobs simply fail with a clear error
# (conf/models.get_vision_model() returns None, see srv/worker.py),
# generation/editing jobs are unaffected either way.
VISION_MODEL = os.environ.get("VISION_MODEL", "moondream2-text-model-f16_ct-vicuna.gguf")
_explicit_vision_model_path = os.environ.get("YCPLT_VISION_MODEL_PATH", "")
VISION_MODEL_PATH = (
    _explicit_vision_model_path
    if _explicit_vision_model_path
    else os.path.join(MODELS_DIR, VISION_MODEL)
)

VISION_MMPROJ = os.environ.get("VISION_MMPROJ", "moondream2-mmproj-f16-20250414.gguf")
_explicit_vision_mmproj_path = os.environ.get("YCPLT_VISION_MMPROJ_PATH", "")
VISION_MMPROJ_PATH = (
    _explicit_vision_mmproj_path
    if _explicit_vision_mmproj_path
    else os.path.join(MODELS_DIR, VISION_MMPROJ)
)

VISION_N_CTX = int(os.environ.get("YCPLT_VISION_N_CTX", "2048"))

# Automatic object segmentation (CLIPSeg) for mode="img2img" jobs carrying
# a remove_target (see conf/segmentation.py and README "Removing a named
# object"). Unlike every other model here, this one isn't a GGUF file you
# download manually — it's pulled automatically from the Hugging Face Hub
# the first time it's needed (via the `transformers` library) and cached
# locally, the same way ycplt's own RAG feature already auto-downloads its
# sentence-transformers embedding model. Requires internet access on first
# use only; after that it's cached and works offline.
CLIPSEG_MODEL = os.environ.get("CLIPSEG_MODEL", "CIDAS/clipseg-rd64-refined")

# How many pixels to grow CLIPSeg's mask outward before repainting it.
# CLIPSeg's confidence reliably drops off at soft edges (fur, whiskers,
# hair), so the raw thresholded mask undershoots the object's true
# silhouette and leaves a visible sliver of it just outside the masked
# region. This mattered a lot with the previous SD-based removal
# implementation (a visible remnant could bias the model into regenerating
# a similar object) and matters somewhat less with LaMa (conf/models.py's
# get_lama_model(), see srv/worker.py's _generate_removal_edit) since LaMa
# has no prompt to be biased by — but a clean, slightly generous mask still
# gives it a better boundary to blend from. 12px is a reasonable default
# for ~512px images; raise it if a removal still shows a faint remnant.
REMOVE_MASK_DILATE_PX = int(os.environ.get("YCPLT_REMOVE_MASK_DILATE_PX", "12"))

# Where torch caches files it downloads itself (torch.hub) — specifically
# LaMa's checkpoint (conf/lama_model.py). Pointed explicitly at a folder
# under MODELS_DIR rather than left at torch's own default
# (a "torch" folder under $HOME/.cache, or $XDG_CACHE_HOME), for two
# reasons: (1) under systemd with a dedicated service user, $HOME isn't
# always set to something stable/writable — a silently-different or
# missing HOME each restart would make torch re-download the checkpoint
# every time even though nothing is actually wrong; (2) every other model
# this project uses already lives under MODELS_DIR, so this answers "where
# did my downloaded models go" the same way for all of them, instead of
# LaMa's checkpoint being the one exception hidden in some user's home
# directory. Must be set before conf/lama_model.py's first download call —
# safe here since conf/config.py is imported before that ever happens.
TORCH_HOME = os.environ.get("YCPLT_TORCH_HOME", os.path.join(MODELS_DIR, ".torch_cache"))
os.environ.setdefault("TORCH_HOME", TORCH_HOME)

# (There used to be a REMOVE_TARGET_STRENGTH setting here, for the
# StableDiffusion-based removal implementation's denoising strength.
# Removed along with that implementation — LaMa (conf/models.py's
# get_lama_model()) has no "strength" concept, it isn't a diffusion model.)

# Escape hatch for reconstruct_prompt jobs (srv/worker.py's
# _generate_removal_edit — the branch that routes a remove_target job
# through the StableDiffusion "inpaint" slot instead of LaMa, when the
# user's instruction also described what should appear in the removed
# object's place).
#
# DEFAULTS TO OFF. Real, repeated testing (four separate rounds, each
# with a different targeted fix — a missing VAE; upgrading past upstream
# stable-diffusion.cpp PR #926's masked-input corruption; aligning
# generation width/height with the mask; capping that resolution to stop
# a real out-of-memory crash) never changed the actual visual defect: the
# masked region consistently comes back as the same striped/scrambled
# pattern, unchanged in shape across every one of those fixes and across
# different generation resolutions — strong evidence the cause is
# something deeper than any of the four hypotheses tried, most likely a
# genuine incompatibility between this specific SD1.5-style 5-channel
# inpainting checkpoint (Realistic Vision's inpainting variant) and this
# build of stable-diffusion-cpp-python's "unet inpainting concat"
# handling (see PR #926's own description, which patched exactly that
# mechanism but evidently not for every checkpoint/version combination).
# Continuing to chase this blind, one hypothesis per live test, stopped
# being a good use of the round-trip once the SAME defect survived four
# independent, individually-reasonable fixes — see README.md's "Why not
# StableDiffusion inpainting" for the fuller history. LaMa (the
# already-working, previously-confirmed-"substantially better" default)
# is the safe choice until this is actually root-caused with real
# hands-on access to the failing checkpoint/binding pair, e.g. testing
# the plain sd-v1-5-inpainting.ckpt checkpoint (Option A in "Editing and
# inpainting models" above) instead of Realistic Vision's inpainting
# variant, or testing outside this Python service via
# stable-diffusion.cpp's own CLI directly.
#
# Set to true only to deliberately re-enable this path for further
# investigation — with it false (the default), EVERY remove_target job
# uses LaMa unconditionally, exactly as if reconstruct_prompt were never
# sent at all, regardless of what the main app's classifier decided.
RECONSTRUCT_ENABLED = os.environ.get("YCPLT_RECONSTRUCT_ENABLED", "false").strip().lower() not in (
    "false", "0", "no", "off",
)

# Upper bound (px, longer side) for the resolution a reconstruct_prompt job
# actually generates at (srv/worker.py's _generate_removal_edit). Real,
# reported crash: generating at the uploaded photo's own real resolution
# (the fix for the width/height-mismatch corruption bug above) is only
# safe up to a point — a real phone-camera-sized photo pushed the UNet's
# compute buffer to over 10GB and crashed the process outright, since
# stable-diffusion.cpp's per-image working memory for SD1.x-family models
# scales worse than linearly with resolution (self-attention over the
# latent grid). Capping the LONGER side at this value (scaling the photo
# down to fit, preserving aspect ratio, before rounding to a 64px
# multiple) keeps memory bounded regardless of how large the upload is,
# while still keeping init_image/mask/generation resolution all mutually
# aligned (the actual fix for the corruption bug — see
# _generate_removal_edit's own docstring; this cap doesn't undo that, it
# just stops the aligned resolution from being unbounded). 512 matches
# DEFAULT_WIDTH/DEFAULT_HEIGHT above — this project's SD1.x-family
# checkpoints were trained at 512x512, so generating any larger doesn't
# reliably buy more real detail anyway, only more memory/time. Raise this
# only if the deployment machine has RAM to spare and coarser textures
# (like the perforated-metal example that motivated reconstruct_prompt)
# are worth the extra cost.
RECONSTRUCT_MAX_DIMENSION = int(os.environ.get("YCPLT_RECONSTRUCT_MAX_DIMENSION", "512"))

# ---------- FLUX.1 Kontext (experimental, general instruction-based editing) ----------
#
# Every model above (MODEL/INPAINT_MODEL, LaMa) has a narrow, fixed job —
# they're fast and reliable at exactly that job, but plain img2img (the
# fallback for any edit instruction that ISN'T a remove_target) has no
# real way to follow an arbitrary described transformation: it's just a
# low-strength denoise starting from the uploaded image, guided by a text
# prompt describing the DESTINATION picture, not an instruction to APPLY.
# Real, reported failure: "redraw this photo as if she were a man" came
# back essentially unchanged — img2img has no mechanism to express "keep
# the composition, change this one attribute" the way a real instruction-
# following edit model does.
#
# FLUX.1 Kontext [dev] (Black Forest Labs, 12B-parameter rectified flow
# transformer) is trained specifically for this: given a source image and
# a text instruction, it edits the image accordingly, not a masked region
# or a strength-blended repaint — see
# https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev. This is
# EXPERIMENTAL and off by default: it's a much heavier model (12B vs
# SD1.x's ~1B) with a real memory/time cost (see KONTEXT_MAX_DIMENSION
# below), and hasn't been tested end-to-end against real hardware from
# this codebase's own development environment — see README.md
# "Instruction-based editing (FLUX.1 Kontext, experimental)" for the
# current status and setup instructions.
KONTEXT_ENABLED = os.environ.get("YCPLT_KONTEXT_ENABLED", "false").strip().lower() not in (
    "false", "0", "no", "off",
)

# Kontext has no single "full model" file the way MODEL/INPAINT_MODEL do —
# stable-diffusion.cpp loads it from four separate files: the diffusion
# transformer itself (a GGUF quantization — a full-precision safetensors
# checkpoint is ~24GB; Q4_K_M is a reasonable quality/size balance at
# ~6.9GB, see https://huggingface.co/unsloth/FLUX.1-Kontext-dev-GGUF),
# the clip-l and t5xxl text encoders (Flux uses both, unlike SD1.x's
# single CLIP), and Flux's own VAE. Same MODELS_DIR/explicit-full-path-
# override convention as VISION_MODEL/VISION_MMPROJ above.
KONTEXT_DIFFUSION_MODEL = os.environ.get("KONTEXT_DIFFUSION_MODEL", "flux1-kontext-dev-Q4_K_M.gguf")
_explicit_kontext_diffusion_model_path = os.environ.get("YCPLT_KONTEXT_DIFFUSION_MODEL_PATH", "")
KONTEXT_DIFFUSION_MODEL_PATH = (
    _explicit_kontext_diffusion_model_path
    if _explicit_kontext_diffusion_model_path
    else os.path.join(MODELS_DIR, KONTEXT_DIFFUSION_MODEL)
)

KONTEXT_CLIP_L = os.environ.get("KONTEXT_CLIP_L", "clip_l.safetensors")
_explicit_kontext_clip_l_path = os.environ.get("YCPLT_KONTEXT_CLIP_L_PATH", "")
KONTEXT_CLIP_L_PATH = (
    _explicit_kontext_clip_l_path if _explicit_kontext_clip_l_path else os.path.join(MODELS_DIR, KONTEXT_CLIP_L)
)

# fp8 rather than the full fp16 t5xxl (~9.5GB): a meaningful size saving
# (~4.9GB) for a text encoder that's already distilled/guided rather than
# doing the heavy lifting the diffusion transformer itself does — the
# common choice in the Flux/ComfyUI community for exactly this tradeoff.
KONTEXT_T5XXL = os.environ.get("KONTEXT_T5XXL", "t5xxl_fp8_e4m3fn.safetensors")
_explicit_kontext_t5xxl_path = os.environ.get("YCPLT_KONTEXT_T5XXL_PATH", "")
KONTEXT_T5XXL_PATH = (
    _explicit_kontext_t5xxl_path if _explicit_kontext_t5xxl_path else os.path.join(MODELS_DIR, KONTEXT_T5XXL)
)

KONTEXT_VAE = os.environ.get("KONTEXT_VAE", "ae.safetensors")
_explicit_kontext_vae_path = os.environ.get("YCPLT_KONTEXT_VAE_PATH", "")
KONTEXT_VAE_PATH = (
    _explicit_kontext_vae_path if _explicit_kontext_vae_path else os.path.join(MODELS_DIR, KONTEXT_VAE)
)

# Flux is guidance-DISTILLED rather than classifier-free-guided the way
# SD1.x's cfg_scale works — stable-diffusion-cpp-python exposes this as a
# separate `guidance` parameter (distinct from `cfg_scale`, which Flux
# barely uses; left near 1.0). 2.5 matches the value used in Black Forest
# Labs' own reference usage example on the model card.
KONTEXT_GUIDANCE = float(os.environ.get("YCPLT_KONTEXT_GUIDANCE", "2.5"))
KONTEXT_CFG_SCALE = float(os.environ.get("YCPLT_KONTEXT_CFG_SCALE", "1.0"))

# Same rationale as RECONSTRUCT_MAX_DIMENSION above (real OOM crash risk
# from generating at an uncapped, potentially huge upload resolution) —
# applied proactively here rather than waiting to rediscover the same
# mistake, since a 12B diffusion transformer's memory scaling is at least
# as steep as SD1.x's UNet. 1024 (vs 512 for the SD1.x-family models
# above) since Flux's native training resolution is higher; lower this if
# the deployment machine doesn't have RAM to spare.
KONTEXT_MAX_DIMENSION = int(os.environ.get("YCPLT_KONTEXT_MAX_DIMENSION", "1024"))

# Lower bound (px, longer side) — the OTHER direction of the same problem
# KONTEXT_MAX_DIMENSION solves, discovered via real, reported testing: a
# small 256x198 source photo produced a Kontext "edit" that came back
# identical to the input, regardless of prompt language/content or which
# diffusion GGUF quant was loaded. Confirmed to genuinely be a resolution
# problem (not a prompt/parameter bug) by re-testing the exact same setup
# against a 2048x1536 source instead — which KONTEXT_MAX_DIMENSION alone
# naturally scales down to Kontext's own ~1-megapixel training
# resolution — and getting a real, correct edit. FLUX/Kontext is a
# patch-based diffusion transformer trained at roughly 1024x1024-class
# resolutions; fed a much smaller image, the latent grid is too small for
# it to encode any meaningful edit, so it collapses toward reproducing
# the (strongly-conditioning) reference almost unchanged. srv/worker.py's
# _fit_gen_size scales a source photo smaller than this back UP (never
# past KONTEXT_MAX_DIMENSION) before generation, then resizes the result
# back down to the original upload size afterward, same as it already
# does for oversized photos.
#
# REAL COST, confirmed on this project's own CPU-only test hardware:
# generating at ~256x192 took ~570s; generating at ~1024x768 (the
# resolution this setting pushes small photos up to) took ~11300s
# (~3.1 HOURS) for the same 20 sampling steps — compute scales far worse
# than linearly with resolution for a full-attention diffusion
# transformer. 768 here (rather than defaulting to the full 1024 of
# KONTEXT_MAX_DIMENSION) is a deliberate middle ground, not the
# best-quality choice — raise it toward 1024 only if the hardware and
# patience for multi-hour edits are both available; lower it (accepting
# a worse edit-quality risk on small photos, per the failure mode above)
# if even that is too slow.
KONTEXT_MIN_DIMENSION = int(os.environ.get("YCPLT_KONTEXT_MIN_DIMENSION", "768"))

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
