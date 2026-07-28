"""Model factory: picks and lazily loads the right StableDiffusion instance
for a job's mode, instead of forcing one checkpoint to serve every kind of
job.

_MODEL_SLOTS maps a job mode to the pair of conf.config variable names that
name its checkpoint path and weight type:

  - "txt2img" and "img2img" both point at MODEL_PATH/WTYPE — a base SD1.x
    checkpoint handles img2img directly (it just starts denoising from the
    given image instead of pure noise; no dedicated checkpoint is needed —
    see stable-diffusion.cpp's own img2img example, which reuses the exact
    same checkpoint as its txt2img example).
  - "inpaint" points at INPAINT_MODEL_PATH/INPAINT_WTYPE, which default to
    the same MODEL_PATH/WTYPE unless INPAINT_MODEL is set in .env.
    stable-diffusion-cpp-python's own docs note that masked inpainting
    "with a base model gives poor results" and recommend a model fine-tuned
    for inpainting — see README.md for a suggested one.

Instances are cached by (model_path, wtype), not by mode name, so two modes
that resolve to the same checkpoint+wtype (the common case, when
INPAINT_MODEL is unset) share one loaded instance instead of loading the
same weights twice.

Adding a new slot later (e.g. a captioning/vision model, once one is
wired up): add its path/wtype variables to conf/config.py, add one line to
_MODEL_SLOTS. Nothing else needs to change — srv/worker.py only ever calls
get_model_for_mode(), it doesn't know about individual checkpoints.
"""
from typing import Dict, Tuple

from conf import config

_MODEL_SLOTS = {
    "txt2img": ("MODEL_PATH", "WTYPE"),
    "img2img": ("MODEL_PATH", "WTYPE"),
    "inpaint": ("INPAINT_MODEL_PATH", "INPAINT_WTYPE"),
}

# Keyed by (model_path, wtype) rather than mode name, so slots that resolve
# to the same checkpoint share one loaded instance instead of double-loading.
_loaded: Dict[Tuple[str, str], object] = {}


def _load(model_path: str, wtype: str):
    from stable_diffusion_cpp import StableDiffusion

    print(f"[models] loading {model_path} (wtype={wtype}) — this may take a while...")
    instance = StableDiffusion(model_path=model_path, wtype=wtype)
    print(f"[models] loaded {model_path}, staying resident in memory.")
    return instance


def get_model_for_mode(mode: str):
    """Returns the (lazily loaded, cached) model instance for a job's mode.
    An unrecognized mode falls back to the txt2img slot."""
    path_attr, wtype_attr = _MODEL_SLOTS.get(mode, _MODEL_SLOTS["txt2img"])
    model_path = getattr(config, path_attr)
    wtype = getattr(config, wtype_attr)

    key = (model_path, wtype)
    if key not in _loaded:
        _loaded[key] = _load(model_path, wtype)
    return _loaded[key]


def preload_default() -> None:
    """Called once at startup: eagerly loads the txt2img/img2img model (the
    common case for nearly every job) so the daemon fails fast if it's
    missing or misconfigured — the same guarantee the old single-model
    startup had. Other slots (e.g. a dedicated inpainting model, or the
    vision model below) load lazily on first use instead (see
    get_model_for_mode), so their memory is never spent unless a job
    actually needs them."""
    get_model_for_mode("txt2img")


# ---------- Vision / captioning (mode="caption") ----------
# A separate code path from the StableDiffusion factory above: it's a
# different backend entirely (llama-cpp-python, not stable-diffusion-cpp),
# with no "wtype" concept and its own failure modes (missing files, or a
# load error worth surfacing distinctly rather than crashing the worker
# loop) — see srv/worker.py, which calls get_vision_model() and turns a
# None return into a normal job error instead of raising.
_vision_llm = None
_vision_load_failed = False
_vision_load_error = None


def _make_vision_chat_handler(clip_model_path: str):
    """llama-cpp-python consolidated its per-model multimodal chat handler
    classes (MoondreamChatHandler included) into the generic
    Llava15ChatHandler, backed by llama.cpp's newer mtmd multimodal API —
    current releases no longer have MoondreamChatHandler at all. moondream2
    uses the same "USER: ... ASSISTANT: ..." prompt style Llava15ChatHandler
    already defaults to, so it's a correct drop-in rather than a hack. Try
    the specific handler first (older installs), fall back to the generic
    one, so this keeps working across llama-cpp-python versions."""
    from llama_cpp import llama_chat_format

    handler_cls = getattr(llama_chat_format, "MoondreamChatHandler", None)
    if handler_cls is None:
        handler_cls = llama_chat_format.Llava15ChatHandler
    return handler_cls(clip_model_path=clip_model_path)


def _vision_files_exist() -> bool:
    import os

    return os.path.exists(config.VISION_MODEL_PATH) and os.path.exists(config.VISION_MMPROJ_PATH)


def vision_status() -> dict:
    """Diagnostic snapshot for GET /health — never raises."""
    return {
        "model_path": config.VISION_MODEL_PATH,
        "mmproj_path": config.VISION_MMPROJ_PATH,
        "files_found": _vision_files_exist(),
        "loaded": _vision_llm is not None,
        "load_failed": _vision_load_failed,
        "load_error": _vision_load_error,
    }


def get_vision_model():
    """Lazily loads and caches the moondream2 vision model. Returns None
    (never raises) if the files aren't present or loading fails — callers
    turn that into a normal job error rather than crashing the worker."""
    global _vision_llm, _vision_load_failed, _vision_load_error
    if _vision_llm is not None:
        return _vision_llm
    if _vision_load_failed:
        return None
    if not _vision_files_exist():
        _vision_load_failed = True
        _vision_load_error = "model files not found (VISION_MODEL_PATH / VISION_MMPROJ_PATH)"
        return None

    try:
        from llama_cpp import Llama

        print(f"[models] loading vision model {config.VISION_MODEL_PATH} — this may take a while...")
        chat_handler = _make_vision_chat_handler(config.VISION_MMPROJ_PATH)
        _vision_llm = Llama(
            model_path=config.VISION_MODEL_PATH,
            chat_handler=chat_handler,
            n_ctx=config.VISION_N_CTX,
            verbose=False,
        )
        print("[models] vision model loaded, staying resident in memory.")
    except Exception as e:
        print(f"[models] vision model failed to load: {e}")
        _vision_load_error = str(e)
        _vision_load_failed = True
        return None
    return _vision_llm
