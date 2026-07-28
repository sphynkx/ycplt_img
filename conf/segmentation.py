"""Automatic object segmentation via CLIPSeg — finds a mask for a named
object in an image (e.g. "cat") from a short text query, without needing
a manually-drawn mask.

This exists for mode="img2img" jobs carrying a remove_target (see
srv/worker.py): plain img2img has no mechanism to execute a "remove X"
instruction — it just partially re-renders the whole image guided by the
prompt text, so the named object doesn't actually disappear, it just gets
restyled. Segmenting the object first and inpainting only that region
(with the existing inpainting-tuned checkpoint, see conf/models.py) is a
real masked edit instead.

CLIPSeg (CIDAS/clipseg-rd64-refined, ~600MB) is a small CLIP-based
segmentation model, loaded via `transformers` + `torch` — a different
stack from both stable-diffusion-cpp-python (the SD models) and
llama-cpp-python (the vision/caption model), since neither of those
libraries does open-vocabulary segmentation. Loaded lazily on first actual
use, same fail-safe pattern as the vision model: if it's not
installed/downloaded, get_mask() returns None and the caller (worker.py)
turns that into a normal job error rather than crashing.
"""
from typing import Optional

from conf import config

_processor = None
_model = None
_load_failed = False
_load_error: Optional[str] = None


def status() -> dict:
    """Diagnostic snapshot for GET /health — never raises."""
    return {
        "model": config.CLIPSEG_MODEL,
        "loaded": _model is not None,
        "load_failed": _load_failed,
        "load_error": _load_error,
    }


def _get_model():
    global _processor, _model, _load_failed, _load_error
    if _model is not None:
        return _processor, _model
    if _load_failed:
        return None, None

    try:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        print(f"[models] loading CLIPSeg ({config.CLIPSEG_MODEL}) — this may take a while "
              f"the first time (downloads from Hugging Face Hub if not already cached)...")
        _processor = CLIPSegProcessor.from_pretrained(config.CLIPSEG_MODEL)
        _model = CLIPSegForImageSegmentation.from_pretrained(config.CLIPSEG_MODEL)
        print("[models] CLIPSeg loaded, staying resident in memory.")
    except Exception as e:
        print(f"[models] CLIPSeg failed to load: {e}")
        _load_error = str(e)
        _load_failed = True
        return None, None
    return _processor, _model


def get_mask(image, object_name: str, threshold: float = 0.35):
    """Returns a grayscale PIL Image mask for object_name in image (255 =
    edit this area, 0 = keep — the same convention as a manually-drawn
    inpainting mask), or None if CLIPSeg isn't available.

    threshold is applied to CLIPSeg's per-pixel sigmoid confidence map;
    0.35 is a middle-ground default — lower catches more of the object's
    edges/shadow at the risk of over-masking, higher is more conservative.
    """
    processor, model = _get_model()
    if processor is None:
        return None

    import numpy as np
    import torch
    from PIL import Image

    inputs = processor(text=[object_name], images=[image], padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits: (1, H', W') low-resolution heatmap — resize back up
    # to the source image's size after thresholding into a binary mask.
    probs = torch.sigmoid(outputs.logits.squeeze(0)).numpy()
    binary = (probs > threshold).astype(np.uint8) * 255
    mask = Image.fromarray(binary, mode="L").resize(image.size)
    return mask
