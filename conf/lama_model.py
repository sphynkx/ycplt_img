"""Vendored LaMa (Large Mask Inpainting) loading + inference logic.

This used to be a plain `pip install simple-lama-inpainting` dependency
(see conf/models.get_lama_model()). Vendored instead after a real,
reported install failure: that package (last released Dec 2022, PyPI:
https://pypi.org/project/simple-lama-inpainting/) pins
`pillow>=9.5.0,<10.0.0` and `numpy>=1.24.3,<2.0.0` — versions old enough
that Python 3.14 has no prebuilt wheel for them at all, forcing pip to
compile Pillow from source, which then fails on any system missing
libjpeg's development headers ("headers or library files could not be
found for jpeg"). That's not a problem with this project's own
dependencies — this file and conf/models.py only ever need whatever
torch/numpy/Pillow versions are already installed for CLIPSeg
(conf/segmentation.py) and the rest of this project; there was never a
real reason to pull in a second, older, conflicting set of pins for what
amounts to about 60 lines of actual logic.

The package's own upstream logic also unconditionally imports `cv2`
(opencv-python) at module level, even though that import is only used by
a code path (`scale_image`, for a `scale_factor` argument) this project
never exercises — dropped entirely here, so this file needs no new
dependency beyond what's already required.

Adapted from https://github.com/enesmsahin/simple-lama-inpainting
(Apache-2.0 license), which itself credits https://github.com/advimman/lama
(the original LaMa research code, Suvorov et al. 2021) for the core
algorithm this pre/post-processing wraps.
"""
import os
import sys
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file, get_dir

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)


def _get_image_array(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError("Input image should be either PIL Image or numpy array!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3
    return img.astype(np.float32) / 255


def _ceil_modulo(x: int, mod: int) -> int:
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def _pad_img_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    channels, height, width = img.shape
    out_height = _ceil_modulo(height, mod)
    out_width = _ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def _prepare_img_and_mask(image, mask, device, pad_out_to_modulo: int = 8):
    out_image = _get_image_array(image)
    out_mask = _get_image_array(mask)

    if pad_out_to_modulo and pad_out_to_modulo > 1:
        out_image = _pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = _pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)
    out_mask = (out_mask > 0) * 1

    return out_image, out_mask


def _cache_path_for_url(url: str) -> str:
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    filename = os.path.basename(urlparse(url).path)
    return os.path.join(model_dir, filename)


def _download_model(url: str) -> str:
    cached_file = _cache_path_for_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=True)
    return cached_file


class SimpleLama:
    """Drop-in equivalent of simple_lama_inpainting.SimpleLama's public
    API: `SimpleLama()(image, mask) -> PIL.Image`, image/mask as PIL
    Images or numpy arrays, mask 255=inpaint/0=keep."""

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path_override = os.environ.get("LAMA_MODEL")
        if model_path_override:
            if not os.path.exists(model_path_override):
                raise FileNotFoundError(f"lama torchscript model not found: {model_path_override}")
            model_path = model_path_override
        else:
            model_path = _download_model(LAMA_MODEL_URL)

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image, mask):
        image_t, mask_t = _prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image_t, mask_t)

            result = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(result)
