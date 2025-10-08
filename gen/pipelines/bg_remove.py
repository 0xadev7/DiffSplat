from __future__ import annotations
from typing import Tuple
from PIL import Image
import numpy as np
import torch
from loguru import logger


class BgRemover:
    """
    Tries RMBG-1.4 (transformers). If unavailable, returns original image with opaque alpha as mask.
    Output:
      - rgba: PIL RGBA (background made transparent where model estimates)
      - mask:  HxW float32 numpy array in [0,1] (1 = foreground)
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self._pipe = None
        if enabled:
            self._try_load()

    def _try_load(self):
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                "image-segmentation",
                model="briaai/RMBG-1.4",
                device=0 if "cuda" in str(self.device) else -1,
            )
            logger.info("[BgRemover] Loaded briaai/RMBG-1.4")
        except Exception as e:
            logger.warning(
                f"[BgRemover] RMBG not available ({e}); using opaque fallback."
            )
            self._pipe = None

    async def remove(self, img_rgba: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        if not self.enabled:
            if img_rgba.mode != "RGBA":
                img_rgba = img_rgba.convert("RGBA")
            alpha = np.ones((img_rgba.height, img_rgba.width), dtype=np.float32)
            return img_rgba, alpha

        if self._pipe is None:
            if img_rgba.mode != "RGBA":
                img_rgba = img_rgba.convert("RGBA")
            alpha = np.ones((img_rgba.height, img_rgba.width), dtype=np.float32)
            return img_rgba, alpha

        # RMBG returns a matte; normalize to [0,1]
        try:
            matte = self._pipe(img_rgba)[0]["mask"]  # PIL.Image
            matte = matte.convert("L")
            alpha = np.array(matte).astype(np.float32) / 255.0
            rgba = img_rgba.convert("RGBA")
            arr = np.array(rgba).astype(np.uint8)
            arr[..., 3] = (alpha * 255.0).clip(0, 255).astype(np.uint8)
            out = Image.fromarray(arr, mode="RGBA")
            return out, alpha
        except Exception as e:
            logger.warning(
                f"[BgRemover] error during inference: {e}; using opaque fallback."
            )
            if img_rgba.mode != "RGBA":
                img_rgba = img_rgba.convert("RGBA")
            alpha = np.ones((img_rgba.height, img_rgba.width), dtype=np.float32)
            return img_rgba, alpha
