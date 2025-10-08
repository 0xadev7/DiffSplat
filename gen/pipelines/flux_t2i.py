from __future__ import annotations
from typing import Optional
from PIL import Image
import torch
from loguru import logger

try:
    from diffusers import AutoPipelineForText2Image
except Exception:
    AutoPipelineForText2Image = None


class FluxT2I:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        allow_tf32: bool,
        seed: int,
        resolution: int,
    ):
        self.model_id = model_id or "black-forest-labs/FLUX.1-schnell"
        self.device = device
        self.dtype = dtype
        self.resolution = int(resolution or 1024)
        self.seed = seed
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.pipe = None
        if AutoPipelineForText2Image is None:
            logger.warning(
                "[FluxT2I] diffusers AutoPipelineForText2Image unavailable; will return blank placeholders."
            )
        else:
            try:
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model_id, torch_dtype=self.dtype
                )
                self.pipe.set_progress_bar_config(disable=True)
                self.pipe.to(self.device)
            except Exception as e:
                logger.warning(f"[FluxT2I] failed to load {self.model_id}: {e}")
                self.pipe = None

    async def generate_pil(self, prompt: str) -> Image.Image:
        if self.pipe is None:
            # Fallback: return uniform gray image so pipeline continues (validator will score low)
            return Image.new(
                "RGBA", (self.resolution, self.resolution), (200, 200, 200, 255)
            )

        gen = torch.Generator(device=self.device)
        if self.seed >= 0:
            gen = gen.manual_seed(self.seed)

        with torch.autocast("cuda", dtype=self.dtype):
            out = self.pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=6,  # FLUX.1-schnell is fast; keep small.
                guidance_scale=2.0,
                width=self.resolution,
                height=self.resolution,
                generator=gen,
            )
        img = out.images[0]
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return img
