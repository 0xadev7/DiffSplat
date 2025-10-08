from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np
from PIL import Image


class DiffsplatImgCond:
    """
    Image-conditioned call into StableMVDiffusion3Pipeline + GSVAE/GsRecon.
    Mirrors your text-cond setup but feeds 'image' + 'binary_mask' when available.

    Notes:
      - Some forks expect the RGBA/mask baked into 'image' and set input flags:
        input_concat_binary_mask=True. We pass the mask when present.
      - Tweak 'init_noise_strength' / 'init_bg' if your fork expects them.
    """

    def __init__(
        self,
        device,
        gsvae,
        gsrecon,
        pipeline,
        opt,
        input_C2W,
        input_fxfycxcy,
        plucker,
        half_precision: bool,
        triangle_cfg_scaling: float,
        min_guidance_scale: float,
    ):
        self.device = device
        self.gsvae = gsvae
        self.gsrecon = gsrecon
        self.pipeline = pipeline
        self.opt = opt
        self.input_C2W = input_C2W
        self.input_fxfycxcy = input_fxfycxcy
        self.plucker = plucker
        self.half_precision = half_precision
        self.triangle_cfg_scaling = triangle_cfg_scaling
        self.min_guidance_scale = min_guidance_scale

    def _pil_to_tensor(self, im: Image.Image) -> torch.Tensor:
        # HWC uint8 -> CHW float in [0,1]
        arr = np.array(im).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            rgb = arr[..., :3]
        else:
            rgb = arr[..., :3]
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return t

    def _mask_to_tensor(self, mask_f01: np.ndarray) -> torch.Tensor:
        # HxW float32 [0,1] -> BCHW
        m = torch.from_numpy(mask_f01).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return m

    async def run_image_cond(
        self,
        *,
        rgba: Image.Image,
        mask: np.ndarray,
        steps: int,
        guidance: float,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        img_t = self._pil_to_tensor(rgba)
        mask_t = self._mask_to_tensor(mask) if mask is not None else None

        gen = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if self.half_precision else torch.float16,
            enabled=True,
        ):
            out = self.pipeline(
                image=img_t,  # image-conditioned
                image_mask=mask_t,  # if your fork uses a different arg name, align here
                prompt=None,
                prompt_2=None,
                prompt_3=None,
                negative_prompt="",
                negative_prompt_2="",
                negative_prompt_3="",
                num_inference_steps=steps,
                guidance_scale=guidance,
                triangle_cfg_scaling=self.triangle_cfg_scaling,
                min_guidance_scale=self.min_guidance_scale,
                max_guidance_scale=guidance,
                output_type="latent",
                generator=gen,
                plucker=self.plucker,
                num_views=self.opt.num_input_views,
                init_std=0.0,
                init_noise_strength=0.96,
                init_bg=0.0,
            ).images

        # Decode to render dict (images + pc)
        return self.decode_latents(out)

    async def run_latents_only(
        self,
        rgba: Image.Image,
        mask: np.ndarray,
        steps: int,
        guidance: float,
        seed: Optional[int],
    ) -> Tuple[torch.Tensor, dict]:
        img_t = self._pil_to_tensor(rgba)
        mask_t = self._mask_to_tensor(mask) if mask is not None else None
        gen = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if self.half_precision else torch.float16,
            enabled=True,
        ):
            lat = self.pipeline(
                image=img_t,
                image_mask=mask_t,
                prompt=None,
                prompt_2=None,
                prompt_3=None,
                negative_prompt="",
                negative_prompt_2="",
                negative_prompt_3="",
                num_inference_steps=steps,
                guidance_scale=guidance,
                triangle_cfg_scaling=self.triangle_cfg_scaling,
                min_guidance_scale=self.min_guidance_scale,
                max_guidance_scale=guidance,
                output_type="latent",
                generator=gen,
                plucker=self.plucker,
                num_views=self.opt.num_input_views,
                init_std=0.0,
                init_noise_strength=0.96,
                init_bg=0.0,
            ).images
        return lat, {}

    def decode_latents(
        self,
        latents: torch.Tensor,
        render_res: Optional[int] = None,
        opacity_threshold: float = 0.01,
    ) -> dict:
        latents = latents / self.gsvae.scaling_factor + self.gsvae.shift_factor
        return self.gsvae.decode_and_render_gslatents(
            self.gsrecon,
            latents,
            self.input_C2W.unsqueeze(0),
            self.input_fxfycxcy.unsqueeze(0),
            height=render_res or self.opt.input_res,
            width=render_res or self.opt.input_res,
            opacity_threshold=opacity_threshold,
        )
