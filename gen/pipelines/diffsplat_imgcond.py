from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np
from PIL import Image


class DiffsplatImgCond:
    """
    Image-conditioned call into StableMVDiffusion3Pipeline + GSVAE/GSRecon.

    Notes:
      - Many forks expect input_concat_binary_mask=True; we keep an RGBA flow
        and (optionally) a mask, but pass just the image tensor because the
        reference pipeline typically reads the alpha or ignores mask.
      - Adjust init_noise_strength / init_bg if your model expects different values.
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
        rgb = arr[..., :3]  # ignore alpha here; pipeline/opt may handle mask channel internally
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
        prompt: Optional[str] = "",
    ) -> Dict[str, Any]:
        img_t = self._pil_to_tensor(rgba)
        # If your StableMVDiffusion3Pipeline takes an explicit 'binary_mask' arg, you can pass:
        # mask_t = self._mask_to_tensor(mask)

        gen = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        with torch.autocast(
            "cuda", torch.bfloat16 if self.half_precision else torch.float32
        ):
            out = self.pipeline(
                image=img_t,  # image-conditioned
                prompt=prompt or "",
                prompt_2=prompt or "",
                prompt_3=prompt or "",
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
                # binary_mask=mask_t,  # uncomment if your pipeline supports it explicitly
            ).images

        return self.decode_latents(out)

    async def run_latents_only(
        self,
        rgba: Image.Image,
        mask: np.ndarray,
        steps: int,
        guidance: float,
        seed: Optional[int],
        prompt: Optional[str] = "",
    ) -> Tuple[torch.Tensor, dict]:
        img_t = self._pil_to_tensor(rgba)
        gen = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        with torch.autocast(
            "cuda", torch.bfloat16 if self.half_precision else torch.float32
        ):
            lat = self.pipeline(
                image=img_t,
                prompt=prompt or "",
                prompt_2=prompt or "",
                prompt_3=prompt or "",
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
                # binary_mask=mask_t,
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
