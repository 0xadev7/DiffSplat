from __future__ import annotations

import io
import os
from time import time
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
import imageio
from loguru import logger

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL

from src.options import opt_dict
from src.models import GSAutoencoderKL, GSRecon
import src.utils.util as util
import src.utils.geo_util as geo_util
import src.utils.vis_util as vis_util

from extensions.diffusers_diffsplat import (
    SD3TransformerMV2DModel,
    StableMVDiffusion3Pipeline,
    FlowDPMSolverMultistepScheduler,
)

from .validation import ClipValidator
from .settings import Config


class DiffSplatState:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gpu = f"cuda:{cfg.gpu_id}"
        self.device = torch.device(self.gpu)
        self.output_dir = cfg.output_dir

        # Validator
        self.validator = ClipValidator(
            device=self.device,
            enabled=cfg.vld_enabled,
            model_name=cfg.vld_model,
            threshold=cfg.vld_threshold,
            sample_views=cfg.vld_sample_views,
            use_bfloat16=True,
        )

        # Load config
        self.configs = util.get_configs(cfg.config_file, [])
        opt = opt_dict[self.configs["opt_type"]]
        if "opt" in self.configs:
            for k, v in self.configs["opt"].items():
                setattr(opt, k, v)
        self.opt = opt

        # Performance knobs
        if cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TF32 enabled")

        # Tokenizers / Encoders / VAE
        tok = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer")
        te = CLIPTextModelWithProjection.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16"
        )
        tok2 = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer_2")
        te2 = CLIPTextModelWithProjection.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="text_encoder_2", variant="fp16"
        )
        tok3 = T5TokenizerFast.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer_3")
        te3 = T5EncoderModel.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="text_encoder_3", variant="fp16"
        )
        vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")

        gsvae = GSAutoencoderKL(opt)
        gsrecon = GSRecon(opt)

        # Scheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
        if "dpmsolver" in cfg.scheduler_type:
            new_noise_scheduler = FlowDPMSolverMultistepScheduler.from_pretrained(
                opt.pretrained_model_name_or_path, subfolder="scheduler"
            )
            new_noise_scheduler.config.algorithm_type = cfg.scheduler_type
            new_noise_scheduler.config.flow_shift = noise_scheduler.config.shift
            noise_scheduler = new_noise_scheduler

        # Transformer checkpoint
        exp_tag = cfg.tag or "runtime_server"
        self.exp_dir = os.path.join(self.output_dir, exp_tag)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        infer_iter = util.load_ckpt(self.ckpt_dir, cfg.infer_from_iter, None, None)
        self.infer_iter = infer_iter
        ckpt_path = os.path.join(self.ckpt_dir, f"{infer_iter:06d}")
        os.system(f"python3 extensions/merge_safetensors.py {ckpt_path}/transformer_ema")

        in_channels = (
            16
            + (6 if opt.input_concat_plucker else 0)
            + (1 if opt.input_concat_binary_mask else 0)
        )
        transformer, loading_info = SD3TransformerMV2DModel.from_pretrained_new(
            ckpt_path,
            subfolder="transformer_ema",
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
            sample_size=opt.input_res // 8,
            in_channels=in_channels,
            zero_init_conv_in=opt.zero_init_conv_in,
            view_concat_condition=opt.view_concat_condition,
            input_concat_plucker=opt.input_concat_plucker,
            input_concat_binary_mask=opt.input_concat_binary_mask,
        )
        for k, v in loading_info.items():
            assert len(v) == 0, f"Transformer load issue for {k}: {v}"

        # Load GSVAE / GSRecon checkpoints
        gsvae = util.load_ckpt(
            os.path.join(self.output_dir, cfg.load_pretrained_gsvae, "checkpoints"),
            cfg.load_pretrained_gsvae_ckpt,
            None,
            gsvae,
        )
        gsrecon = util.load_ckpt(
            os.path.join(self.output_dir, cfg.load_pretrained_gsrecon, "checkpoints"),
            cfg.load_pretrained_gsrecon_ckpt,
            None,
            gsrecon,
        )

        # To device
        for m in [te, te2, te3, vae, gsvae, gsrecon, transformer]:
            m.requires_grad_(False)
            m.eval().to(self.device)

        # Pipeline
        self.pipeline = StableMVDiffusion3Pipeline(
            text_encoder=te,
            tokenizer=tok,
            text_encoder_2=te2,
            tokenizer_2=tok2,
            text_encoder_3=te3,
            tokenizer_3=tok3,
            vae=vae,
            transformer=transformer,
            scheduler=noise_scheduler,
        )
        self.pipeline.set_progress_bar_config(disable=True)

        # Seed
        self.base_seed = cfg.seed
        self.generator = (
            torch.Generator(device=self.device).manual_seed(self.base_seed)
            if self.base_seed >= 0
            else None
        )

        # Canonical 4-view rig
        self.V_in = self.opt.num_input_views
        fxfycxcy = torch.tensor([self.opt.fxfy, self.opt.fxfy, 0.5, 0.5], device=self.device).float()
        elevation = 10.0
        elevations = torch.tensor([-elevation] * 4, device=self.device).deg2rad().float()
        azimuths = torch.tensor([0.0, 90.0, 180.0, 270.0], device=self.device).deg2rad().float()
        radius = torch.tensor([1.4] * 4, device=self.device).float()
        input_C2W = geo_util.orbit_camera(elevations, azimuths, radius, is_degree=False)
        input_C2W[:, :3, 1:3] *= -1
        self.input_C2W = input_C2W
        self.input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(self.input_C2W.shape[0], 1)

        if self.opt.input_concat_plucker:
            H = W = self.opt.input_res
            plucker, _ = geo_util.plucker_ray(
                H, W, self.input_C2W.unsqueeze(0), self.input_fxfycxcy.unsqueeze(0)
            )
            plucker = plucker.squeeze(0)
            if self.opt.view_concat_condition:
                plucker = torch.cat([plucker[0:1, ...], plucker], dim=0)
            self.plucker = plucker
        else:
            self.plucker = None

        self.gsvae = gsvae
        self.gsrecon = gsrecon

        logger.info(f"DiffSplat ready @ iter {self.infer_iter:06d}")

    # ---------------- Core ----------------
    @torch.no_grad()
    def _run_latents(self, prompt: str, steps: int, guidance: float, seed: Optional[int]) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.cfg.half_precision):
            out = self.pipeline(
                image=None,
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                negative_prompt="low quality, blurry, lowres, artifacts, worst quality, deformed, incoherent",
                negative_prompt_2="low quality, blurry, lowres, artifacts, worst quality, deformed, incoherent",
                negative_prompt_3="low quality, blurry, lowres, artifacts, worst quality, deformed, incoherent",
                num_inference_steps=steps,
                guidance_scale=guidance,
                triangle_cfg_scaling=self.cfg.triangle_cfg_scaling,
                min_guidance_scale=self.cfg.min_guidance_scale,
                max_guidance_scale=guidance,
                output_type="latent",
                generator=gen,
                plucker=self.plucker,
                num_views=self.V_in,
                init_std=0.0,
                init_noise_strength=0.96,  # slightly lower to stabilize identity
                init_bg=0.0,
            ).images
        return out

    @torch.no_grad()
    def _decode_gs(self, latents: torch.Tensor, render_res: Optional[int] = None, opacity_threshold: float = 0.01):
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

    def _views_to_pils(self, render_outputs, sample_n: int) -> List[Image.Image]:
        images = render_outputs["image"].squeeze(0)  # (V,3,H,W)
        v = images.shape[0]
        take = min(max(1, sample_n), v)
        idxs = list(range(take))
        return [vis_util.tensor_to_image(images[i, ...], return_pil=True) for i in idxs]

    # ---------------- API helpers ----------------
    def generate_ply_bytes_validated(self, prompt: str) -> tuple[bytes, float, int]:
        attempts = 0
        best_bytes, best_score = b"", -1.0

        trials: list[tuple[int, float, Optional[int]]] = [
            (self.cfg.num_inference_steps, self.cfg.guidance_scale, self.cfg.seed)
        ]
        if self.cfg.vld_max_retries > 0:
            trials.append(
                (
                    min(self.cfg.num_inference_steps + 4, 24),
                    min(self.cfg.guidance_scale + 0.7, 7.2),
                    None if self.cfg.seed < 0 else self.cfg.seed + 1337,
                )
            )

        for steps, guidance, seed in trials:
            attempts += 1
            t0 = time()
            lat = self._run_latents(prompt, steps=steps, guidance=guidance, seed=seed)
            render = self._decode_gs(lat, render_res=self.opt.input_res, opacity_threshold=0)
            pil_list = self._views_to_pils(render, self.cfg.vld_sample_views)
            score = self.validator.score(prompt, pil_list) if self.validator.enabled else 1.0
            logger.info(f"[attempt {attempts}] CLIP={score:.3f} (steps={steps}, gs={guidance}, seed={seed})")

            pc = render["pc"][0]
            buf = io.BytesIO()
            pc.save_ply_buffer(buf)
            buf.seek(0)
            ply_bytes = buf.getvalue()

            if score > best_score:
                best_score, best_bytes = score, ply_bytes

            if self.validator.passes(score):
                logger.info(f"Validation PASSED in {time()-t0:.2f}s")
                return ply_bytes, score, attempts

        logger.warning(f"Validation FAILED after {attempts} attempts; best={best_score:.3f}")
        return (b"", best_score, attempts)

    def generate_orbit_mp4_validated(self, prompt: str, res: int = 1088) -> tuple[io.BytesIO, float, int]:
        attempts = 0
        best_buf, best_score = io.BytesIO(), -1.0

        trials: list[tuple[int, float, Optional[int]]] = [
            (max(16, min(self.cfg.num_inference_steps, 24)), self.cfg.guidance_scale, self.cfg.seed)
        ]
        if self.cfg.vld_max_retries > 0:
            trials.append(
                (
                    min(self.cfg.num_inference_steps + 4, 24),
                    min(self.cfg.guidance_scale + 0.7, 7.2),
                    None if self.cfg.seed < 0 else self.cfg.seed + 7331,
                )
            )

        val_azis = [0.0, 120.0, 240.0][: max(1, self.cfg.vld_sample_views)]
        full_azis = np.arange(0.0, 360.0, 2.0)

        fxfycxcy = torch.tensor([self.opt.fxfy, self.opt.fxfy, 0.5, 0.5], device=self.device).float()
        elevation = 10.0
        radius_val = 1.4

        for steps, guidance, seed in trials:
            attempts += 1
            t0 = time()
            lat = self._run_latents(prompt, steps=steps, guidance=guidance, seed=seed)

            # quick validation frames
            val_pils: List[Image.Image] = []
            for azi in val_azis:
                elev_t = torch.tensor([-elevation], device=self.device, dtype=torch.float32)
                azim_t = torch.tensor([float(azi)], device=self.device, dtype=torch.float32)
                rad_t = torch.tensor([radius_val], device=self.device, dtype=torch.float32)
                c2w = geo_util.orbit_camera(elev_t, azim_t, radius=rad_t, opengl=True).squeeze(0)
                c2w[:3, 1:3] *= -1

                render = self.gsvae.decode_and_render_gslatents(
                    self.gsrecon,
                    lat,
                    self.input_C2W.unsqueeze(0),
                    self.input_fxfycxcy.unsqueeze(0),
                    c2w.unsqueeze(0).unsqueeze(0),
                    fxfycxcy.unsqueeze(0).unsqueeze(0),
                    height=res,
                    width=res,
                    opacity_threshold=0.01,
                )
                img = render["image"].squeeze(0).squeeze(0)
                val_pils.append(vis_util.tensor_to_image(img, return_pil=True))

            score = self.validator.score(prompt, val_pils) if self.validator.enabled else 1.0
            logger.info(f"[video attempt {attempts}] CLIP={score:.3f} (steps={steps}, gs={guidance}, seed={seed})")

            if self.validator.passes(score):
                frames: List[np.ndarray] = []
                for azi in full_azis:
                    elev_t = torch.tensor([-elevation], device=self.device, dtype=torch.float32)
                    azim_t = torch.tensor([float(azi)], device=self.device, dtype=torch.float32)
                    rad_t = torch.tensor([radius_val], device=self.device, dtype=torch.float32)
                    c2w = geo_util.orbit_camera(elev_t, azim_t, radius=rad_t, opengl=True).squeeze(0)
                    c2w[:3, 1:3] *= -1

                    render = self.gsvae.decode_and_render_gslatents(
                        self.gsrecon,
                        lat,
                        self.input_C2W.unsqueeze(0),
                        self.input_fxfycxcy.unsqueeze(0),
                        c2w.unsqueeze(0).unsqueeze(0),
                        fxfycxcy.unsqueeze(0).unsqueeze(0),
                        height=res,
                        width=res,
                        opacity_threshold=0.01,
                    )
                    img = render["image"].squeeze(0).squeeze(0)
                    frames.append(vis_util.tensor_to_image(img))

                mp4_buf = io.BytesIO()
                imageio.mimwrite(mp4_buf, np.stack(frames, axis=0), fps=30, format="mp4")
                mp4_buf.seek(0)
                logger.info(f"Video render after validation took {time()-t0:.2f}s")
                return mp4_buf, score, attempts

            if score > best_score:
                best_score = score
                best_buf = io.BytesIO()

        logger.warning(f"Video validation FAILED after {attempts} attempts; best={best_score:.3f}")
        best_buf.seek(0)
        return best_buf, best_score, attempts
