from __future__ import annotations

import asyncio
import base64
import io
import os
from time import time
from typing import Optional, Tuple, List, Callable, Awaitable

import numpy as np
import torch
from PIL import Image
import imageio
from loguru import logger
import httpx

# ---------------- External deps you already use ----------------
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

# ---------------- Local wrappers ----------------
from .pipelines.flux_t2i import FluxT2I
from .pipelines.bg_remove import BgRemover
from .pipelines.diffsplat_imgcond import DiffsplatImgCond

# Kept for compatibility, but local CLIP is disabled.
from .settings import Config


class MinerState:
    """
    Orchestrates:
      - text -> FLUX image
      - (text image) or (input image) -> BG removal -> DiffSplat image-conditioned
      - PLY generation -> call appropriate external validator
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gpu = f"cuda:{cfg.gpu_id}"
        self.device = torch.device(self.gpu)
        self.output_dir = cfg.output_dir

        # External validator endpoints (env overrides)
        self.validator_url_text: str = os.environ.get(
            "VALIDATOR_URL_TXT",
            "http://localhost:8094/validate_txt_to_3d_ply/",
        )
        self.validator_url_image: str = os.environ.get(
            "VALIDATOR_URL_IMG",
            "http://localhost:8094/validate_img_to_3d_ply/",
        )

        
        # Load DiffSplat base (shared with text-cond; we reuse for img-cond)
        self._init_diffsplat_backbone()

        # Wrap image-conditioned recon helper
        self.imgcond = DiffsplatImgCond(
            device=self.device,
            gsvae=self.gsvae,
            gsrecon=self.gsrecon,
            pipeline=self.pipeline,
            opt=self.opt,
            input_C2W=self.input_C2W,
            input_fxfycxcy=self.input_fxfycxcy,
            plucker=self.plucker,
            half_precision=self.cfg.half_precision,
            triangle_cfg_scaling=self.cfg.triangle_cfg_scaling,
            min_guidance_scale=self.cfg.min_guidance_scale,
        )

        # Optional FLUX (lazy init is okay, but load now to avoid first-call stall)
        self.flux = FluxT2I(
            model_id=self.cfg.t2i_model_id,
            device=self.device,
            dtype=torch.bfloat16 if self.cfg.half_precision else torch.float16,
            allow_tf32=self.cfg.allow_tf32,
            seed=self.cfg.seed,
            resolution=self.cfg.t2i_resolution,
        )

        # Background remover (robust fallback)
        self.bg = BgRemover(device=self.device, enabled=self.cfg.bg_remove_enabled)

        # Seed
        self.base_seed = cfg.seed

        logger.info(f"MinerState ready @ iter {self.infer_iter:06d}")

    # ---------------- DiffSplat backbone init (shared) ----------------
    def _init_diffsplat_backbone(self) -> None:
        if self.cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TF32 enabled")

        # Load config
        self.configs = util.get_configs(self.cfg.config_file, [])
        opt = opt_dict[self.configs["opt_type"]]
        if "opt" in self.configs:
            for k, v in self.configs["opt"].items():
                setattr(opt, k, v)
        self.opt = opt

        tok = CLIPTokenizer.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        te = CLIPTextModelWithProjection.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16"
        )
        tok2 = CLIPTokenizer.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        te2 = CLIPTextModelWithProjection.from_pretrained(
            opt.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            variant="fp16",
        )
        tok3 = T5TokenizerFast.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="tokenizer_3"
        )
        te3 = T5EncoderModel.from_pretrained(
            opt.pretrained_model_name_or_path,
            subfolder="text_encoder_3",
            variant="fp16",
        )
        vae = AutoencoderKL.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="vae"
        )

        gsvae = GSAutoencoderKL(opt)
        gsrecon = GSRecon(opt)

        # Scheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
        if "dpmsolver" in self.cfg.scheduler_type:
            new_noise_scheduler = FlowDPMSolverMultistepScheduler.from_pretrained(
                opt.pretrained_model_name_or_path, subfolder="scheduler"
            )
            new_noise_scheduler.config.algorithm_type = self.cfg.scheduler_type
            new_noise_scheduler.config.flow_shift = noise_scheduler.config.shift
            noise_scheduler = new_noise_scheduler

        # Transformer checkpoint
        exp_tag = self.cfg.tag or "runtime_server"
        self.exp_dir = os.path.join(self.output_dir, exp_tag)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        infer_iter = util.load_ckpt(self.ckpt_dir, self.cfg.infer_from_iter, None, None)
        self.infer_iter = infer_iter
        ckpt_path = os.path.join(self.ckpt_dir, f"{infer_iter:06d}")
        os.system(
            f"python3 extensions/merge_safetensors.py {ckpt_path}/transformer_ema"
        )

        in_channels = (
            16
            + (6 if self.opt.input_concat_plucker else 0)
            + (1 if self.opt.input_concat_binary_mask else 0)
        )
        transformer, loading_info = SD3TransformerMV2DModel.from_pretrained_new(
            ckpt_path,
            subfolder="transformer_ema",
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
            sample_size=self.opt.input_res // 8,
            in_channels=in_channels,
            zero_init_conv_in=self.opt.zero_init_conv_in,
            view_concat_condition=self.opt.view_concat_condition,
            input_concat_plucker=self.opt.input_concat_plucker,
            input_concat_binary_mask=self.opt.input_concat_binary_mask,
        )
        for k, v in loading_info.items():
            assert len(v) == 0, f"Transformer load issue for {k}: {v}"

        # Load GSVAE / GSRecon checkpoints
        gsvae = util.load_ckpt(
            os.path.join(
                self.output_dir, self.cfg.load_pretrained_gsvae, "checkpoints"
            ),
            self.cfg.load_pretrained_gsvae_ckpt,
            None,
            gsvae,
        )
        gsrecon = util.load_ckpt(
            os.path.join(
                self.output_dir, self.cfg.load_pretrained_gsrecon, "checkpoints"
            ),
            self.cfg.load_pretrained_gsrecon_ckpt,
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

        # Canonical 4-view rig
        self.V_in = self.opt.num_input_views
        fxfycxcy = torch.tensor(
            [self.opt.fxfy, self.opt.fxfy, 0.5, 0.5], device=self.device
        ).float()
        elevation = 10.0
        elevations = (
            torch.tensor([-elevation] * 4, device=self.device).deg2rad().float()
        )
        azimuths = (
            torch.tensor([0.0, 90.0, 180.0, 270.0], device=self.device)
            .deg2rad()
            .float()
        )
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

    # ---------------- Utilities ----------------
    def _b64_to_pil(self, b64_str: str) -> Image.Image:
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw)).convert("RGBA")

    async def _call_external_validator(
        self,
        *,
        mode: str,
        prompt: Optional[str],
        prompt_image_b64: Optional[str],
        ply_bytes: bytes,
    ) -> Tuple[float, bool, dict]:
        """
        Sends base64 PLY to the external validator and returns (score, passed, raw_json).
        Uses the `score` field of ValidationResponse directly.
        """
        url = self.validator_url_text if mode == "text" else self.validator_url_image
        payload = {
            "prompt": prompt,
            "prompt_image": prompt_image_b64,
            "data": base64.b64encode(ply_bytes).decode("utf-8"),
            "compression": 0,
            "generate_single_preview": False,
            "generate_grid_preview": False,
            "preview_score_threshold": float(self.cfg.vld_threshold),
        }
        try:
            timeout = httpx.Timeout(connect=3.0, read=6.0, write=3.0, pool=3.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                js = resp.json()
        except Exception as e:
            logger.warning(f"[validator:{mode}] error: {e}")
            return 0.0, False, {"error": str(e)}
        try:
            score = float(js.get("score", 0.0))
        except Exception:
            score = 0.0
        passed = score >= float(self.cfg.vld_threshold)
        return score, passed, js

    async def _retry_generation(
        self,
        prompt: Optional[str],
        prompt_image_b64: Optional[str],
        generate_fn: Callable[[int, float, Optional[int]], Awaitable[bytes]],
        validate_first_ply_fn: Callable[[bytes], Awaitable[float]],
        *,
        max_retries: int,
        seed_base: int,
        num_steps: int,
        guidance: float,
        seed_stride: int,
    ):
        best_ply = b""
        best_score = -1.0

        async def attempt(attempt_idx: int):
            cur_steps = max(min(num_steps + attempt_idx * 4, 40), 24)
            cur_guidance = max(min(guidance + attempt_idx * 0.5, 6.0), 4.0)
            cur_seed = None if seed_base < 0 else seed_base + seed_stride * attempt_idx

            t0 = time()
            ply_bytes = await generate_fn(cur_steps, cur_guidance, cur_seed)
            score = await validate_first_ply_fn(ply_bytes)
            logger.info(
                f"[attempt {attempt_idx+1}] EXT={score:.3f} "
                f"(steps={cur_steps}, gs={cur_guidance}, seed={cur_seed}, time={time()-t0:.1f}s)"
            )
            return attempt_idx, score, ply_bytes

        for attempt_idx in range(max_retries + 1):
            _, score, ply_bytes = await attempt(attempt_idx)
            if score > best_score:
                best_score, best_ply = score, ply_bytes
            if score >= float(self.cfg.vld_threshold):
                return best_ply, best_score, attempt_idx + 1

        logger.warning(
            f"Validation FAILED after {max_retries+1} attempts; best={best_score:.3f}"
        )
        return best_ply, best_score, max_retries + 1

    # ---------------- Public: TEXT path ----------------
    async def generate_from_text_to_ply(
        self, prompt: str
    ) -> tuple[memoryview, float, int]:
        """
        Text -> FLUX image -> BG removal -> DiffSplat (img-cond) -> PLY -> external validate (TXT endpoint).
        """
        # 1) text->image
        img = await self.flux.generate_pil(prompt)

        # 2) bg removal
        rgba, mask = await self.bg.remove(img)

        # 3) image-conditioned diffsplat -> PLY bytes (single pass wrapper used by retry)
        async def gen(
            cur_steps: int, cur_guidance: float, cur_seed: Optional[int]
        ) -> bytes:
            render = await self.imgcond.run_image_cond(
                rgba=rgba,
                mask=mask,
                steps=cur_steps,
                guidance=cur_guidance,
                seed=cur_seed,
            )
            pc = render["pc"][0]
            buf = io.BytesIO()
            pc.save_ply_buffer_sn17(buf)
            return buf.getvalue()

        async def vld_first(ply_bytes: bytes) -> float:
            score, _, _ = await self._call_external_validator(
                mode="text", prompt=prompt, prompt_image_b64=None, ply_bytes=ply_bytes
            )
            return score

        ply_bytes, best_score, attempts = await self._retry_generation(
            prompt=prompt,
            prompt_image_b64=None,
            generate_fn=gen,
            validate_first_ply_fn=vld_first,
            max_retries=self.cfg.vld_max_retries,
            seed_base=self.cfg.seed,
            num_steps=self.cfg.num_inference_steps,
            guidance=self.cfg.guidance_scale,
            seed_stride=1337,
        )

        return memoryview(ply_bytes), best_score, attempts

    async def generate_video_from_text(
        self, prompt: str, res: int = 1088
    ) -> tuple[io.BytesIO, float, int]:
        # generate latents via img-cond path by first making image:
        img = await self.flux.generate_pil(prompt)
        rgba, mask = await self.bg.remove(img)

        # quick-validate during retries on decoded partial PLY at `res`
        async def gen(
            cur_steps: int, cur_guidance: float, cur_seed: Optional[int]
        ) -> Tuple[torch.Tensor, dict]:
            return await self.imgcond.run_latents_only(
                rgba, mask, cur_steps, cur_guidance, cur_seed
            )

        async def vld_lat(lat_and_aux) -> float:
            lat, _aux = lat_and_aux
            # decode to ply quickly
            render = self.imgcond.decode_latents(
                lat, render_res=res, opacity_threshold=0.01
            )
            pc = render["pc"][0]
            buf = io.BytesIO()
            pc.save_ply_buffer_sn17(buf)
            ply_bytes = buf.getvalue()
            score, _, _ = await self._call_external_validator(
                mode="text", prompt=prompt, prompt_image_b64=None, ply_bytes=ply_bytes
            )
            return score

        # small adaptation of retry: we want latents; rewrap to bytes
        best_lat = None
        best_score = -1.0
        attempts = 0
        for i in range(self.cfg.vld_max_retries + 1):
            steps = max(min(self.cfg.num_inference_steps + i * 4, 40), 24)
            gs = max(min(self.cfg.guidance_scale + i * 0.5, 6.0), 4.0)
            seed = None if self.cfg.seed < 0 else self.cfg.seed + 7331 * i
            lat, aux = await gen(steps, gs, seed)
            score = await vld_lat((lat, aux))
            attempts = i + 1
            if score > best_score:
                best_score, best_lat = score, lat
            if score >= float(self.cfg.vld_threshold):
                break

        if best_lat is None:
            return io.BytesIO(), best_score, attempts

        # orbit render
        frames: List[np.ndarray] = []
        val_azis = np.arange(0.0, 360.0, 2.0)
        fxfycxcy = torch.tensor(
            [self.opt.fxfy, self.opt.fxfy, 0.5, 0.5], device=self.device
        ).float()
        elevation = 10.0
        radius_val = 1.4
        for azi in val_azis:
            elev_t = torch.tensor([-elevation], device=self.device)
            azim_t = torch.tensor([float(azi)], device=self.device)
            rad_t = torch.tensor([radius_val], device=self.device)
            c2w = geo_util.orbit_camera(
                elev_t, azim_t, radius=rad_t, opengl=True
            ).squeeze(0)
            c2w[:3, 1:3] *= -1
            render = self.gsvae.decode_and_render_gslatents(
                self.gsrecon,
                best_lat,
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

        buf = io.BytesIO()
        imageio.mimwrite(buf, np.stack(frames, axis=0), fps=30, format="mp4")
        buf.seek(0)
        return buf, best_score, attempts

    # ---------------- Public: IMAGE path ----------------
    async def generate_from_image_to_ply(
        self, image_prompt_b64: str
    ) -> tuple[memoryview, float, int]:
        """
        Image (base64) -> BG removal -> DiffSplat (img-cond) -> PLY -> external validate (IMG endpoint).
        """
        img = self._b64_to_pil(image_prompt_b64)
        rgba, mask = await self.bg.remove(img)

        async def gen(
            cur_steps: int, cur_guidance: float, cur_seed: Optional[int]
        ) -> bytes:
            render = await self.imgcond.run_image_cond(
                rgba=rgba,
                mask=mask,
                steps=cur_steps,
                guidance=cur_guidance,
                seed=cur_seed,
            )
            pc = render["pc"][0]
            buf = io.BytesIO()
            pc.save_ply_buffer_sn17(buf)
            return buf.getvalue()

        async def vld_first(ply_bytes: bytes) -> float:
            score, _, _ = await self._call_external_validator(
                mode="image",
                prompt=None,
                prompt_image_b64=image_prompt_b64,
                ply_bytes=ply_bytes,
            )
            return score

        ply_bytes, best_score, attempts = await self._retry_generation(
            prompt=None,
            prompt_image_b64=image_prompt_b64,
            generate_fn=gen,
            validate_first_ply_fn=vld_first,
            max_retries=self.cfg.vld_max_retries,
            seed_base=self.cfg.seed,
            num_steps=self.cfg.num_inference_steps,
            guidance=self.cfg.guidance_scale,
            seed_stride=1777,
        )
        return memoryview(ply_bytes), best_score, attempts

    async def generate_video_from_image(
        self, image_prompt_b64: str, res: int = 1088
    ) -> tuple[io.BytesIO, float, int]:
        img = self._b64_to_pil(image_prompt_b64)
        rgba, mask = await self.bg.remove(img)

        # same pattern as text version
        best_lat = None
        best_score = -1.0
        attempts = 0
        for i in range(self.cfg.vld_max_retries + 1):
            steps = max(min(self.cfg.num_inference_steps + i * 4, 40), 24)
            gs = max(min(self.cfg.guidance_scale + i * 0.5, 6.0), 4.0)
            seed = None if self.cfg.seed < 0 else self.cfg.seed + 7559 * i
            lat, aux = await self.imgcond.run_latents_only(rgba, mask, steps, gs, seed)
            render = self.imgcond.decode_latents(
                lat, render_res=res, opacity_threshold=0.01
            )
            pc = render["pc"][0]
            buf = io.BytesIO()
            pc.save_ply_buffer_sn17(buf)
            ply_bytes = buf.getvalue()
            score, _, _ = await self._call_external_validator(
                mode="image",
                prompt=None,
                prompt_image_b64=image_prompt_b64,
                ply_bytes=ply_bytes,
            )
            attempts = i + 1
            if score > best_score:
                best_score, best_lat = score, lat
            if score >= float(self.cfg.vld_threshold):
                break

        if best_lat is None:
            return io.BytesIO(), best_score, attempts

        frames: List[np.ndarray] = []
        val_azis = np.arange(0.0, 360.0, 2.0)
        fxfycxcy = torch.tensor(
            [self.opt.fxfy, self.opt.fxfy, 0.5, 0.5], device=self.device
        ).float()
        elevation = 10.0
        radius_val = 1.4
        for azi in val_azis:
            elev_t = torch.tensor([-elevation], device=self.device)
            azim_t = torch.tensor([float(azi)], device=self.device)
            rad_t = torch.tensor([radius_val], device=self.device)
            c2w = geo_util.orbit_camera(
                elev_t, azim_t, radius=rad_t, opengl=True
            ).squeeze(0)
            c2w[:3, 1:3] *= -1
            render = self.gsvae.decode_and_render_gslatents(
                self.gsrecon,
                best_lat,
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

        buf = io.BytesIO()
        imageio.mimwrite(buf, np.stack(frames, axis=0), fps=30, format="mp4")
        buf.seek(0)
        return buf, best_score, attempts
