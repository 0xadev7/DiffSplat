import os
import io
import argparse
from time import time
from typing import Optional, Tuple, List

import torch
import numpy as np
import imageio

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
from loguru import logger
from omegaconf import OmegaConf

# ===== DiffSplat / SD3 imports (mirroring infer_gsdiff_sd3.py) =====
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL

from src.options import opt_dict
from src.models import GSAutoencoderKL, GSRecon, ElevEst
import src.utils.util as util
import src.utils.geo_util as geo_util
import src.utils.vis_util as vis_util

from extensions.diffusers_diffsplat import (
    SD3TransformerMV2DModel,
    StableMVDiffusion3Pipeline,
    FlowDPMSolverMultistepScheduler,
)

# =========================
# CLI & App Wiring
# =========================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=10006)
    # DiffSplat runtime config (maps to infer_gsdiff_sd3.py expectations)
    p.add_argument("--config_file", type=str, default="configs/gsdiff_sd35m_80g.yaml")
    p.add_argument("--output_dir", type=str, default="out")
    p.add_argument("--tag", type=str, default="gsdiff_gobj83k_sd35m__render")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--infer_from_iter", type=int, default=-1)

    # Optional performance toggles
    p.add_argument("--half_precision", action="store_true", help="Use bfloat16 autocast")
    p.add_argument("--allow_tf32", action="store_true", help="Enable TF32 on matmul")
    p.add_argument("--scheduler_type", type=str, default="flow")
    p.add_argument("--num_inference_steps", type=int, default=18)  # tuned for <=30s
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--triangle_cfg_scaling", action="store_true", help="Triangle CFG")
    p.add_argument("--min_guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    # Pretrained tags (match infer script defaults)
    p.add_argument("--load_pretrained_gsrecon", type=str, default="gsrecon_gobj265k_cnp_even4")
    p.add_argument("--load_pretrained_gsrecon_ckpt", type=int, default=-1)
    p.add_argument("--load_pretrained_gsvae", type=str, default="gsvae_gobj265k_sd3")
    p.add_argument("--load_pretrained_gsvae_ckpt", type=int, default=-1)

    return p.parse_args()

args = get_args()
app = FastAPI()


# =========================
# Config + Model State
# =========================

class DiffSplatState:
    """
    Preloads & owns all heavy models, cameras, and pipeline state,
    exposing lightweight generate methods used by FastAPI handlers.
    """
    def __init__(self, cfg_path: str, output_dir: str, gpu_id: int):
        self.gpu = f"cuda:{gpu_id}"
        self.device = torch.device(self.gpu)
        self.output_dir = output_dir

        # Load config YAML via util.get_configs (keeps parity with infer script)
        self.configs = util.get_configs(cfg_path, [])
        opt = opt_dict[self.configs["opt_type"]]
        if "opt" in self.configs:
            for k, v in self.configs["opt"].items():
                setattr(opt, k, v)
        self.opt = opt

        # Performance toggles
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TF32 enabled")

        # Tokenizers / encoders / VAE
        tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer_2")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="text_encoder_2", variant="fp16"
        )
        tokenizer_3 = T5TokenizerFast.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer_3")
        text_encoder_3 = T5EncoderModel.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="text_encoder_3", variant="fp16"
        )
        vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")

        # GS models
        gsvae = GSAutoencoderKL(opt)
        gsrecon = GSRecon(opt)

        # Scheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
        if "dpmsolver" in args.scheduler_type:
            new_noise_scheduler = FlowDPMSolverMultistepScheduler.from_pretrained(
                opt.pretrained_model_name_or_path, subfolder="scheduler"
            )
            new_noise_scheduler.config.algorithm_type = args.scheduler_type
            new_noise_scheduler.config.flow_shift = noise_scheduler.config.shift
            noise_scheduler = new_noise_scheduler

        # Load SD3 transformer checkpoint
        # This mirrors infer script’s structure: .../out/<tag>/checkpoints/<iter>/transformer_ema
        exp_tag = args.tag or "runtime_server"
        self.exp_dir = os.path.join(self.output_dir, exp_tag)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        infer_iter = util.load_ckpt(self.ckpt_dir, args.infer_from_iter, None, None)
        self.infer_iter = infer_iter
        ckpt_path = os.path.join(self.ckpt_dir, f"{infer_iter:06d}")
        os.system(f"python3 extensions/merge_safetensors.py {ckpt_path}/transformer_ema")

        in_channels = 16
        if opt.input_concat_plucker:
            in_channels += 6
        if opt.input_concat_binary_mask:
            in_channels += 1

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
            assert len(v) == 0, f"Non-empty loading info for {k}: {v}"

        # Load pretrained GSVAE / GSRecon
        gsvae = util.load_ckpt(
            os.path.join(self.output_dir, args.load_pretrained_gsvae, "checkpoints"),
            args.load_pretrained_gsvae_ckpt,
            None,
            gsvae,
        )
        gsrecon = util.load_ckpt(
            os.path.join(self.output_dir, args.load_pretrained_gsrecon, "checkpoints"),
            args.load_pretrained_gsrecon_ckpt,
            None,
            gsrecon,
        )

        # Move to device & eval
        for m in [text_encoder, text_encoder_2, text_encoder_3, vae, gsvae, gsrecon, transformer]:
            m.requires_grad_(False)
            m.eval()
            m.to(self.device)

        # Build diffusion pipeline (single-time init)
        self.pipeline = StableMVDiffusion3Pipeline(
            text_encoder=text_encoder, tokenizer=tokenizer,
            text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3, tokenizer_3=tokenizer_3,
            vae=vae, transformer=transformer, scheduler=noise_scheduler,
        )
        self.pipeline.set_progress_bar_config(disable=True)

        # Seed (deterministic but can be overridden)
        self.generator = (
            torch.Generator(device=self.device).manual_seed(args.seed)
            if args.seed >= 0 else None
        )

        # Precompute canonical 4-view camera rig & (optional) plücker
        self.V_in = self.opt.num_input_views
        fxfycxcy = torch.tensor([self.opt.fxfy, self.opt.fxfy, 0.5, 0.5], device=self.device).float()
        elevation = 10.0  # good default for text-only gen
        self._elevations = torch.tensor([-elevation] * 4, device=self.device).deg2rad().float()
        self._azimuths = torch.tensor([0., 90., 180., 270.], device=self.device).deg2rad().float()
        self._radius = torch.tensor([1.4] * 4, device=self.device).float()

        input_C2W = geo_util.orbit_camera(self._elevations, self._azimuths, self._radius, is_degree=False)
        input_C2W[:, :3, 1:3] *= -1  # OpenGL -> OpenCV
        self.input_C2W = input_C2W  # (V_in, 4, 4)
        self.input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(self.input_C2W.shape[0], 1)

        if self.opt.input_concat_plucker:
            H = W = self.opt.input_res
            plucker, _ = geo_util.plucker_ray(H, W, self.input_C2W.unsqueeze(0), self.input_fxfycxcy.unsqueeze(0))
            plucker = plucker.squeeze(0)  # (V_in, 6, H, W)
            if self.opt.view_concat_condition:
                plucker = torch.cat([plucker[0:1, ...], plucker], dim=0)  # (V_in+1, ...)
            self.plucker = plucker
        else:
            self.plucker = None

        self.gsvae = gsvae
        self.gsrecon = gsrecon

        logger.info(f"DiffSplat server ready @ iter {self.infer_iter:06d}")

    @torch.no_grad()
    def _run_latents(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, lowres, artifacts, worst quality, deformed",
        steps: int = 18,
        guidance: float = 5.0,
        triangle_cfg: bool = True,
        min_guidance: float = 1.0,
        half_precision: bool = True,
    ) -> torch.Tensor:
        """
        Runs the text-to-latent generation via StableMVDiffusion3Pipeline.
        Returns latents tensor (to be decoded/rendered by GSVAE).
        """
        amp_dtype = torch.bfloat16 if (half_precision and torch.cuda.is_available()) else torch.float32
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = self.pipeline(
                image=None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_2=prompt,
                negative_prompt_2=negative_prompt,
                prompt_3=prompt,
                negative_prompt_3=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                triangle_cfg_scaling=triangle_cfg,
                min_guidance_scale=min_guidance,
                max_guidance_scale=guidance,
                output_type="latent",
                generator=self.generator,
                plucker=self.plucker,
                num_views=self.V_in,
                init_std=0.0,
                init_noise_strength=0.98,
                init_bg=0.0,
            ).images
        return out

    @torch.no_grad()
    def _decode_render(
        self,
        latents: torch.Tensor,
        render_res: Optional[int] = None,
        opacity_threshold: float = 0.0,
        scaling_modifier: float = 1.0,
        target_C2W: Optional[torch.Tensor] = None,
        target_fx: Optional[torch.Tensor] = None,
    ):
        """
        Decodes latents to GS and optionally renders from target views.
        Returns dict from gsvae.decode_and_render_gslatents.
        """
        latents = latents / self.gsvae.scaling_factor + self.gsvae.shift_factor
        return self.gsvae.decode_and_render_gslatents(
            self.gsrecon,
            latents,
            self.input_C2W.unsqueeze(0),
            self.input_fxfycxcy.unsqueeze(0),
            target_C2W.unsqueeze(0).unsqueeze(0) if target_C2W is not None else None,
            target_fx.unsqueeze(0).unsqueeze(0) if target_fx is not None else None,
            height=render_res or self.opt.input_res,
            width=render_res or self.opt.input_res,
            scaling_modifier=scaling_modifier,
            opacity_threshold=opacity_threshold,
        )

    def generate_ply_bytes(self, prompt: str) -> bytes:
        """
        One-shot: prompt -> latents -> GS -> PLY bytes in memory.
        """
        t0 = time()
        latents = self._run_latents(
            prompt=prompt,
            steps=args.num_inference_steps,
            guidance=args.guidance_scale,
            triangle_cfg=True if args.triangle_cfg_scaling or args.triangle_cfg_scaling is False else True,
            min_guidance=args.min_guidance_scale,
            half_precision=args.half_precision or True,
        )
        render = self._decode_render(latents, render_res=self.opt.input_res, opacity_threshold=0.01)
        pc = render["pc"][0]
        buf = io.BytesIO()
        pc.save_ply(buf, opacity_threshold=0.01)
        buf.seek(0)
        t1 = time()
        logger.info(f"[/generate] prompt='{prompt}' done in {t1 - t0:.2f}s")
        return buf.getvalue()

    def generate_orbit_mp4(self, prompt: str, res: int = 1088) -> io.BytesIO:
        """
        Prompt -> latents -> orbit render -> MP4 bytes.
        """
        t0 = time()
        latents = self._run_latents(
            prompt=prompt,
            steps=max(14, min(args.num_inference_steps, 22)),  # keep video path snappy
            guidance=args.guidance_scale,
            triangle_cfg=True,
            min_guidance=args.min_guidance_scale,
            half_precision=args.half_precision or True,
        )

        # Build a 360° orbit (every 2°) for smooth 30fps ~12s video
        render_azimuths = np.arange(0.0, 360.0, 2.0)
        fxfycxcy = torch.tensor([self.opt.fxfy, self.opt.fxfy, 0.5, 0.5], device=self.device).float()
        elevation = 10.0
        radius = 1.4

        frames: List[np.ndarray] = []
        for k, azi in enumerate(render_azimuths):
            c2w = torch.from_numpy(geo_util.orbit_camera(-elevation, azi, radius=radius, opengl=True)).to(self.device)
            c2w[:3, 1:3] *= -1
            fxfycxcy_V = fxfycxcy

            render = self._decode_render(
                latents,
                render_res=res,
                opacity_threshold=0.01,
                scaling_modifier=1.0,
                target_C2W=c2w,
                target_fx=fxfycxcy_V,
            )
            img = render["image"].squeeze(0).squeeze(0)  # (3,H,W)
            frames.append(vis_util.tensor_to_image(img))  # (H,W,3) uint8

        # Encode MP4 to memory (requires ffmpeg/ffmpeg-plugin available)
        mp4_buf = io.BytesIO()
        imageio.mimwrite(mp4_buf, np.stack(frames, axis=0), fps=30, format="mp4")
        mp4_buf.seek(0)
        t1 = time()
        logger.info(f"[/generate_video] prompt='{prompt}' frames={len(frames)} done in {t1 - t0:.2f}s")
        return mp4_buf


def get_config() -> OmegaConf:
    # Keep a small OmegaConf for parity with the base server dependency
    # (not used to drive generation — DiffSplat config is in args.config_file)
    conf = OmegaConf.create({
        "it ers": 0,  # dummy; present to mirror original signature
    })
    return conf


@app.on_event("startup")
def startup_event() -> None:
    global STATE
    torch.cuda.set_device(args.gpu_id)
    STATE = DiffSplatState(args.config_file, args.output_dir, args.gpu_id)


# =========================
# FastAPI Endpoints
# =========================

@app.post("/generate/")
async def generate(
    prompt: str = Form(...),
    opt: OmegaConf = Depends(get_config),
) -> Response:
    t0 = time()
    ply_bytes = STATE.generate_ply_bytes(prompt.strip())
    logger.info(f"Generation took {(time() - t0):.2f}s total")
    return Response(ply_bytes, media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(...),
    video_res: int = Form(1088),
    opt: OmegaConf = Depends(get_config),
):
    t0 = time()
    mp4_buf = STATE.generate_orbit_mp4(prompt.strip(), res=video_res)
    logger.info(f"Video total took {(time() - t0):.2f}s")
    return StreamingResponse(content=mp4_buf, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
