import os
import io
import argparse
from time import time
from typing import Optional, List, Tuple

import torch
import numpy as np
import imageio
from PIL import Image

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
from loguru import logger
from omegaconf import OmegaConf

# ===== DiffSplat / SD3 imports =====
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL

# ===== CLIP validator =====
import clip

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


# --------------------------
# Validation Settings (env)
# --------------------------
VLD_ENABLED = os.getenv("VALIDATION_ENABLE", "1") != "0"
VLD_MODEL = os.getenv("VALIDATION_MODEL", "ViT-L/14")
VLD_THRESHOLD = float(os.getenv("VALIDATION_THRESHOLD", "0.285"))  # conservative
VLD_MAX_RETRIES = int(os.getenv("VALIDATION_MAX_RETRIES", "1"))
VLD_SAMPLE_VIEWS = int(
    os.getenv("VALIDATION_SAMPLE_VIEWS", "2")
)  # how many rendered views/frames to check

# =========================
# CLI & App Wiring
# =========================


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=10006)

    # DiffSplat runtime config
    p.add_argument("--config_file", type=str, default="configs/sd3_text.yaml")
    p.add_argument("--output_dir", type=str, default="out")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--infer_from_iter", type=int, default=-1)

    # Performance toggles
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--scheduler_type", type=str, default="flow")
    p.add_argument("--num_inference_steps", type=int, default=18)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--triangle_cfg_scaling", action="store_true")
    p.add_argument("--min_guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    # Pretrained tags
    p.add_argument(
        "--load_pretrained_gsrecon", type=str, default="gsrecon_gobj265k_cnp_even4"
    )
    p.add_argument("--load_pretrained_gsrecon_ckpt", type=int, default=-1)
    p.add_argument("--load_pretrained_gsvae", type=str, default="gsvae_gobj265k_sd3")
    p.add_argument("--load_pretrained_gsvae_ckpt", type=int, default=-1)

    return p.parse_args()


args = get_args()
app = FastAPI()


# =========================
# CLIP Validator
# =========================


class ClipValidator:
    def __init__(self, device: torch.device):
        self.device = device
        if not VLD_ENABLED:
            logger.info("Validation disabled via VALIDATION_ENABLE=0")
            self.model = None
            self.preprocess = None
            return
        self.model, self.preprocess = clip.load(VLD_MODEL, device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        logger.info(f"CLIP validator loaded: {VLD_MODEL}")

    @torch.no_grad()
    def score(self, prompt: str, pil_images: List[Image.Image]) -> float:
        """Return mean cosine similarity across sampled images vs. prompt."""
        if not VLD_ENABLED or self.model is None:
            return 1.0  # effectively bypass
        # Sample a subset to save time
        imgs = pil_images[: max(1, min(VLD_SAMPLE_VIEWS, len(pil_images)))]
        image_tensors = torch.stack(
            [self.preprocess(im).to(self.device) for im in imgs], dim=0
        )
        text_tokens = clip.tokenize([prompt]).to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            image_features = self.model.encode_image(image_tensors)
            text_features = self.model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        sims = (image_features @ text_features.T).squeeze(-1)  # (N,)
        return sims.mean().item()

    def passes(self, score: float) -> bool:
        return score >= VLD_THRESHOLD


# =========================
# DiffSplat Model State
# =========================


class DiffSplatState:
    def __init__(self, cfg_path: str, output_dir: str, gpu_id: int):
        self.gpu = f"cuda:{gpu_id}"
        self.device = torch.device(self.gpu)
        self.output_dir = output_dir

        self.validator = ClipValidator(self.device)

        # Load config
        self.configs = util.get_configs(cfg_path, [])
        opt = opt_dict[self.configs["opt_type"]]
        if "opt" in self.configs:
            for k, v in self.configs["opt"].items():
                setattr(opt, k, v)
        self.opt = opt

        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TF32 enabled")

        # Tokenizers/encoders/vae
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

        # Transformer checkpoint load
        exp_tag = args.tag or "runtime_server"
        self.exp_dir = os.path.join(self.output_dir, exp_tag)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        infer_iter = util.load_ckpt(self.ckpt_dir, args.infer_from_iter, None, None)
        self.infer_iter = infer_iter
        ckpt_path = os.path.join(self.ckpt_dir, f"{infer_iter:06d}")
        os.system(
            f"python3 extensions/merge_safetensors.py {ckpt_path}/transformer_ema"
        )

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

        # To device
        for m in [te, te2, te3, vae, gsvae, gsrecon, transformer]:
            m.requires_grad_(False)
            m.eval().to(self.device)

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

        self.base_seed = args.seed
        self.generator = (
            torch.Generator(device=self.device).manual_seed(self.base_seed)
            if self.base_seed >= 0
            else None
        )

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

        logger.info(f"DiffSplat server ready @ iter {self.infer_iter:06d}")

    # ---------- Core generation primitives ----------

    @torch.no_grad()
    def _run_latents(
        self, prompt: str, steps: int, guidance: float, seed: Optional[int]
    ) -> torch.Tensor:
        # set per-attempt seed
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if args.half_precision or True else torch.float32,
        ):
            out = self.pipeline(
                image=None,
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                negative_prompt="low quality, blurry, lowres, artifacts, worst quality, deformed",
                negative_prompt_2="low quality, blurry, lowres, artifacts, worst quality, deformed",
                negative_prompt_3="low quality, blurry, lowres, artifacts, worst quality, deformed",
                num_inference_steps=steps,
                guidance_scale=guidance,
                triangle_cfg_scaling=True,
                min_guidance_scale=args.min_guidance_scale,
                max_guidance_scale=guidance,
                output_type="latent",
                generator=gen,
                plucker=self.plucker,
                num_views=self.V_in,
                init_std=0.0,
                init_noise_strength=0.98,
                init_bg=0.0,
            ).images
        return out

    @torch.no_grad()
    def _decode_gs(
        self,
        latents: torch.Tensor,
        render_res: Optional[int] = None,
        opacity_threshold: float = 0.01,
    ):
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

    # ---------- Validation-integrated flows ----------

    def _score_views(self, render_outputs, sample_n: int = VLD_SAMPLE_VIEWS):
        """Take a few views from render and compute CLIP score."""
        images = render_outputs["image"].squeeze(0)  # (V,3,H,W)
        v = images.shape[0]
        take = min(max(1, sample_n), v)
        idxs = list(range(take))
        pil_images = [
            vis_util.tensor_to_image(images[i, ...], return_pil=True) for i in idxs
        ]
        return pil_images  # return pil list; score computed in validator

    def generate_ply_bytes_validated(self, prompt: str) -> Tuple[bytes, float, int]:
        """
        Try 1 + retries attempts; return (ply_bytes, best_score, attempts_used).
        If validation fails after retries, returns (b"", score, attempts).
        """
        attempts = 0
        best_bytes, best_score = b"", -1.0

        trials = [(args.num_inference_steps, args.guidance_scale, args.seed)]
        if VLD_MAX_RETRIES > 0:
            trials.append(
                (
                    min(args.num_inference_steps + 4, 22),
                    min(args.guidance_scale + 0.8, 7.0),
                    None if args.seed < 0 else args.seed + 1337,
                )
            )

        for steps, guidance, seed in trials:
            attempts += 1
            t0 = time()
            lat = self._run_latents(prompt, steps=steps, guidance=guidance, seed=seed)
            render = self._decode_gs(lat, render_res=self.opt.input_res)
            pil_list = self._score_views(render)
            score = self.validator.score(prompt, pil_list) if VLD_ENABLED else 1.0
            logger.info(
                f"[attempt {attempts}] CLIP score={score:.3f} (steps={steps}, guidance={guidance}, seed={seed})"
            )

            # Save candidate PLY
            pc = render["pc"][0]
            buf = io.BytesIO()
            pc.save_ply_buffer(buf)
            buf.seek(0)
            ply_bytes = buf.getvalue()

            # Track best
            if score > best_score:
                best_score, best_bytes = score, ply_bytes

            if self.validator.passes(score):
                logger.info(f"Validation PASSED in {time()-t0:.2f}s")
                return ply_bytes, score, attempts

        logger.warning(
            f"Validation FAILED after {attempts} attempts; best={best_score:.3f}"
        )
        return (b"", best_score, attempts)

    def generate_orbit_mp4_validated(
        self, prompt: str, res: int = 1088
    ) -> Tuple[io.BytesIO, float, int]:
        """
        Validate on a few frames of the orbit. Same retry policy.
        Returns (mp4_buf, score, attempts). Empty BytesIO on fail.
        """
        attempts = 0
        best_buf, best_score = io.BytesIO(), -1.0

        trials = [
            (max(14, min(args.num_inference_steps, 22)), args.guidance_scale, args.seed)
        ]
        if VLD_MAX_RETRIES > 0:
            trials.append(
                (
                    min(args.num_inference_steps + 4, 22),
                    min(args.guidance_scale + 0.8, 7.0),
                    None if args.seed < 0 else args.seed + 7331,
                )
            )

        # Precompute a small set of azimuths for validation quick-check
        val_azis = [0.0, 120.0, 240.0][: max(1, VLD_SAMPLE_VIEWS)]
        full_azis = np.arange(0.0, 360.0, 2.0)

        fxfycxcy = torch.tensor(
            [self.opt.fxfy, self.opt.fxfy, 0.5, 0.5], device=self.device
        ).float()
        elevation = 10.0
        radius = 1.4

        for steps, guidance, seed in trials:
            attempts += 1
            t0 = time()
            lat = self._run_latents(prompt, steps=steps, guidance=guidance, seed=seed)

            # --- quick validation frames ---
            val_pils: List[Image.Image] = []
            for azi in val_azis:
                c2w = torch.from_numpy(
                    geo_util.orbit_camera(-elevation, azi, radius=radius, opengl=True)
                ).to(self.device)
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

            score = self.validator.score(prompt, val_pils) if VLD_ENABLED else 1.0
            logger.info(
                f"[video attempt {attempts}] CLIP score={score:.3f} (steps={steps}, guidance={guidance}, seed={seed})"
            )

            if self.validator.passes(score):
                # Render full orbit
                frames: List[np.ndarray] = []
                for azi in full_azis:
                    c2w = torch.from_numpy(
                        geo_util.orbit_camera(
                            -elevation, azi, radius=radius, opengl=True
                        )
                    ).to(self.device)
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
                imageio.mimwrite(
                    mp4_buf, np.stack(frames, axis=0), fps=30, format="mp4"
                )
                mp4_buf.seek(0)
                logger.info(f"Video render after validation took {time()-t0:.2f}s")
                return mp4_buf, score, attempts

            # track best score even if not passed
            if score > best_score:
                best_score = score
                best_buf = io.BytesIO()  # keep empty to ensure "ignored" behavior

        logger.warning(
            f"Video validation FAILED after {attempts} attempts; best={best_score:.3f}"
        )
        best_buf.seek(0)
        return best_buf, best_score, attempts


def get_config() -> OmegaConf:
    # keep the dependency to match original signature
    return OmegaConf.create({"iters": 0})


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
    ply_bytes, score, attempts = STATE.generate_ply_bytes_validated(prompt.strip())
    logger.info(
        f"[/generate] score={score:.3f}, attempts={attempts}, total={time()-t0:.2f}s"
    )
    # If validation failed, return empty body with same media type
    return Response(ply_bytes, media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(...),
    video_res: int = Form(1088),
    opt: OmegaConf = Depends(get_config),
):
    t0 = time()
    mp4_buf, score, attempts = STATE.generate_orbit_mp4_validated(
        prompt.strip(), res=video_res
    )
    logger.info(
        f"[/generate_video] score={score:.3f}, attempts={attempts}, total={time()-t0:.2f}s"
    )
    return StreamingResponse(content=mp4_buf, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
