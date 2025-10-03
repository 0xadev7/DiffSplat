from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional


# -----------------------------
# Simple ".env" loader
# -----------------------------
def load_env_file(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from a file (default: ".env").
    Tries python-dotenv if available; otherwise does a minimal parse
    of KEY=VALUE lines (ignores blanks and comments).
    """
    path = env_file or os.getenv("ENV_FILE", ".env")
    if not path or not os.path.isfile(path):
        return

    # Try python-dotenv if present
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=path, override=False)
        return
    except Exception:
        pass

    # Minimal manual parsing
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        # Best-effort only
        return


# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class Config:
    # Server
    port: int = 10006
    gpu_id: int = 0

    # Runtime config paths/tags
    config_file: str = "configs/gsdiff_sd35m_80g.yaml"
    output_dir: str = "out"
    tag: str = "gsdiff_gobj83k_sd35m__render"
    infer_from_iter: int = -1

    # Performance toggles (tuned for higher CLIP)
    half_precision: bool = True
    allow_tf32: bool = True
    scheduler_type: str = "flow"  # or "dpmsolver", "dpmsolver++"

    # Generation (tuned upwards)
    num_inference_steps: int = 22
    guidance_scale: float = 6.8      # bump for stronger prompt adherence
    triangle_cfg_scaling: bool = True
    min_guidance_scale: float = 1.4  # higher floor for CFG schedule
    seed: int = 0

    # Pretrained tags
    load_pretrained_gsrecon: str = "gsrecon_gobj265k_cnp_even4"
    load_pretrained_gsrecon_ckpt: int = -1
    load_pretrained_gsvae: str = "gsvae_gobj265k_sd3"
    load_pretrained_gsvae_ckpt: int = -1

    # Validation (tuned to *improve score* and still pass conservatively)
    vld_enabled: bool = True
    vld_model: str = "ViT-L/14"
    vld_threshold: float = 0.285     # keep conservative gate
    vld_max_retries: int = 1
    vld_sample_views: int = 3        # score on more views for stability

    # Env file path
    env_file: Optional[str] = None


# -----------------------------
# Build config from env + CLI
# -----------------------------
def get_config_from_env_and_cli(argv: Optional[list[str]] = None) -> Config:
    # 1) Load env file first (so CLI can still override)
    load_env_file()

    # 2) Build from env
    def _get_bool(env_key: str, default: bool) -> bool:
        val = os.getenv(env_key)
        if val is None:
            return default
        return val not in ("0", "false", "False", "no", "No", "")

    cfg = Config(
        port=int(os.getenv("PORT", "10006")),
        gpu_id=int(os.getenv("GPU_ID", "0")),
        config_file=os.getenv("CONFIG_FILE", "configs/gsdiff_sd35m_80g.yaml"),
        output_dir=os.getenv("OUTPUT_DIR", "out"),
        tag=os.getenv("TAG", "gsdiff_gobj83k_sd35m__render"),
        infer_from_iter=int(os.getenv("INFER_FROM_ITER", "-1")),
        half_precision=_get_bool("HALF_PRECISION", True),
        allow_tf32=_get_bool("ALLOW_TF32", True),
        scheduler_type=os.getenv("SCHEDULER_TYPE", "flow"),
        num_inference_steps=int(os.getenv("NUM_INFERENCE_STEPS", "22")),
        guidance_scale=float(os.getenv("GUIDANCE_SCALE", "6.8")),
        triangle_cfg_scaling=_get_bool("TRIANGLE_CFG_SCALING", True),
        min_guidance_scale=float(os.getenv("MIN_GUIDANCE_SCALE", "1.4")),
        seed=int(os.getenv("SEED", "0")),
        load_pretrained_gsrecon=os.getenv("LOAD_PRETRAINED_GSRECON", "gsrecon_gobj265k_cnp_even4"),
        load_pretrained_gsrecon_ckpt=int(os.getenv("LOAD_PRETRAINED_GSRECON_CKPT", "-1")),
        load_pretrained_gsvae=os.getenv("LOAD_PRETRAINED_GSVAE", "gsvae_gobj265k_sd3"),
        load_pretrained_gsvae_ckpt=int(os.getenv("LOAD_PRETRAINED_GSVAE_CKPT", "-1")),
        vld_enabled=_get_bool("VALIDATION_ENABLE", True),
        vld_model=os.getenv("VALIDATION_MODEL", "ViT-L/14"),
        vld_threshold=float(os.getenv("VALIDATION_THRESHOLD", "0.285")),
        vld_max_retries=int(os.getenv("VALIDATION_MAX_RETRIES", "1")),
        vld_sample_views=int(os.getenv("VALIDATION_SAMPLE_VIEWS", "3")),
        env_file=os.getenv("ENV_FILE"),
    )

    # 3) CLI parser to override any of the above
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int)
    p.add_argument("--gpu_id", type=int)
    p.add_argument("--config_file", type=str)
    p.add_argument("--output_dir", type=str)
    p.add_argument("--tag", type=str)
    p.add_argument("--infer_from_iter", type=int)
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--no-half_precision", dest="half_precision", action="store_false")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--no-allow_tf32", dest="allow_tf32", action="store_false")
    p.add_argument("--scheduler_type", type=str)
    p.add_argument("--num_inference_steps", type=int)
    p.add_argument("--guidance_scale", type=float)
    p.add_argument("--triangle_cfg_scaling", action="store_true")
    p.add_argument("--no-triangle_cfg_scaling", dest="triangle_cfg_scaling", action="store_false")
    p.add_argument("--min_guidance_scale", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--load_pretrained_gsrecon", type=str)
    p.add_argument("--load_pretrained_gsrecon_ckpt", type=int)
    p.add_argument("--load_pretrained_gsvae", type=str)
    p.add_argument("--load_pretrained_gsvae_ckpt", type=int)
    p.add_argument("--vld_enabled", action="store_true")
    p.add_argument("--no-vld_enabled", dest="vld_enabled", action="store_false")
    p.add_argument("--vld_model", type=str)
    p.add_argument("--vld_threshold", type=float)
    p.add_argument("--vld_max_retries", type=int)
    p.add_argument("--vld_sample_views", type=int)
    p.add_argument("--env_file", type=str)

    p.set_defaults(
        half_precision=cfg.half_precision,
        allow_tf32=cfg.allow_tf32,
        triangle_cfg_scaling=cfg.triangle_cfg_scaling,
        vld_enabled=cfg.vld_enabled,
    )

    ns = p.parse_args(argv)
    for k, v in vars(ns).items():
        if v is not None:
            setattr(cfg, k, v)

    # If an explicit env file was given via CLI, re-load it once (values that already exist stay)
    if cfg.env_file:
        load_env_file(cfg.env_file)

    return cfg
