from __future__ import annotations

import os
import argparse
from dataclasses import dataclass, asdict, replace
from typing import Optional, Callable, Any, Dict


# =============================
# .env loader (best-effort)
# =============================
def load_env_file(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from a file (default: ".env").
    Tries python-dotenv if available; otherwise minimally parses KEY=VALUE lines.
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
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        # Best effort only
        return


# =============================
# Parsing helpers
# =============================
_FALSEY = {"0", "false", "no", "off", ""}


def _parse_bool(v: str | None, default: bool) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() not in _FALSEY


def _parse_int(v: str | None, default: int) -> int:
    try:
        return int(str(v)) if v is not None else default
    except Exception:
        return default


def _parse_float(v: str | None, default: float) -> float:
    try:
        return float(str(v)) if v is not None else default
    except Exception:
        return default


def _get(
    key: str,
    default: Any,
    cast: Callable[[str | None, Any], Any],
    *,
    aliases: tuple[str, ...] = (),
    env: Dict[str, str] | None = None,
) -> Any:
    """
    Fetch env var with optional aliases. First match wins.
    """
    env = env or os.environ
    if key in env:
        return cast(env.get(key), default)
    for a in aliases:
        if a in env:
            return cast(env.get(a), default)
    return default


# =============================
# Config
# =============================
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

    # Performance toggles
    half_precision: bool = True
    allow_tf32: bool = True
    scheduler_type: str = "flow"  # or "dpmsolver", "dpmsolver++"

    # Generation
    num_inference_steps: int = 22
    guidance_scale: float = 6.8
    triangle_cfg_scaling: bool = True
    min_guidance_scale: float = 1.4
    seed: int = 0

    # Pretrained tags
    load_pretrained_gsrecon: str = "gsrecon_gobj265k_cnp_even4"
    load_pretrained_gsrecon_ckpt: int = -1
    load_pretrained_gsvae: str = "gsvae_gobj265k_sd3"
    load_pretrained_gsvae_ckpt: int = -1

    # Validation
    vld_enabled: bool = True
    vld_model: str = "ViT-L/14"
    vld_threshold: float = 0.65
    vld_max_retries: int = 1
    vld_sample_views: int = 3

    # Env file path (only used when explicitly passed)
    env_file: Optional[str] = None

    # Flux
    t2i_model_id: str = "black-forest-labs/FLUX.1-schnell"
    t2i_resolution: int = 768

    # Bg Remove
    bg_remove_enabled: bool = True

    # -------- Convenience --------
    def as_dict(self) -> dict:
        return asdict(self)

    # -------- Factories --------
    @staticmethod
    def from_env(env: Dict[str, str] | None = None) -> "Config":
        """
        Build Config from environment variables (with alias support).
        """
        env = env or os.environ

        return Config(
            # Server
            port=_get("PORT", 10006, _parse_int, env=env),
            gpu_id=_get("GPU_ID", 0, _parse_int, env=env),
            # Runtime
            config_file=_get(
                "CONFIG_FILE",
                "configs/gsdiff_sd35m_80g.yaml",
                lambda v, d: v or d,
                env=env,
            ),
            output_dir=_get("OUTPUT_DIR", "out", lambda v, d: v or d, env=env),
            tag=_get(
                "TAG",
                "gsdiff_gobj83k_sd35m_image__render",
                lambda v, d: v or d,
                env=env,
            ),
            infer_from_iter=_get("INFER_FROM_ITER", -1, _parse_int, env=env),
            # Performance
            half_precision=_get("HALF_PRECISION", False, _parse_bool, env=env),
            allow_tf32=_get("ALLOW_TF32", True, _parse_bool, env=env),
            scheduler_type=_get(
                "SCHEDULER_TYPE", "dpmsolver++", lambda v, d: (v or d), env=env
            ),
            # Generation
            num_inference_steps=_get("NUM_INFERENCE_STEPS", 22, _parse_int, env=env),
            guidance_scale=_get("GUIDANCE_SCALE", 4.0, _parse_float, env=env),
            triangle_cfg_scaling=_get(
                "TRIANGLE_CFG_SCALING", True, _parse_bool, env=env
            ),
            min_guidance_scale=_get("MIN_GUIDANCE_SCALE", 1.4, _parse_float, env=env),
            seed=_get("SEED", 0, _parse_int, env=env),
            # Pretrained
            load_pretrained_gsrecon=_get(
                "LOAD_PRETRAINED_GSRECON",
                "gsrecon_gobj265k_cnp_even4",
                lambda v, d: v or d,
                env=env,
            ),
            load_pretrained_gsrecon_ckpt=_get(
                "LOAD_PRETRAINED_GSRECON_CKPT", -1, _parse_int, env=env
            ),
            load_pretrained_gsvae=_get(
                "LOAD_PRETRAINED_GSVAE",
                "gsvae_gobj265k_sd3",
                lambda v, d: v or d,
                env=env,
            ),
            load_pretrained_gsvae_ckpt=_get(
                "LOAD_PRETRAINED_GSVAE_CKPT", -1, _parse_int, env=env
            ),
            # Validation (support old & new env keys)
            vld_enabled=_get(
                "VALIDATION_ENABLE",
                True,
                _parse_bool,
                aliases=("VLD_ENABLED",),
                env=env,
            ),
            vld_threshold=_get(
                "VALIDATION_THRESHOLD",
                0.7,
                _parse_float,
                aliases=("VLD_THRESHOLD",),
                env=env,
            ),
            vld_max_retries=_get(
                "VALIDATION_MAX_RETRIES",
                3,
                _parse_int,
                aliases=("VLD_MAX_RETRIES",),
                env=env,
            ),
            vld_sample_views=_get(
                "VALIDATION_SAMPLE_VIEWS",
                4,
                _parse_int,
                aliases=("VLD_SAMPLE_VIEWS",),
                env=env,
            ),
            # .env location (optional)
            env_file=_get("ENV_FILE", None, lambda v, d: v or d, env=env),
        )

    @staticmethod
    def from_cli(base: "Config", argv: Optional[list[str]] = None) -> "Config":
        """
        Override fields from CLI flags.
        """
        p = argparse.ArgumentParser(add_help=False)

        # Server
        p.add_argument("--port", type=int)
        p.add_argument("--gpu_id", type=int)

        # Runtime
        p.add_argument("--config_file", type=str)
        p.add_argument("--output_dir", type=str)
        p.add_argument("--tag", type=str)
        p.add_argument("--infer_from_iter", type=int)

        # Performance
        p.add_argument("--scheduler_type", type=str)
        p.add_argument("--half_precision", dest="half_precision", action="store_true")
        p.add_argument(
            "--no-half_precision", dest="half_precision", action="store_false"
        )
        p.add_argument("--allow_tf32", dest="allow_tf32", action="store_true")
        p.add_argument("--no-allow_tf32", dest="allow_tf32", action="store_false")

        # Generation
        p.add_argument("--num_inference_steps", type=int)
        p.add_argument("--guidance_scale", type=float)
        p.add_argument(
            "--triangle_cfg_scaling", dest="triangle_cfg_scaling", action="store_true"
        )
        p.add_argument(
            "--no-triangle_cfg_scaling",
            dest="triangle_cfg_scaling",
            action="store_false",
        )
        p.add_argument("--min_guidance_scale", type=float)
        p.add_argument("--seed", type=int)

        # Pretrained
        p.add_argument("--load_pretrained_gsrecon", type=str)
        p.add_argument("--load_pretrained_gsrecon_ckpt", type=int)
        p.add_argument("--load_pretrained_gsvae", type=str)
        p.add_argument("--load_pretrained_gsvae_ckpt", type=int)

        # Validation
        p.add_argument("--vld_enabled", dest="vld_enabled", action="store_true")
        p.add_argument("--no-vld_enabled", dest="vld_enabled", action="store_false")
        p.add_argument("--vld_threshold", type=float)
        p.add_argument("--vld_max_retries", type=int)
        p.add_argument("--vld_sample_views", type=int)

        # .env file
        p.add_argument("--env_file", type=str)

        p.set_defaults(
            half_precision=base.half_precision,
            allow_tf32=base.allow_tf32,
            triangle_cfg_scaling=base.triangle_cfg_scaling,
            vld_enabled=base.vld_enabled,
        )

        ns, _ = p.parse_known_args(argv)
        updates = {k: v for k, v in vars(ns).items() if v is not None}
        return replace(base, **updates)


# =============================
# Public factory
# =============================
def get_config_from_env_and_cli(argv: Optional[list[str]] = None) -> Config:
    """
    Load .env, build Config from env (with alias support), then override by CLI.
    If --env_file is provided on CLI, we re-load that .env once (without overriding existing keys),
    then rebuild from env to ensure consistency with the new .env, and finally re-apply CLI overrides.
    """
    # 1) Load default .env first
    load_env_file()

    # 2) Build from current env
    cfg = Config.from_env()

    # 3) Peek CLI for --env_file (without overriding other flags yet)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env_file", type=str)
    pre_ns, _ = pre.parse_known_args(argv)
    explicit_env_file = pre_ns.env_file

    # 4) If explicit env file is given, load (non-overriding) and rebuild from env
    if explicit_env_file:
        load_env_file(explicit_env_file)
        cfg = Config.from_env()
        cfg = replace(cfg, env_file=explicit_env_file)

    # 5) Apply full CLI overrides
    cfg = Config.from_cli(cfg, argv)

    return cfg
