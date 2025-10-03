from __future__ import annotations

from time import time
import base64
import os

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
import torch
from omegaconf import OmegaConf
from loguru import logger
import httpx

from .settings import get_config_from_env_and_cli, Config
from .state import DiffSplatState


app = FastAPI()
STATE: DiffSplatState | None = None
CFG: Config | None = None

# ---- NEW: helper to get validator URL (env overrides) ----
def _validator_url() -> str:
    # allow overriding via env; default to localhost
    return os.environ.get(
        "VALIDATOR_URL",
        "http://localhost:8094/validate_txt_to_3d_ply/"
    )

def get_config_dep() -> OmegaConf:
    return OmegaConf.create({"iters": 0})

@app.on_event("startup")
def startup_event() -> None:
    global STATE, CFG
    CFG = get_config_from_env_and_cli()
    torch.cuda.set_device(CFG.gpu_id)
    STATE = DiffSplatState(CFG)
    logger.info(f"Server up. Port={CFG.port}, GPU={CFG.gpu_id}")

@app.post("/generate/")
async def generate(
    prompt: str = Form(...),
    opt = Depends(get_config_dep),
) -> Response:
    assert STATE is not None
    t0 = time()
    ply_bytes, score, attempts = STATE.generate_ply_bytes_validated(prompt.strip())
    logger.info(f"[/generate] score={score:.3f}, attempts={attempts}, total={time()-t0:.2f}s")
    return Response(ply_bytes, media_type="application/octet-stream")

@app.post("/generate_and_validate/")
async def generate_and_validate(
    prompt: str = Form(...),
    # pass-through of validator knobs; defaults mirror your validator
    generate_single_preview: bool = Form(False),
    generate_grid_preview: bool = Form(False),
    preview_score_threshold: float = Form(0.6),
    compression: int = Form(0),
    opt = Depends(get_config_dep),
) -> JSONResponse:
    """
    Generates a PLY, then immediately validates it via the validator server.
    Returns JSON with base64-encoded PLY and validator response.
    """
    assert STATE is not None
    t0 = time()
    # 1) Generate PLY
    ply_bytes, gen_score, attempts = STATE.generate_ply_bytes_validated(prompt.strip())
    elapsed = time() - t0

    # 2) Base64 encode for JSON transport
    ply_b64 = base64.b64encode(ply_bytes).decode("ascii")

    # 3) Call validator
    payload = {
        "prompt": prompt,
        "prompt_image": None,  # set if you have one
        "data": ply_b64,
        "compression": compression,
        "generate_single_preview": generate_single_preview,
        "generate_grid_preview": generate_grid_preview,
        "preview_score_threshold": preview_score_threshold,
    }

    v_url = _validator_url()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=120.0)) as client:
            v_resp = await client.post(v_url, json=payload)
            v_resp.raise_for_status()
            validator_json = v_resp.json()
    except httpx.HTTPError as e:
        logger.exception("Validator call failed")
        # still return the generated PLY and gen metadata so callers can proceed
        return JSONResponse(
            status_code=502,
            content={
                "prompt": prompt,
                "gen": {"score": gen_score, "attempts": attempts, "elapsed_s": elapsed},
                "validation_error": str(e),
            },
        )

    # 4) Bundle response
    return JSONResponse(
        content={
            "prompt": prompt,
            "gen": {"score": gen_score, "attempts": attempts, "elapsed_s": elapsed},
            "validation": validator_json,
        }
    )

@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(...),
    video_res: int = Form(1088),
    opt = Depends(get_config_dep),
):
    assert STATE is not None
    t0 = time()
    mp4_buf, score, attempts = STATE.generate_orbit_mp4_validated(prompt.strip(), res=video_res)
    logger.info(f"[/generate_video] score={score:.3f}, attempts={attempts}, total={time()-t0:.2f}s")
    return StreamingResponse(content=mp4_buf, media_type="video/mp4")

if __name__ == "__main__":
    cfg = get_config_from_env_and_cli()
    uvicorn.run(app, host="0.0.0.0", port=cfg.port)
