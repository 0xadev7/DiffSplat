from __future__ import annotations

from time import time

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import torch
from omegaconf import OmegaConf
from loguru import logger

from .settings import get_config_from_env_and_cli, Config
from .state import DiffSplatState


app = FastAPI()
STATE: DiffSplatState | None = None
CFG: Config | None = None


def get_config_dep() -> OmegaConf:
    # Keep signature compatibility with your previous Depends
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
