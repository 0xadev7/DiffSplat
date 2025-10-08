from __future__ import annotations

import asyncio
from time import time
from typing import Optional

from fastapi import FastAPI, Depends, Form, HTTPException
from fastapi.responses import Response, StreamingResponse
import uvicorn
import torch
from omegaconf import OmegaConf
from loguru import logger

from .settings import get_config_from_env_and_cli, Config
from .state import MinerState

app = FastAPI()
STATE: MinerState | None = None
CFG: Config | None = None


def get_config_dep() -> OmegaConf:
    # Per-request overrides could be added later via this dep
    return OmegaConf.create({})


@app.on_event("startup")
def startup_event() -> None:
    global STATE, CFG
    CFG = get_config_from_env_and_cli()
    torch.cuda.set_device(CFG.gpu_id)
    STATE = MinerState(CFG)
    logger.info(
        f"Server up. Port={CFG.port}, GPU={CFG.gpu_id}, "
        f"Validators: txt={STATE.validator_url_text}, img={STATE.validator_url_image}"
    )


def _choose_mode(prompt: Optional[str], image_prompt: Optional[str]) -> str:
    if prompt and image_prompt:
        raise HTTPException(400, "Provide either 'prompt' or 'image_prompt', not both.")
    if not prompt and not image_prompt:
        raise HTTPException(400, "Provide one of 'prompt' or 'image_prompt'.")
    return "text" if prompt else "image"


@app.post("/generate/")
async def generate(
    prompt: Optional[str] = Form(None),
    image_prompt: Optional[str] = Form(None),  # base64 image
    opt=Depends(get_config_dep),
) -> Response:
    """
    Generate a PLY and validate it via the external validator.
    MUST return within ~30 seconds. If over time, returns empty bytes.
    """
    assert STATE is not None
    mode = _choose_mode(prompt, image_prompt)
    t0 = time()
    try:
        if mode == "text":
            task = STATE.generate_from_text_to_ply(prompt.strip())
        else:
            task = STATE.generate_from_image_to_ply(image_prompt)

        ply_buf, gen_score, attempts = await asyncio.wait_for(task, timeout=30.0)
        elapsed = time() - t0
        logger.info(
            f"[/generate:{mode}] score={gen_score:.3f}, attempts={attempts}, total={elapsed:.2f}s"
        )
        return Response(ply_buf, media_type="application/octet-stream")
    except asyncio.TimeoutError:
        logger.warning(f"[/generate:{mode}] timed out at 30s; returning empty bytes")
        return Response(b"", media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: Optional[str] = Form(None),
    image_prompt: Optional[str] = Form(None),  # base64 image
    video_res: int = Form(1088),
    opt=Depends(get_config_dep),
):
    """
    Returns an orbit MP4 from the best-scoring latents (validated quickly).
    No hard timeout; keep your own infra watchdog if needed.
    """
    assert STATE is not None
    mode = _choose_mode(prompt, image_prompt)
    t0 = time()

    if mode == "text":
        mp4_buf, gen_score, attempts = await STATE.generate_video_from_text(
            prompt.strip(), res=video_res
        )
    else:
        mp4_buf, gen_score, attempts = await STATE.generate_video_from_image(
            image_prompt, res=video_res
        )

    logger.info(
        f"[/generate_video:{mode}] score={gen_score:.3f}, attempts={attempts}, total={time()-t0:.2f}s"
    )
    return StreamingResponse(content=mp4_buf, media_type="video/mp4")


if __name__ == "__main__":
    cfg = get_config_from_env_and_cli()
    uvicorn.run(app, host="0.0.0.0", port=cfg.port)
