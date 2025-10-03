from __future__ import annotations

from typing import List
import torch
from PIL import Image
from loguru import logger
import clip


class ClipValidator:
    """
    Thin wrapper around OpenAI CLIP for prompt-image similarity.
    """

    def __init__(
        self,
        device: torch.device,
        enabled: bool = True,
        model_name: str = "ViT-L/14",
        threshold: float = 0.285,
        sample_views: int = 3,
        use_bfloat16: bool = True,
    ):
        self.device = device
        self.enabled = enabled
        self.threshold = threshold
        self.sample_views = max(1, int(sample_views))
        self.use_bfloat16 = use_bfloat16

        if not self.enabled:
            logger.info("Validation disabled.")
            self.model = None
            self.preprocess = None
            return

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        logger.info(f"CLIP validator loaded: {model_name}")

    @torch.no_grad()
    def score(self, prompt: str, pil_images: List[Image.Image]) -> float:
        if not self.enabled or self.model is None:
            return 1.0

        imgs = pil_images[: self.sample_views]
        image_tensors = torch.stack([self.preprocess(im).to(self.device) for im in imgs], dim=0)
        text_tokens = clip.tokenize([prompt]).to(self.device)

        dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            image_features = self.model.encode_image(image_tensors)
            text_features = self.model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).squeeze(-1)
        return sims.mean().item()

    def passes(self, score: float) -> bool:
        return float(score) >= float(self.threshold)
