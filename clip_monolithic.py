import base64
import io

import ray
import torch
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from typing import Dict, Any, List
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

app = FastAPI()

def decode_image(b64_image: str) -> Image.Image:
    data = base64.b64decode(b64_image)
    return Image.open(io.BytesIO(data)).convert("RGB")


@serve.deployment(
    num_replicas=4,
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=256,
)
@serve.ingress(app)
class CLIPMonolithic:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.model.eval()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    @torch.inference_mode()
    async def _infer_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        images: List[Image.Image] = []
        texts: List[str] = []

        for p in payloads:
            images.append(decode_image(p["image"]))
            texts.append(p["text"])

        enc = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=32,
            return_tensors="pt",
        )
        enc = {k: v.to("cuda") for k, v in enc.items()}

        out = self.model(**enc)
        img = out.image_embeds.detach().cpu()
        txt = out.text_embeds.detach().cpu()

        return [
            {"image_embed": img[i].tolist(), "text_embed": txt[i].tolist()}
            for i in range(len(payloads))
        ]

    @app.post("/")
    async def handle(self, request: Request):
        payload = await request.json()
        return await self._infer_batch(payload)


app = CLIPMonolithic.bind()
