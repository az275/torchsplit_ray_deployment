import base64
import csv
import io
import os
import time
import uuid
from datetime import datetime, timezone

import ray
import torch
from fastapi import FastAPI
from pathlib import Path
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

        # logging
        out_dir = Path("./data")
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            ctx = serve.get_replica_context()
            replica_tag = ctx.replica_tag
        except Exception:
            replica_tag = f"pid{os.getpid()}"

        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_", "#") else "_" for ch in replica_tag)
        self.csv_path = os.path.join(out_dir, f"clip_model_time_{safe_tag}.csv")

        self.csv_fields = [
            "request_id",
            "replica_tag",
            "model_runtime_ms",
        ]

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                w.writeheader()

        self.replica_tag = replica_tag


    def _append_row(self, row: Dict[str, Any]) -> None:
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.csv_fields)
            w.writerow(row)


    @serve.batch(max_batch_size=1, batch_wait_timeout_s=0.1)
    @torch.inference_mode()
    async def _infer_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        request_id = str(uuid.uuid4())
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

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = self.model(**enc)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        model_runtime_ms = (t1 - t0) * 1000.0
    
        self._append_row({
            "request_id": request_id,
            "replica_tag": self.replica_tag,
            "model_runtime_ms": model_runtime_ms,
        })

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
