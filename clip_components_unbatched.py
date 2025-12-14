import asyncio
import base64
import io
from pathlib import Path
from typing import Any, Dict, List

import ray
import torch
from ray import serve
from starlette.requests import Request
from PIL import Image
from transformers import CLIPProcessor

from torch_split.runtime import SwitchboardRuntime

def decode_images(b64_images):
    images = []
    for s in b64_images:
        data = base64.b64decode(s)
        images.append(Image.open(io.BytesIO(data)).convert("RGB"))
    return images

def chunk(x: torch.Tensor, n: int) -> List[torch.Tensor]:
    return [x[i:i+n] for i in range(0, x.shape[0], n)]


@serve.deployment(ray_actor_options={"num_gpus": 0})
class ClipPreprocessor:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.inference_mode()
    async def __call__(self, requests):
        enc = self.processor(
            images=requests["images"],
            text=requests["texts"],
            padding="max_length",
            max_length=32,
            return_tensors="pt",
        )
        return enc


@serve.deployment(ray_actor_options={"num_gpus": 1})
class ComponentA:
    def __init__(self):
        self.switchboard = SwitchboardRuntime(Path("/dev/shm/switchboard.tspartd"), load_only=["A"])

    @torch.inference_mode()
    async def __call__(self, items):
        pixel_values = items["l_pixel_values_"].to("cuda")

        out = self.switchboard.call("A", pixel_values)

        if isinstance(out, (list, tuple)):
            out = torch.stack(out, dim=0)

        return {"to_5": out.to("cpu")}


@serve.deployment(ray_actor_options={"num_gpus": 1})
class ComponentB:
    def __init__(self):
        self.switchboard = SwitchboardRuntime(Path("/dev/shm/switchboard.tspartd"), load_only=["B"])

    @torch.inference_mode()
    async def __call__(self, items):
        input_ids = items["l_input_ids_"].to("cuda")
        attention_mask = items["l_attention_mask_"].to("cuda")

        _, text_embeds = self.switchboard.call("B", input_ids, attention_mask)
        return {"text_embeds_1": text_embeds.to("cpu")}


@serve.deployment(ray_actor_options={"num_gpus": 1})
class ComponentC:
    def __init__(self):
        self.switchboard = SwitchboardRuntime(Path("/dev/shm/switchboard.tspartd"), load_only=["C"])

    @torch.inference_mode()
    async def __call__(self, items: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        to_5 = items["to_5"].to("cuda")
        text_embeds_1 = items["text_embeds_1"].to("cuda")

        out = self.switchboard.call("C", text_embeds_1, to_5)
        return {"output": out.to("cpu")}


@serve.deployment
class Pipeline:
    def __init__(self, pre, A, B, C, text_chunk_size: int = 32):
        self.pre = pre
        self.A = A
        self.B = B
        self.C = C
        self.text_chunk_size = text_chunk_size

    @torch.inference_mode()
    async def __call__(self, request: Request):
        payload = await request.json()
        images = decode_images(payload["images"])
        texts = payload["texts"]

        enc = await self.pre.remote({"images": images, "texts": texts})

        a_ref = self.A.remote({"l_pixel_values_": enc["pixel_values"]})

        input_chunks = chunk(enc["input_ids"], self.text_chunk_size)
        mask_chunks = chunk(enc["attention_mask"], self.text_chunk_size)
        b_refs = [
            self.B.remote({"l_input_ids_": ic, "l_attention_mask_": mc})
            for ic, mc in zip(input_chunks, mask_chunks)
        ]

        A_out, *B_outs = await asyncio.gather(a_ref, *b_refs)
        text_embeds_1 = torch.cat([bo["text_embeds_1"] for bo in B_outs], dim=0)

        C_out = await self.C.remote({"to_5": A_out["to_5"], "text_embeds_1": text_embeds_1})

        return {"output": C_out["output"].tolist()}


app = Pipeline.bind(
    pre=ClipPreprocessor.bind(),
    A=ComponentA.bind(),
    B=ComponentB.bind(),
    C=ComponentC.bind(),
)
