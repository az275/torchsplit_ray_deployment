from starlette.requests import Request
from typing import Dict

from ray import serve
from PIL import Image

from transformers import CLIPProcessor
import io

import asyncio
import base64
import ray
from fastapi import FastAPI, UploadFile, File, Form
from ray import serve
from ray.serve.handle import DeploymentHandle
from pathlib import Path
import torch

from torch_split.runtime import SwitchboardRuntime

app = FastAPI()

def pad_list_before_stack(tensor_list: list[torch.Tensor], target: int = 32):
    B = len(tensor_list)

    if B == 0:
        raise ValueError("Cannot pad an empty list of tensors.")

    if B < target:
        # repeat the last tensor reference (cheap)
        last = tensor_list[-1]
        tensor_list = tensor_list + [last] * (target - B)

    return tensor_list, B


@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 0.25})
class ComponentA:
    def __init__(self):
        self.switchboard = SwitchboardRuntime(Path("/dev/shm/switchboard_bs_32.tspartd"), load_only=["A"])

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=1)
    async def __call__(self, batch: list[dict]) -> list[dict]:
        print([x["l_pixel_values_"].shape for x in batch])
        pixel_values, B = pad_list_before_stack([x["l_pixel_values_"] for x in batch])
        pixel_values = torch.stack(pixel_values).to("cuda")
        print("pixel_values shape:", pixel_values.shape)
        results = self.switchboard.call("A", pixel_values)
        results = results[:B]
        print(len(results))
        return [{"to_5": r.to("cpu")} for r in results]


@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 0.25})
class ComponentB:
    def __init__(self):
        self.switchboard = SwitchboardRuntime(Path("/dev/shm/switchboard_bs_32.tspartd"), load_only=["B"])

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=1)
    async def __call__(self, batch: list[dict]) -> list[dict]:
        print([x["l_input_ids_"].shape for x in batch])
        print([x["l_attention_mask_"].shape for x in batch])
        input_ids, B = pad_list_before_stack([x["l_input_ids_"] for x in batch])
        attn_mask, _ = pad_list_before_stack([x["l_attention_mask_"] for x in batch])

        input_ids = torch.stack(input_ids).to("cuda")
        attn_mask = torch.stack(attn_mask).to("cuda")
        print("input_ids shape:", input_ids.shape)
        print("attn_mask shape:", attn_mask.shape)

        masked_fill, text_embeds = self.switchboard.call("B", input_ids, attn_mask)
        text_embeds = text_embeds[:B]
        print(len(text_embeds))

        return [{"text_embeds_1": r.to("cpu")} for r in text_embeds]


@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 0.25})
class ComponentC:
    def __init__(self):
        self.switchboard = SwitchboardRuntime(Path("/dev/shm/switchboard_bs_32.tspartd"), load_only=["C"])

    @serve.batch(max_batch_size=1, batch_wait_timeout_s=1)
    async def __call__(self, batch: list[dict]) -> list[dict]:
        # print([x["to_5"].shape for x in batch])
        # print([x["text_embeds_1"].shape for x in batch])
        # to_5, B = pad_list_before_stack([x["to_5"] for x in batch])
        # to_5 = torch.stack([x["to_5"] for x in batch]).to("cuda")
        # # text_embeds_1 = torch.stack([x["text_embeds_1"] for x in batch]).to("cuda")
        # text_embeds_1, _ = pad_list_before_stack([x["text_embeds_1"] for x in batch])
        # text_embeds_1 = torch.stack(text_embeds_1).to("cuda")

        # print("to_5 shape:", to_5.shape)
        # print("text_embeds_1 shape:", text_embeds_1.shape)
        a = batch[0]["to_5"]
        b = batch[0]["text_embeds_1"]
        to_5 = a.to("cuda")
        text_embeds_1 = b.to("cuda")

        results = self.switchboard.call("C", text_embeds_1, to_5)
        results = results.diag.to("cpu")
        print(results[0].shape)

        return [{"output": r.to("cpu")} for r in results]


@serve.deployment(max_ongoing_requests=64)
class ClipPreprocessor:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=1, max_concurrent_batches=2)
    async def __call__(self, requests):
        images = [req["image"] for req in requests]
        texts = [req["text"] for req in requests]

        return self.processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=32,
            return_tensors="pt",
        )

        print("processor.infer", enc["pixel_values"].shape)
        print("processor.infer", enc["attention_mask"].shape)
        print("processor.infer", enc["input_ids"].shape)

        B = enc["input_ids"].shape[0]
        return [
            {
                "pixel_values": enc["pixel_values"][i],
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
            }
            for i in range(B)
        ]


@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 0.25})
class Pipeline:
    def __init__(
        self,
        preprocessor: DeploymentHandle,
        A_node: DeploymentHandle,
        B_node: DeploymentHandle,
        C_node: DeploymentHandle,
    ):
        self.pre = preprocessor
        self.A = A_node
        self.B = B_node
        self.C = C_node

    @torch.inference_mode()
    async def __call__(self, request: Request):
        payload = await request.json()
        image = Image.open(io.BytesIO(base64.b64decode(payload["image"]))).convert("RGB")

        enc = await self.pre.remote({"image": image, "text": payload["text"]})

        a_ref = self.A.remote({"l_pixel_values_": enc["pixel_values"]})
        b_ref = self.B.remote({"l_input_ids_": enc["input_ids"], "l_attention_mask_": enc["attention_mask"]})
        A_out, B_out = await asyncio.gather(a_ref, b_ref)

        C_out = await self.C.remote({"to_5": A_out["to_5"], "text_embeds_1": B_out["text_embeds_1"]})

        return {"output": C_out["output"].tolist()}

app = Pipeline.bind(preprocessor=ClipPreprocessor.bind(),
                    A_node=ComponentA.bind(),
                    B_node=ComponentB.bind(),
                    C_node=ComponentC.bind())
