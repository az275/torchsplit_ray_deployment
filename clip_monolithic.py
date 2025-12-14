import base64
import io

import ray
import torch
from ray import serve
from starlette.requests import Request
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def decode_image(b64_image: str) -> Image.Image:
    data = base64.b64decode(b64_image)
    return Image.open(io.BytesIO(data)).convert("RGB")


@serve.deployment(
    num_replicas=4,
    ray_actor_options={"num_gpus": 1},
)
class CLIPMonolithic:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.model.eval()

    @torch.inference_mode()
    async def __call__(self, request: Request):
        payload = await request.json()

        image_b64 = payload["image"]
        text = payload["text"]

        image = decode_image(image_b64)

        enc = self.processor(
            images=[image],
            text=[text],
            padding="max_length",
            max_length=32,
            return_tensors="pt",
        )

        enc = {k: v.to("cuda") for k, v in enc.items()}

        outputs = self.model(**enc)

        return {
            "image_embed": outputs.image_embeds[0].detach().cpu().tolist(),
            "text_embed": outputs.text_embeds[0].detach().cpu().tolist(),
        }


app = CLIPMonolithic.bind()
