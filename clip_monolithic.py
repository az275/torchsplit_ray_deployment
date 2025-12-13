import base64
import io
import torch
from ray import serve
from starlette.requests import Request
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List

def decode_images(b64_images):
    images = []
    for s in b64_images:
        data = base64.b64decode(s)
        images.append(Image.open(io.BytesIO(data)).convert("RGB"))
    return images

@serve.deployment(
    num_replicas=4,
    ray_actor_options={"num_gpus": 1},
)
class CLIPMonolithic:
    def __init__(self):
        assert torch.cuda.device_count() >= 1
        torch.cuda.set_device(0)

        self.device = torch.device("cuda")

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    async def __call__(self, request: Request):
        payload = await request.json()
        images_b64 = payload["images"]
        texts = payload["texts"]

        images = decode_images(images_b64)

        enc = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=32,
            return_tensors="pt",
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)

        return {
            "image_embeds": outputs.image_embeds.cpu().tolist(),
            "text_embeds": outputs.text_embeds.cpu().tolist(),
        }

app = CLIPMonolithic.bind()
