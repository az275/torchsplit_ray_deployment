import base64
import io
import os
import time
import statistics
import requests
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

SERVE_URL = "http://127.0.0.1:8000/"

os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_datasets_cache"

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def food101_prompts(classes):
    return [f"a photo of {cls.replace('_', ' ')} food" for cls in classes]

dataset = load_dataset(
    "ethz/food101",
    split="validation",
)

N_SAMPLES = 4800
BATCH_SIZE = 16

dataset = dataset.select(range(N_SAMPLES))

class_names = dataset.features["label"].names
text_prompts = food101_prompts(class_names)

print(f"Running CLIP inference on {N_SAMPLES} samples (batch_size={BATCH_SIZE})")

correct = 0
total = 0

req_latencies_s = []
img_latencies_s = []
t0_all = time.perf_counter()
n_requests = 0

for start in tqdm(range(0, N_SAMPLES, BATCH_SIZE)):
    batch = dataset[start : start + BATCH_SIZE]

    images = [pil_to_base64(img) for img in batch["image"]]
    labels = batch["label"]

    payload = {
        "images": images,
        "texts": text_prompts,
    }

    t0 = time.perf_counter()
    resp = requests.post(SERVE_URL, json=payload, timeout=120)
    resp.raise_for_status()
    out = resp.json()
    t1 = time.perf_counter()

    dt = t1 - t0
    n_requests += 1
    req_latencies_s.append(dt)

    bsz = len(images)
    img_latencies_s.append(dt / max(bsz, 1))

    image_embeds = np.asarray(out["image_embeds"])
    text_embeds = np.asarray(out["text_embeds"])

    image_embeds /= np.linalg.norm(image_embeds, axis=1, keepdims=True)
    text_embeds /= np.linalg.norm(text_embeds, axis=1, keepdims=True)

    logits = image_embeds @ text_embeds.T
    preds = logits.argmax(axis=1)

    for p, gt in zip(preds, labels):
        if p == gt:
            correct += 1
        total += 1

t1_all = time.perf_counter()
elapsed_s = t1_all - t0_all

acc = correct / total
print(f"Top-1 accuracy on {total} samples: {acc:.4f}")

throughput_img_s = total / elapsed_s
throughput_req_s = n_requests / elapsed_s

def pct(values, p):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))

if req_latencies_s:
    avg_req_ms = statistics.mean(req_latencies_s) * 1e3
    avg_img_ms = statistics.mean(img_latencies_s) * 1e3

    print("\n--- Performance (client-side end-to-end) ---")
    print(f"Total wall time: {elapsed_s:.3f} s")
    print(f"Requests: {n_requests}, Images: {total}")
    print(f"Throughput: {throughput_img_s:.2f} images/s  |  {throughput_req_s:.2f} req/s")
    print(f"Avg latency: {avg_req_ms:.2f} ms/request  |  {avg_img_ms:.2f} ms/image (req_latency/batch)")

    print("Request latency percentiles (ms): "
          f"p50={pct(req_latencies_s, 50)*1e3:.2f}, "
          f"p90={pct(req_latencies_s, 90)*1e3:.2f}, "
          f"p99={pct(req_latencies_s, 99)*1e3:.2f}")
