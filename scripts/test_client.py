import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-qwen")

resp = client.images.generate(
    model="ovedrive/Qwen-Image-2512-4bit",
    prompt="A highly detailed cyberpunk alley in Auckland at night, rain, neon reflections, cinematic",
    size="1536x1024",
    extra_body={
        "response_format": "b64_json",
        "output_format": "png",
        "seed": 42,
        "num_inference_steps": 20,
        "true_cfg_scale": 4.0,
        "negative_prompt": "blurry, low quality, malformed text",
    },
)

img_b64 = resp.data[0].b64_json
Path("output.png").write_bytes(base64.b64decode(img_b64))
print("saved output.png")
