# Qwen Image OpenAI-Compatible Server

OpenAI-compatible local image generation server for:
- `GET /v1/models`
- `POST /v1/images/generations`
- `GET /health`

It wraps the Hugging Face model `ovedrive/Qwen-Image-2512-4bit` through Diffusers and exposes an OpenAI-style API for image generation.

## What is included
- FastAPI server with OpenAI-style image endpoint
- Lazy loading on first request
- Idle unload to free VRAM after inactivity
- Simple queueing through configurable concurrency
- Static URL serving for generated images
- Conda environment file
- systemd unit template
- nginx reverse proxy config
- test client using the OpenAI Python SDK

## 1) Create the conda environment

```bash
cd qwen-image-openai-server
conda env create -f environment.yml
conda activate qwenimg
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate qwenimg
```

## 2) Configure env vars

```bash
cp .env.example .env
nano .env
```

Suggested values for a 3090 24 GB:

```env
MODEL_ID=ovedrive/Qwen-Image-2512-4bit
API_KEY=local-qwen
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO
ENABLE_CPU_OFFLOAD=0
DEFAULT_STEPS=20
DEFAULT_CFG=4.0
DEFAULT_NEGATIVE=
MAX_N=4
MAX_CONCURRENCY=1
IDLE_UNLOAD_SECONDS=900
ENABLE_LAZY_LOADING=1
ALLOW_ORIGINS=*
GENERATED_DIR=generated
```

For lower VRAM behavior, set:

```env
ENABLE_CPU_OFFLOAD=1
```

## 3) Run the server

```bash
./scripts/start_server.sh
```

Or manually:

```bash
conda activate qwenimg
set -a && source .env && set +a
python -m uvicorn app.main:app --host "$HOST" --port "$PORT" --workers 1
```

## 4) Quick checks

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Models

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer local-qwen"
```

### Generate image using URL response

```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-qwen" \
  -d '{
    "model": "ovedrive/Qwen-Image-2512-4bit",
    "prompt": "A cinematic cyberpunk street in Auckland at night, wet pavement, neon reflections, ultra detailed",
    "size": "1536x1024",
    "response_format": "url",
    "output_format": "png",
    "n": 1,
    "seed": 42,
    "num_inference_steps": 20,
    "true_cfg_scale": 4.0,
    "negative_prompt": "blurry, distorted, low quality, malformed text"
  }'
```

### Generate image using b64_json response

```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-qwen" \
  -d '{
    "model": "ovedrive/Qwen-Image-2512-4bit",
    "prompt": "A futuristic coffee shop entrance with neon signage and cinematic lighting",
    "size": "1024x1024",
    "response_format": "b64_json",
    "output_format": "png",
    "n": 1
  }'
```

## 5) OpenAI Python client example

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-qwen",
)

resp = client.images.generate(
    model="ovedrive/Qwen-Image-2512-4bit",
    prompt="A highly detailed fantasy library inside a giant tree, warm light, volumetric rays",
    size="1024x1536",
    extra_body={
        "response_format": "b64_json",
        "output_format": "png",
        "seed": 123,
        "num_inference_steps": 20,
        "true_cfg_scale": 4.0,
        "negative_prompt": "blurry, low quality, deformed"
    },
)

img_b64 = resp.data[0].b64_json
Path("output.png").write_bytes(base64.b64decode(img_b64))
print("saved output.png")
```

## 6) Request parameters supported

Top-level request fields:
- `prompt`
- `model`
- `n`
- `size`
- `response_format` = `b64_json` or `url`
- `output_format` = `png`, `webp`, `jpeg`

Additional generation controls can be sent as extra JSON body fields:
- `seed`
- `negative_prompt`
- `num_inference_steps`
- `true_cfg_scale`

## 7) Size mapping

OpenAI sizes are mapped internally to Qwen-native sizes:
- `1024x1024` -> `1328x1328`
- `1024x1536` -> `928x1664`
- `1536x1024` -> `1664x928`

Native Qwen sizes are also accepted directly:
- `1328x1328`
- `1664x928`
- `928x1664`
- `1472x1104`
- `1104x1472`
- `1472x1140`
- `1140x1472`
- `1584x1056`
- `1056x1584`

## 8) Production deployment

### Install app files

```bash
sudo mkdir -p /opt/qwen-image-openai-server
sudo rsync -av ./ /opt/qwen-image-openai-server/
sudo chown -R $USER:$USER /opt/qwen-image-openai-server
```

### systemd

Copy the unit file:

```bash
sudo cp deploy/systemd/qwen-image-openai.service /etc/systemd/system/qwen-image-openai.service
```

Edit the unit if needed and replace the `User=` value manually with your Linux username instead of `%i`, or run a templated install of your own.

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable qwen-image-openai.service
sudo systemctl start qwen-image-openai.service
sudo systemctl status qwen-image-openai.service
```

A simpler alternative is to edit the unit file to:

```ini
User=YOUR_USERNAME
```

### nginx

```bash
sudo cp deploy/nginx/qwen-image-openai.conf /etc/nginx/sites-available/qwen-image-openai.conf
sudo ln -s /etc/nginx/sites-available/qwen-image-openai.conf /etc/nginx/sites-enabled/qwen-image-openai.conf
sudo nginx -t
sudo systemctl reload nginx
```

## 9) Tuning notes

For a single RTX 3090, start with:
- `MAX_CONCURRENCY=1`
- `ENABLE_CPU_OFFLOAD=0`
- `DEFAULT_STEPS=20`
- `IDLE_UNLOAD_SECONDS=900`

If you are tight on VRAM:
- set `ENABLE_CPU_OFFLOAD=1`
- keep `MAX_CONCURRENCY=1`
- prefer `n=1`

## 10) Known limitations

- This scaffold exposes only text-to-image via `/v1/images/generations`
- `/v1/images/edits` and `/v1/images/variations` return `501`
- It intentionally runs a single worker because one diffusion pipeline on one GPU does not benefit from multiple Uvicorn workers unless you split devices manually
