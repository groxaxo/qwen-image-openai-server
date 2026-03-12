import asyncio
import base64
import io
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image

APP_NAME = "Qwen Image OpenAI-Compatible Server"
APP_VERSION = "0.2.0"
MODEL_ID = os.getenv("MODEL_ID", "ovedrive/Qwen-Image-2512-4bit")
API_KEY = os.getenv("API_KEY", "local-qwen")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))
DEFAULT_CFG = float(os.getenv("DEFAULT_CFG", "4.0"))
DEFAULT_NEGATIVE = os.getenv("DEFAULT_NEGATIVE", " ")
MAX_N = int(os.getenv("MAX_N", "4"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
IDLE_UNLOAD_SECONDS = int(os.getenv("IDLE_UNLOAD_SECONDS", "900"))
ENABLE_LAZY_LOADING = os.getenv("ENABLE_LAZY_LOADING", "1") == "1"
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
GENERATED_DIR = Path(os.getenv("GENERATED_DIR", "generated"))

GENERATED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("qwen-image-server")


class ServerState:
    def __init__(self) -> None:
        self.pipe: Optional[DiffusionPipeline] = None
        self.pipe_lock = threading.Lock()
        self.last_used_ts = 0.0
        self.load_count = 0
        self.gen_count = 0
        self._stop_evt = threading.Event()
        self.unload_thread: Optional[threading.Thread] = None

    def start_watcher(self) -> None:
        if self.unload_thread is not None:
            return
        self.unload_thread = threading.Thread(target=self._watch_idle, daemon=True)
        self.unload_thread.start()

    def stop_watcher(self) -> None:
        self._stop_evt.set()
        if self.unload_thread and self.unload_thread.is_alive():
            self.unload_thread.join(timeout=2)

    def _watch_idle(self) -> None:
        while not self._stop_evt.is_set():
            try:
                time.sleep(5)
                if IDLE_UNLOAD_SECONDS <= 0:
                    continue
                if self.pipe is None:
                    continue
                idle_for = time.time() - self.last_used_ts
                if idle_for >= IDLE_UNLOAD_SECONDS:
                    logger.info("Idle timeout reached (%.1fs), unloading pipeline", idle_for)
                    self.unload_pipeline()
            except Exception as exc:  # pragma: no cover
                logger.exception("Idle watcher error: %s", exc)

    def get_device_and_dtype(self) -> tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            # Qwen examples use bfloat16 on CUDA.
            return "cuda", torch.bfloat16
        return "cpu", torch.float32

    def load_pipeline(self) -> DiffusionPipeline:
        if self.pipe is not None:
            self.last_used_ts = time.time()
            return self.pipe

        with self.pipe_lock:
            if self.pipe is not None:
                self.last_used_ts = time.time()
                return self.pipe

            device, dtype = self.get_device_and_dtype()
            logger.info("Loading pipeline %s on %s with dtype=%s", MODEL_ID, device, dtype)
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                use_safetensors=True,
            )

            if device == "cuda":
                if ENABLE_CPU_OFFLOAD:
                    pipe.enable_model_cpu_offload()
                    logger.info("Enabled model CPU offload")
                else:
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cpu")

            self.pipe = pipe
            self.last_used_ts = time.time()
            self.load_count += 1
            logger.info("Pipeline loaded successfully")
            return pipe

    def unload_pipeline(self) -> None:
        with self.pipe_lock:
            if self.pipe is None:
                return
            logger.info("Unloading pipeline and clearing caches")
            try:
                self.pipe = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception as exc:  # pragma: no cover
                logger.exception("Failed during unload: %s", exc)


STATE = ServerState()
REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    model: Optional[str] = None
    n: int = Field(default=1, ge=1, le=16)
    size: str = "1024x1024"
    response_format: Literal["b64_json", "url"] = "b64_json"
    output_format: Literal["png", "webp", "jpeg"] = "png"
    quality: Optional[str] = "high"
    user: Optional[str] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE.start_watcher()
    if not ENABLE_LAZY_LOADING:
        await asyncio.to_thread(STATE.load_pipeline)
    yield
    STATE.stop_watcher()
    STATE.unload_pipeline()


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.strip() for s in ALLOW_ORIGINS.split(",") if s.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")


def verify_api_key(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def map_size(size_str: str) -> tuple[int, int]:
    openai_to_qwen = {
        "1024x1024": (1328, 1328),
        "1024x1536": (928, 1664),
        "1536x1024": (1664, 928),
    }
    native = {
        "1328x1328": (1328, 1328),
        "1664x928": (1664, 928),
        "928x1664": (928, 1664),
        "1472x1104": (1472, 1104),
        "1104x1472": (1104, 1472),
        "1472x1140": (1472, 1140),
        "1140x1472": (1140, 1472),
        "1584x1056": (1584, 1056),
        "1056x1584": (1056, 1584),
    }
    if size_str in openai_to_qwen:
        return openai_to_qwen[size_str]
    if size_str in native:
        return native[size_str]
    raise HTTPException(
        status_code=400,
        detail=(
            "Unsupported size. Use one of: "
            "1024x1024, 1024x1536, 1536x1024, "
            "1328x1328, 1664x928, 928x1664, "
            "1472x1104, 1104x1472, 1472x1140, 1140x1472, "
            "1584x1056, 1056x1584"
        ),
    )


def pil_to_base64(img: Image.Image, fmt: str) -> str:
    buffer = io.BytesIO()
    pil_fmt = "JPEG" if fmt == "jpeg" else fmt.upper()
    save_kwargs = {}
    if pil_fmt == "JPEG":
        img = img.convert("RGB")
        save_kwargs["quality"] = 95
    img.save(buffer, format=pil_fmt, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_image(img: Image.Image, fmt: str) -> str:
    ext = "jpg" if fmt == "jpeg" else fmt
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = GENERATED_DIR / filename
    save_kwargs = {}
    if fmt == "jpeg":
        img = img.convert("RGB")
        save_kwargs["quality"] = 95
    img.save(path, format=("JPEG" if fmt == "jpeg" else fmt.upper()), **save_kwargs)
    return filename


@app.get("/health")
async def health():
    device, dtype = STATE.get_device_and_dtype()
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "model": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "loaded": STATE.pipe is not None,
        "cpu_offload": ENABLE_CPU_OFFLOAD,
        "lazy_loading": ENABLE_LAZY_LOADING,
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "max_concurrency": MAX_CONCURRENCY,
        "load_count": STATE.load_count,
        "gen_count": STATE.gen_count,
    }


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(default=None)):
    verify_api_key(authorization)
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": created,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/images/generations")
async def generate_image(
    body: ImageGenerationRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    verify_api_key(authorization)

    requested_model = body.model or MODEL_ID
    if requested_model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"This server exposes only one model: {MODEL_ID}")

    if body.n > MAX_N:
        raise HTTPException(status_code=400, detail=f"n must be <= {MAX_N}")

    width, height = map_size(body.size)

    extras = body.model_extra or {}
    negative_prompt = str(extras.get("negative_prompt", DEFAULT_NEGATIVE))
    num_inference_steps = int(extras.get("num_inference_steps", DEFAULT_STEPS))
    true_cfg_scale = float(extras.get("true_cfg_scale", DEFAULT_CFG))
    seed = extras.get("seed", None)

    async with REQUEST_SEMAPHORE:
        pipe = await asyncio.to_thread(STATE.load_pipeline)
        created = int(time.time())
        data = []

        for i in range(body.n):
            generator = None
            if seed is not None:
                gen_device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(int(seed) + i)

            result = await asyncio.to_thread(
                pipe,
                prompt=body.prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
            )

            image = result.images[0]
            STATE.gen_count += 1
            STATE.last_used_ts = time.time()

            if body.response_format == "b64_json":
                data.append({
                    "b64_json": pil_to_base64(image, body.output_format),
                    "revised_prompt": body.prompt,
                })
            else:
                filename = save_image(image, body.output_format)
                base = str(request.base_url).rstrip("/")
                data.append({
                    "url": f"{base}/generated/{filename}",
                    "revised_prompt": body.prompt,
                })

        return {"created": created, "data": data}


@app.post("/v1/images/edits")
async def edits_not_supported(authorization: Optional[str] = Header(default=None)):
    verify_api_key(authorization)
    raise HTTPException(status_code=501, detail="Only /v1/images/generations is supported for this text-to-image checkpoint.")


@app.post("/v1/images/variations")
async def variations_not_supported(authorization: Optional[str] = Header(default=None)):
    verify_api_key(authorization)
    raise HTTPException(status_code=501, detail="Only /v1/images/generations is supported for this text-to-image checkpoint.")
