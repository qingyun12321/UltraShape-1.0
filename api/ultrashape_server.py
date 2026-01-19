import base64
import contextlib
import io
import os
import sys
import tempfile
import threading
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from omegaconf import OmegaConf
import uvicorn

API_ROOT = os.path.dirname(os.path.abspath(__file__))
ULTRASHAPE_ROOT = os.path.abspath(os.path.join(API_ROOT, ".."))
if ULTRASHAPE_ROOT not in sys.path:
    sys.path.insert(0, ULTRASHAPE_ROOT)

ULTRASHAPE_VISIBLE_DEVICES = os.environ.get("ULTRASHAPE_CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ULTRASHAPE_VISIBLE_DEVICES)

import torch

from ultrashape.rembg import BackgroundRemover
from ultrashape.utils.misc import instantiate_from_config
from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
from ultrashape.utils import voxelize_from_point
from ultrashape.pipelines import UltraShapePipeline

CKPT_PATH = os.environ.get(
    "ULTRASHAPE_CKPT", os.path.join(ULTRASHAPE_ROOT, "checkpoints", "ultrashape_v1.pt")
)
CONFIG_PATH = os.environ.get(
    "ULTRASHAPE_CONFIG", os.path.join(ULTRASHAPE_ROOT, "configs", "infer_dit_refine.yaml")
)
DEFAULT_STEPS = int(os.environ.get("ULTRASHAPE_STEPS", "30"))
DEFAULT_OCTREE_RES = int(os.environ.get("ULTRASHAPE_OCTREE_RES", "512"))

PIPELINE: Optional[UltraShapePipeline] = None
PIPELINE_LOCK = threading.Lock()
LOADER: Optional[SharpEdgeSurfaceLoader] = None
REMBG: Optional[BackgroundRemover] = None
TOKEN_NUM: Optional[int] = None
VOXEL_RES: Optional[int] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="UltraShape Refine Service")


class RefineRequest(BaseModel):
    image_base64: str
    mesh_base64: str
    steps: Optional[int] = None
    octree_res: Optional[int] = None
    seed: Optional[int] = 42
    remove_bg: Optional[bool] = False
    scale: Optional[float] = 0.99


class RefineResponse(BaseModel):
    status: str
    mesh_base64: str


def _decode_base64(data: str) -> bytes:
    if "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def _load_models() -> Tuple[UltraShapePipeline, int, int, SharpEdgeSurfaceLoader]:
    config = OmegaConf.load(CONFIG_PATH)

    vae = instantiate_from_config(config.model.params.vae_config)
    dit = instantiate_from_config(config.model.params.dit_cfg)
    conditioner = instantiate_from_config(config.model.params.conditioner_config)
    scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
    image_processor = instantiate_from_config(config.model.params.image_processor_cfg)

    weights = torch.load(CKPT_PATH, map_location="cpu")
    vae.load_state_dict(weights["vae"], strict=True)
    dit.load_state_dict(weights["dit"], strict=True)
    conditioner.load_state_dict(weights["conditioner"], strict=True)

    vae.eval().to(DEVICE)
    dit.eval().to(DEVICE)
    conditioner.eval().to(DEVICE)

    if hasattr(vae, "enable_flashvdm_decoder"):
        vae.enable_flashvdm_decoder()

    pipeline = UltraShapePipeline(
        vae=vae,
        model=dit,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor,
    )

    token_num = int(config.model.params.vae_config.params.num_latents)
    voxel_res = int(config.model.params.vae_config.params.voxel_query_res)
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=204800,
        num_uniform_points=204800,
    )
    return pipeline, token_num, voxel_res, loader


def _ensure_models() -> None:
    global PIPELINE, LOADER, REMBG, TOKEN_NUM, VOXEL_RES
    if PIPELINE is not None:
        return
    PIPELINE, TOKEN_NUM, VOXEL_RES, LOADER = _load_models()
    REMBG = BackgroundRemover()


def _write_temp_glb(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_file:
        tmp_file.write(data)
        return tmp_file.name


def _mesh_to_base64(mesh) -> str:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        mesh.export(tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.on_event("startup")
def _startup() -> None:
    _ensure_models()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/refine", response_model=RefineResponse)
def refine(req: RefineRequest):
    if PIPELINE is None or LOADER is None or TOKEN_NUM is None or VOXEL_RES is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        image_bytes = _decode_base64(req.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image base64: {exc}") from exc

    try:
        mesh_bytes = _decode_base64(req.mesh_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid mesh base64: {exc}") from exc

    remove_bg = bool(req.remove_bg) or image.mode != "RGBA"
    if remove_bg and REMBG is not None:
        image = REMBG(image)

    mesh_path = _write_temp_glb(mesh_bytes)
    steps = int(req.steps or DEFAULT_STEPS)
    octree_res = int(req.octree_res or DEFAULT_OCTREE_RES)

    try:
        with PIPELINE_LOCK:
            surface = LOADER(mesh_path, normalize_scale=req.scale).to(
                DEVICE, dtype=torch.float16
            )
            pc = surface[:, :, :3]
            _, voxel_idx = voxelize_from_point(pc, TOKEN_NUM, resolution=VOXEL_RES)

            generator = torch.Generator(DEVICE).manual_seed(int(req.seed or 42))
            with torch.no_grad():
                if DEVICE.type == "cuda":
                    autocast_ctx = torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    )
                else:
                    autocast_ctx = contextlib.nullcontext()
                with autocast_ctx:
                    mesh, _ = PIPELINE(
                        image=image,
                        voxel_cond=voxel_idx,
                        generator=generator,
                        box_v=1.0,
                        mc_level=0.0,
                        octree_resolution=octree_res,
                        num_inference_steps=steps,
                    )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(mesh_path):
            os.remove(mesh_path)

    mesh_base64 = _mesh_to_base64(mesh[0])
    return {"status": "success", "mesh_base64": mesh_base64}


if __name__ == "__main__":
    host = os.environ.get("ULTRASHAPE_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("ULTRASHAPE_SERVICE_PORT", "9085"))
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
