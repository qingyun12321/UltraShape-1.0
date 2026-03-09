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

from refine_presets import resolve_refine_options
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
LOAD_ON_STARTUP = os.environ.get("ULTRASHAPE_LOAD_ON_STARTUP", "0") == "1"
LAZY_REMBG = os.environ.get("ULTRASHAPE_LAZY_REMBG", "1") == "1"
STAGED_EXPORT = os.environ.get("ULTRASHAPE_STAGED_EXPORT", "1") == "1"
IDLE_OFFLOAD_SECS = float(os.environ.get("ULTRASHAPE_IDLE_OFFLOAD_SECS", "60"))
KEEP_ON_GPU_RAW = os.environ.get("ULTRASHAPE_KEEP_ON_GPU", "model,conditioner")
ULTRASHAPE_DINO_MODEL = os.environ.get("ULTRASHAPE_DINO_MODEL", "facebook/dinov2-large")
PREFETCH_DINO_ON_STARTUP = os.environ.get("ULTRASHAPE_PREFETCH_DINO_ON_STARTUP", "1") == "1"

PIPELINE: Optional[UltraShapePipeline] = None
PIPELINE_LOCK = threading.Lock()
LOADER: Optional[SharpEdgeSurfaceLoader] = None
REMBG: Optional[BackgroundRemover] = None
TOKEN_NUM: Optional[int] = None
VOXEL_RES: Optional[int] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDLE_OFFLOAD_TIMER: Optional[threading.Timer] = None
IDLE_OFFLOAD_LOCK = threading.Lock()

app = FastAPI(title="UltraShape Refine Service")


class RefineRequest(BaseModel):
    image_base64: str
    mesh_base64: str
    precision: Optional[str] = "standard"
    steps: Optional[int] = None
    octree_res: Optional[int] = None
    num_latents: Optional[int] = None
    chunk_size: Optional[int] = None
    seed: Optional[int] = 42
    remove_bg: Optional[bool] = False
    scale: Optional[float] = 0.99


class RefineResponse(BaseModel):
    status: str
    mesh_base64: str


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _decode_base64(data: str) -> bytes:
    if "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def _parse_keep_on_gpu(value: str) -> set:
    value = (value or "").strip().lower()
    if not value or value == "all":
        return {"model", "vae", "conditioner"}
    if value == "none":
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


KEEP_ON_GPU = _parse_keep_on_gpu(KEEP_ON_GPU_RAW)


def _cancel_idle_offload() -> None:
    global IDLE_OFFLOAD_TIMER
    if IDLE_OFFLOAD_SECS <= 0:
        return
    with IDLE_OFFLOAD_LOCK:
        if IDLE_OFFLOAD_TIMER is not None:
            IDLE_OFFLOAD_TIMER.cancel()
            IDLE_OFFLOAD_TIMER = None


def _offload_pipeline_components(keep_on_gpu: set) -> None:
    if PIPELINE is None or DEVICE.type != "cuda":
        return
    if keep_on_gpu == {"model", "vae", "conditioner"}:
        return
    if not keep_on_gpu:
        PIPELINE.to("cpu")
    else:
        components = {
            "model": PIPELINE.model,
            "vae": PIPELINE.vae,
            "conditioner": PIPELINE.conditioner,
        }
        for name, module in components.items():
            if name in keep_on_gpu:
                module.to(DEVICE)
            else:
                module.to("cpu")
    torch.cuda.empty_cache()


def _schedule_idle_offload() -> None:
    global IDLE_OFFLOAD_TIMER
    if IDLE_OFFLOAD_SECS <= 0:
        return

    def _do_offload() -> None:
        if PIPELINE is None or DEVICE.type != "cuda":
            return
        if not PIPELINE_LOCK.acquire(blocking=False):
            _schedule_idle_offload()
            return
        try:
            _offload_pipeline_components(KEEP_ON_GPU)
        finally:
            PIPELINE_LOCK.release()

    with IDLE_OFFLOAD_LOCK:
        if IDLE_OFFLOAD_TIMER is not None:
            IDLE_OFFLOAD_TIMER.cancel()
        IDLE_OFFLOAD_TIMER = threading.Timer(IDLE_OFFLOAD_SECS, _do_offload)
        IDLE_OFFLOAD_TIMER.daemon = True
        IDLE_OFFLOAD_TIMER.start()


def _offload_after_diffusion() -> None:
    if PIPELINE is None or DEVICE.type != "cuda":
        return
    PIPELINE.model.to("cpu")
    PIPELINE.conditioner.to("cpu")
    torch.cuda.empty_cache()


def _restore_after_export() -> None:
    if PIPELINE is None or DEVICE.type != "cuda":
        return
    if "model" in KEEP_ON_GPU:
        PIPELINE.model.to(DEVICE)
    if "conditioner" in KEEP_ON_GPU:
        PIPELINE.conditioner.to(DEVICE)


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
    if not LAZY_REMBG:
        REMBG = BackgroundRemover()
    _schedule_idle_offload()


def _prefetch_dino_model() -> None:
    if not ULTRASHAPE_DINO_MODEL or os.path.exists(ULTRASHAPE_DINO_MODEL):
        return

    cache_root = os.environ.get("ULTRASHAPE_CACHE_DIR", "").strip()
    local_dir = os.environ.get("ULTRASHAPE_DINO_LOCAL_DIR", "").strip()
    if not local_dir and cache_root:
        local_dir = os.path.join(cache_root, "pretrained", ULTRASHAPE_DINO_MODEL.replace("/", "--"))

    if local_dir and os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"[dino] using cached DINO config from {local_dir}")
        return

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(f"[dino] huggingface_hub unavailable, skip prefetch: {exc}")
        return

    kwargs = {"repo_id": ULTRASHAPE_DINO_MODEL}
    if local_dir:
        kwargs["local_dir"] = local_dir
        kwargs["local_dir_use_symlinks"] = False

    try:
        resolved = snapshot_download(**kwargs)
        print(f"[dino] prefetched DINO assets to {resolved}")
    except Exception as exc:
        print(f"[dino] prefetch failed: {exc}")


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
    if PREFETCH_DINO_ON_STARTUP:
        threading.Thread(target=_prefetch_dino_model, daemon=True, name="ultrashape-dino-prefetch").start()
    if LOAD_ON_STARTUP:
        _ensure_models()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/refine", response_model=RefineResponse)
def refine(req: RefineRequest):
    _ensure_models()
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

    mesh_path = _write_temp_glb(mesh_bytes)
    options = resolve_refine_options(_model_dump(req), fallback_num_latents=TOKEN_NUM)
    steps = int(options["steps"])
    octree_res = int(options["octree_res"])
    token_num = int(options["num_latents"])
    chunk_size = int(options["chunk_size"])
    seed = int(options["seed"])
    scale = float(options["scale"])
    remove_bg = bool(options["remove_bg"]) or image.mode != "RGBA"

    if remove_bg:
        global REMBG
        if REMBG is None:
            REMBG = BackgroundRemover()
        image = REMBG(image)

    _cancel_idle_offload()
    try:
        with PIPELINE_LOCK:
            PIPELINE.to(DEVICE)
            surface = LOADER(mesh_path, normalize_scale=scale).to(
                DEVICE, dtype=torch.float16
            )
            pc = surface[:, :, :3]
            _, voxel_idx = voxelize_from_point(pc, token_num, resolution=VOXEL_RES)
            del surface, pc

            generator = torch.Generator(DEVICE).manual_seed(seed)
            with torch.no_grad():
                if DEVICE.type == "cuda":
                    autocast_ctx = torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    )
                else:
                    autocast_ctx = contextlib.nullcontext()
                with autocast_ctx:
                    if STAGED_EXPORT and DEVICE.type == "cuda":
                        latents, _ = PIPELINE(
                            image=image,
                            voxel_cond=voxel_idx,
                            generator=generator,
                            box_v=1.0,
                            mc_level=0.0,
                            octree_resolution=octree_res,
                            num_inference_steps=steps,
                            num_chunks=chunk_size,
                            output_type="latent",
                        )
                        _offload_after_diffusion()
                        mesh = PIPELINE._export(
                            latents,
                            output_type="trimesh",
                            box_v=1.0,
                            mc_level=0.0,
                            num_chunks=chunk_size,
                            octree_resolution=octree_res,
                            mc_algo=None,
                            enable_pbar=True,
                        )
                        _restore_after_export()
                    else:
                        mesh, _ = PIPELINE(
                            image=image,
                            voxel_cond=voxel_idx,
                            generator=generator,
                            box_v=1.0,
                            mc_level=0.0,
                            octree_resolution=octree_res,
                            num_inference_steps=steps,
                            num_chunks=chunk_size,
                        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(mesh_path):
            os.remove(mesh_path)
        _schedule_idle_offload()

    mesh_base64 = _mesh_to_base64(mesh[0])
    return {"status": "success", "mesh_base64": mesh_base64}


if __name__ == "__main__":
    host = os.environ.get("ULTRASHAPE_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("ULTRASHAPE_SERVICE_PORT", "9085"))
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
