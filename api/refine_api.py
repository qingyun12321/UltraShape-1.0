import argparse
import base64
import io
import json
import os
import queue
import shlex
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import textwrap

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

API_ROOT = os.path.dirname(os.path.abspath(__file__))
ULTRASHAPE_ROOT = os.path.abspath(os.path.join(API_ROOT, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(ULTRASHAPE_ROOT, ".."))
HUNYUAN_ROOT = os.environ.get(
    "HUNYUAN_ROOT", os.path.join(WORKSPACE_ROOT, "Hunyuan3D-2.1")
)

MODEL_PATH = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
HUNYUAN_OUTPUT_DIR = os.environ.get(
    "HY3D_SAVE_DIR", os.path.join(API_ROOT, "hunyuan_outputs")
)

ULTRASHAPE_CKPT = os.environ.get(
    "ULTRASHAPE_CKPT", os.path.join(ULTRASHAPE_ROOT, "checkpoints", "ultrashape_v1.pt")
)
ULTRASHAPE_CONFIG = os.environ.get(
    "ULTRASHAPE_CONFIG",
    os.path.join(ULTRASHAPE_ROOT, "configs", "infer_dit_refine.yaml"),
)
ULTRASHAPE_OUTPUT_DIR = os.environ.get("ULTRASHAPE_OUTPUT_DIR", API_ROOT)
INPUT_DIR = os.path.join(API_ROOT, "inputs")
KEEP_INTERMEDIATE = os.environ.get("KEEP_INTERMEDIATE", "0") == "1"
HUNYUAN_CACHE_DIR = os.environ.get(
    "HUNYUAN_CACHE_DIR", os.path.join(HUNYUAN_ROOT, "cache")
)
ULTRASHAPE_CACHE_DIR = os.environ.get(
    "ULTRASHAPE_CACHE_DIR", os.path.join(ULTRASHAPE_ROOT, "cache")
)

HUNYUAN_ENV = os.environ.get("HUNYUAN_ENV", "anta3d")
HUNYUAN_CUDA_VISIBLE_DEVICES = os.environ.get(
    "HUNYUAN_CUDA_VISIBLE_DEVICES", "1"
)
ULTRASHAPE_ENV = os.environ.get("ULTRASHAPE_ENV", "ultrashape")
ULTRASHAPE_CUDA_VISIBLE_DEVICES = os.environ.get(
    "ULTRASHAPE_CUDA_VISIBLE_DEVICES", "0"
)
ULTRASHAPE_OCTREE_RES = os.environ.get("ULTRASHAPE_OCTREE_RES", "512")
ULTRASHAPE_STEPS = os.environ.get("ULTRASHAPE_STEPS", "30")
PYTORCH_CUDA_ALLOC_CONF = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)
ULTRASHAPE_OOM_RETRY = os.environ.get("ULTRASHAPE_OOM_RETRY", "1") == "1"
ULTRASHAPE_OOM_OCTREE_RES = os.environ.get("ULTRASHAPE_OOM_OCTREE_RES", "384")
ULTRASHAPE_OOM_STEPS = os.environ.get("ULTRASHAPE_OOM_STEPS", "20")
CONDA_EXE = os.environ.get("CONDA_EXE", "conda")
CONDA_SH = os.environ.get("CONDA_SH")
HOT_START_ENABLED = os.environ.get("HOT_START_ENABLED", "1") == "1"
HOT_START_HUNYUAN = os.environ.get("HOT_START_HUNYUAN", "1") == "1"
HOT_START_ULTRASHAPE = os.environ.get("HOT_START_ULTRASHAPE", "1") == "1"
HOT_START_REMBG = os.environ.get("HOT_START_REMBG", "1") == "1"
HUNYUAN_MODEL_SUBFOLDER = os.environ.get(
    "HUNYUAN_MODEL_SUBFOLDER", "hunyuan3d-dit-v2-1"
)
ULTRASHAPE_DINO_MODEL = os.environ.get(
    "ULTRASHAPE_DINO_MODEL", "facebook/dinov2-large"
)
USE_REMOTE_SERVICES = os.environ.get("USE_REMOTE_SERVICES", "1") == "1"
HUNYUAN_SERVICE_URL = os.environ.get(
    "HUNYUAN_SERVICE_URL", "http://127.0.0.1:9084"
)
ULTRASHAPE_SERVICE_URL = os.environ.get(
    "ULTRASHAPE_SERVICE_URL", "http://127.0.0.1:9085"
)
HUNYUAN_SERVICE_URLS = os.environ.get("HUNYUAN_SERVICE_URLS", HUNYUAN_SERVICE_URL)
ULTRASHAPE_SERVICE_URLS = os.environ.get(
    "ULTRASHAPE_SERVICE_URLS", ULTRASHAPE_SERVICE_URL
)
HUNYUAN_CUDA_VISIBLE_DEVICES_LIST = os.environ.get(
    "HUNYUAN_CUDA_VISIBLE_DEVICES_LIST", HUNYUAN_CUDA_VISIBLE_DEVICES
)
ULTRASHAPE_CUDA_VISIBLE_DEVICES_LIST = os.environ.get(
    "ULTRASHAPE_CUDA_VISIBLE_DEVICES_LIST", ULTRASHAPE_CUDA_VISIBLE_DEVICES
)
SERVICE_TIMEOUT = float(os.environ.get("SERVICE_TIMEOUT", "1200"))

_CONDA_SH_CACHE = None
TASK_QUEUE_MAXSIZE = int(os.environ.get("TASK_QUEUE_MAXSIZE", "8"))
TASK_QUEUE: "queue.Queue[Dict[str, object]]" = queue.Queue(maxsize=TASK_QUEUE_MAXSIZE)
TASKS_LOCK = threading.Lock()
TASKS: Dict[str, Dict[str, Optional[str]]] = {}
WORKER_THREADS: List[threading.Thread] = []
SLOT_QUEUE: "queue.Queue[Dict[str, str]]" = queue.Queue()
SLOTS_LOCK = threading.Lock()
SLOTS: List[Dict[str, str]] = []

app = FastAPI(title="UltraShape Refine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    image_base64: str
    precision: Optional[str] = "standard"


class AsyncInput(BaseModel):
    image_base64: str


class AsyncParameters(BaseModel):
    precision: Optional[str] = "standard"


class AsyncGenerateRequest(BaseModel):
    model: Optional[str] = "ultrashape-refine"
    input: AsyncInput
    parameters: Optional[AsyncParameters] = None


def _build_cache_env(cache_root: str) -> Dict[str, str]:
    cache_root = os.path.abspath(cache_root)
    hf_home = os.path.join(cache_root, "huggingface")
    return {
        "XDG_CACHE_HOME": cache_root,
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": os.path.join(hf_home, "hub"),
        "TRANSFORMERS_CACHE": os.path.join(hf_home, "transformers"),
        "DIFFUSERS_CACHE": os.path.join(hf_home, "diffusers"),
        "TORCH_HOME": os.path.join(cache_root, "torch"),
        "HY3DGEN_MODELS": cache_root,
        "U2NET_HOME": os.path.join(cache_root, "u2net"),
    }


def _ensure_cache_dirs(cache_root: str) -> None:
    cache_root = os.path.abspath(cache_root)
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(os.path.join(cache_root, "huggingface"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "torch"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "u2net"), exist_ok=True)


def _split_slots(value: str) -> List[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def _build_slots() -> List[Dict[str, str]]:
    if USE_REMOTE_SERVICES:
        hunyuan_urls = _split_slots(HUNYUAN_SERVICE_URLS)
        ultrashape_urls = _split_slots(ULTRASHAPE_SERVICE_URLS)
        if len(hunyuan_urls) != len(ultrashape_urls):
            raise RuntimeError(
                "HUNYUAN_SERVICE_URLS and ULTRASHAPE_SERVICE_URLS length mismatch"
            )
        slots = []
        for idx, (hunyuan_url, ultrashape_url) in enumerate(
            zip(hunyuan_urls, ultrashape_urls)
        ):
            slots.append(
                {
                    "slot_id": str(idx),
                    "hunyuan_url": hunyuan_url,
                    "ultrashape_url": ultrashape_url,
                }
            )
        return slots

    hunyuan_devices = _split_slots(HUNYUAN_CUDA_VISIBLE_DEVICES_LIST)
    ultrashape_devices = _split_slots(ULTRASHAPE_CUDA_VISIBLE_DEVICES_LIST)
    if len(hunyuan_devices) != len(ultrashape_devices):
        raise RuntimeError(
            "HUNYUAN_CUDA_VISIBLE_DEVICES_LIST and "
            "ULTRASHAPE_CUDA_VISIBLE_DEVICES_LIST length mismatch"
        )
    slots = []
    for idx, (hunyuan_device, ultrashape_device) in enumerate(
        zip(hunyuan_devices, ultrashape_devices)
    ):
        slots.append(
            {
                "slot_id": str(idx),
                "hunyuan_device": hunyuan_device,
                "ultrashape_device": ultrashape_device,
            }
        )
    return slots


def _init_slots() -> None:
    global SLOTS
    with SLOTS_LOCK:
        if SLOTS:
            return
        SLOTS = _build_slots()
        if not SLOTS:
            raise RuntimeError("No slots configured")
        for slot in SLOTS:
            SLOT_QUEUE.put(slot)
        slot_ids = ",".join(slot["slot_id"] for slot in SLOTS)
        print(f"[scheduler] slots ready: {slot_ids}")


def _acquire_slot() -> Dict[str, str]:
    slot = SLOT_QUEUE.get()
    print(f"[scheduler] acquired slot {slot['slot_id']}")
    return slot


def _release_slot(slot: Dict[str, str]) -> None:
    SLOT_QUEUE.put(slot)
    print(f"[scheduler] released slot {slot['slot_id']}")


def _post_json(url: str, payload: Dict[str, object], timeout: float) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if response.status >= 400:
                raise RuntimeError(f"{url} failed: {response.status} {body}")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RuntimeError(f"{url} failed: {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} unavailable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{url} returned invalid JSON: {exc}") from exc


def _call_hunyuan_service(image_bytes: bytes, base_url: str) -> bytes:
    payload = {"image_base64": base64.b64encode(image_bytes).decode("utf-8")}
    response = _post_json(
        f"{base_url}/generate", payload, timeout=SERVICE_TIMEOUT
    )
    mesh_base64 = response.get("mesh_base64")
    if not mesh_base64:
        raise RuntimeError("Hunyuan service returned empty mesh")
    return base64.b64decode(mesh_base64)


def _call_ultrashape_service(
    image_bytes: bytes, mesh_bytes: bytes, base_url: str
) -> bytes:
    payload = {
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "mesh_base64": base64.b64encode(mesh_bytes).decode("utf-8"),
        "steps": int(ULTRASHAPE_STEPS),
        "octree_res": int(ULTRASHAPE_OCTREE_RES),
    }
    response = _post_json(
        f"{base_url}/refine", payload, timeout=SERVICE_TIMEOUT
    )
    mesh_base64 = response.get("mesh_base64")
    if not mesh_base64:
        raise RuntimeError("UltraShape service returned empty mesh")
    return base64.b64decode(mesh_base64)


def _prefetch_hunyuan_models() -> None:
    if not os.path.isdir(HUNYUAN_ROOT):
        print(f"[hot-start] Hunyuan root not found: {HUNYUAN_ROOT}")
        return

    script = textwrap.dedent(
        """
        import os

        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(f"huggingface_hub unavailable: {exc}")

        model_path = os.environ["HY3D_MODEL_PATH"]
        subfolder = os.environ.get("HUNYUAN_MODEL_SUBFOLDER", "hunyuan3d-dit-v2-1")
        base_dir = os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")
        model_fld = os.path.expanduser(os.path.join(base_dir, model_path))

        snapshot_download(
            repo_id=model_path,
            allow_patterns=[f"{subfolder}/*"],
            local_dir=model_fld,
        )
        print(f"[hot-start] hunyuan cached at {model_fld}")
        """
    ).strip()

    extra_env = _build_cache_env(HUNYUAN_CACHE_DIR)
    extra_env.update(
        {
            "HY3D_MODEL_PATH": MODEL_PATH,
            "HUNYUAN_MODEL_SUBFOLDER": HUNYUAN_MODEL_SUBFOLDER,
        }
    )
    try:
        _run_in_conda(
            HUNYUAN_ENV,
            ["python", "-c", script],
            cwd=HUNYUAN_ROOT,
            extra_env=extra_env,
        )
    except Exception as exc:
        print(f"[hot-start] hunyuan prefetch failed: {exc}")


def _prefetch_ultrashape_models() -> None:
    if not os.path.isdir(ULTRASHAPE_ROOT):
        print(f"[hot-start] UltraShape root not found: {ULTRASHAPE_ROOT}")
        return

    script = textwrap.dedent(
        """
        import os

        dino_model = os.environ.get("ULTRASHAPE_DINO_MODEL", "facebook/dinov2-large")
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(f"huggingface_hub unavailable: {exc}")

        snapshot_download(repo_id=dino_model)
        print(f"[hot-start] dino cached: {dino_model}")

        if os.environ.get("HOT_START_REMBG", "1") == "1":
            try:
                from rembg import new_session
                new_session()
                print("[hot-start] rembg session ready")
            except Exception as exc:
                print(f"[hot-start] rembg warmup failed: {exc}")
        """
    ).strip()

    extra_env = _build_cache_env(ULTRASHAPE_CACHE_DIR)
    extra_env.update(
        {
            "ULTRASHAPE_DINO_MODEL": ULTRASHAPE_DINO_MODEL,
            "HOT_START_REMBG": "1" if HOT_START_REMBG else "0",
        }
    )
    try:
        _run_in_conda(
            ULTRASHAPE_ENV,
            ["python", "-c", script],
            cwd=ULTRASHAPE_ROOT,
            extra_env=extra_env,
        )
    except Exception as exc:
        print(f"[hot-start] ultrashape prefetch failed: {exc}")


def _hot_start() -> None:
    if not HOT_START_ENABLED or USE_REMOTE_SERVICES:
        return
    print("[hot-start] warming model caches")
    if HOT_START_HUNYUAN:
        _prefetch_hunyuan_models()
    if HOT_START_ULTRASHAPE:
        _prefetch_ultrashape_models()


def _ensure_dirs() -> None:
    os.makedirs(HUNYUAN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(ULTRASHAPE_OUTPUT_DIR, exist_ok=True)
    _ensure_cache_dirs(HUNYUAN_CACHE_DIR)
    _ensure_cache_dirs(ULTRASHAPE_CACHE_DIR)


def _resolve_conda_sh() -> str:
    global _CONDA_SH_CACHE
    if _CONDA_SH_CACHE:
        return _CONDA_SH_CACHE

    if CONDA_SH:
        if not os.path.exists(CONDA_SH):
            raise RuntimeError(f"CONDA_SH not found: {CONDA_SH}")
        _CONDA_SH_CACHE = CONDA_SH
        return _CONDA_SH_CACHE

    conda_base = os.environ.get("CONDA_BASE")
    if not conda_base:
        result = subprocess.run(
            [CONDA_EXE, "info", "--base"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Unable to locate conda base. Set CONDA_BASE or CONDA_SH."
            )
        conda_base = result.stdout.strip()

    conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
    if not os.path.exists(conda_sh):
        raise RuntimeError(f"conda.sh not found at {conda_sh}")

    _CONDA_SH_CACHE = conda_sh
    return _CONDA_SH_CACHE


def _run_in_conda(env_name: str, args: List[str], cwd: str, extra_env: Dict) -> None:
    conda_sh = _resolve_conda_sh()
    command = " ".join(shlex.quote(arg) for arg in args)
    bash_cmd = (
        f"source {shlex.quote(conda_sh)} && "
        f"conda activate {shlex.quote(env_name)} && {command}"
    )

    env = os.environ.copy()
    env.update(extra_env or {})

    print(f"[runner] env={env_name} cwd={cwd}")
    print(f"[runner] cmd={command}")
    result = subprocess.run(
        ["bash", "-lc", bash_cmd],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "Command failed"
        raise RuntimeError(details)
    if result.stdout.strip():
        print(result.stdout.strip())


def _run_hunyuan(image_path: str, output_path: str, cuda_devices: str) -> None:
    script = textwrap.dedent(
        """
        import os
        import sys
        from PIL import Image

        repo = os.environ["HUNYUAN_ROOT"]
        sys.path.insert(0, os.path.join(repo, "hy3dshape"))

        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        model_path = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
        image_path = os.environ["INPUT_IMAGE"]
        output_path = os.environ["OUTPUT_GLB"]

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        image = Image.open(image_path)
        mesh = pipeline(image=image)[0]
        mesh.export(output_path)
        """
    ).strip()

    if not os.path.isdir(HUNYUAN_ROOT):
        raise RuntimeError(f"Hunyuan3D root not found: {HUNYUAN_ROOT}")

    print("[hunyuan] generating coarse mesh")
    extra_env = {
        "HUNYUAN_ROOT": HUNYUAN_ROOT,
        "HY3D_MODEL_PATH": MODEL_PATH,
        "INPUT_IMAGE": image_path,
        "OUTPUT_GLB": output_path,
        "CUDA_VISIBLE_DEVICES": cuda_devices,
    }
    extra_env.update(_build_cache_env(HUNYUAN_CACHE_DIR))
    _run_in_conda(
        HUNYUAN_ENV,
        ["python", "-c", script],
        cwd=HUNYUAN_ROOT,
        extra_env=extra_env,
    )
    print(f"[hunyuan] coarse mesh saved: {output_path}")


def _run_ultrashape(
    image_path: str, mesh_path: str, output_dir: str, cuda_devices: str
) -> str:
    if not os.path.isdir(ULTRASHAPE_ROOT):
        raise RuntimeError(f"UltraShape root not found: {ULTRASHAPE_ROOT}")

    env = os.environ.copy()
    env.update(
        {
            "ULTRASHAPE_DISABLE_FLASH_ATTN": "1",
            "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF,
            "ULTRASHAPE_CUDA_VISIBLE_DEVICES": cuda_devices,
            "ULTRASHAPE_OCTREE_RES": str(ULTRASHAPE_OCTREE_RES),
            "ULTRASHAPE_STEPS": str(ULTRASHAPE_STEPS),
            "ULTRASHAPE_OUTPUT_DIR": output_dir,
        }
    )
    env.update(_build_cache_env(ULTRASHAPE_CACHE_DIR))

    print(
        "[ultrashape] ULTRASHAPE_CUDA_VISIBLE_DEVICES="
        f"{env['ULTRASHAPE_CUDA_VISIBLE_DEVICES']}"
    )
    print(f"[ultrashape] PYTORCH_CUDA_ALLOC_CONF={PYTORCH_CUDA_ALLOC_CONF}")
    print(f"[ultrashape] ULTRASHAPE_OCTREE_RES={ULTRASHAPE_OCTREE_RES}")
    print(f"[ultrashape] ULTRASHAPE_STEPS={ULTRASHAPE_STEPS}")

    script_path = os.path.join(ULTRASHAPE_ROOT, "scripts", "run.sh")
    args = ["bash", script_path, image_path, mesh_path]

    def _run_once(run_env: Dict, label: str) -> subprocess.CompletedProcess:
        print(f"[ultrashape] refining mesh via run.sh ({label})")
        result = subprocess.run(
            args,
            cwd=ULTRASHAPE_ROOT,
            env=run_env,
            capture_output=True,
            text=True,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        return result

    result = _run_once(env, "primary")
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "UltraShape refine failed"
        if ULTRASHAPE_OOM_RETRY and "out of memory" in details.lower():
            retry_env = env.copy()
            retry_env["ULTRASHAPE_OCTREE_RES"] = str(ULTRASHAPE_OOM_OCTREE_RES)
            retry_env["ULTRASHAPE_STEPS"] = str(ULTRASHAPE_OOM_STEPS)
            print(
                "[ultrashape] OOM detected, retrying with "
                f"octree_res={retry_env['ULTRASHAPE_OCTREE_RES']} "
                f"steps={retry_env['ULTRASHAPE_STEPS']}"
            )
            result = _run_once(retry_env, "oom-retry")
            if result.returncode != 0:
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                details = stderr or stdout or "UltraShape refine failed"
                raise RuntimeError(details)
        else:
            raise RuntimeError(details)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    refined_path = os.path.join(output_dir, f"{base_name}_refined.glb")
    if not os.path.exists(refined_path):
        raise RuntimeError("UltraShape output missing")

    return refined_path


def _save_image_bytes(image_bytes: bytes, output_path: str) -> None:
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
        pil_image.save(output_path, format="PNG")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc


def _load_image_from_base64(data: str) -> bytes:
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc


def _generate_refined_glb(image_bytes: bytes) -> str:
    _ensure_dirs()
    _init_slots()
    slot = _acquire_slot()
    request_id = uuid.uuid4().hex
    input_image_path = os.path.join(INPUT_DIR, f"{request_id}.png")
    coarse_mesh_path = os.path.join(HUNYUAN_OUTPUT_DIR, f"{request_id}.glb")
    refined_path = ""

    print("[api] received request, saving input image")
    _save_image_bytes(image_bytes, input_image_path)

    try:
        try:
            if USE_REMOTE_SERVICES:
                coarse_mesh_bytes = _call_hunyuan_service(
                    image_bytes, slot["hunyuan_url"]
                )
                with open(coarse_mesh_path, "wb") as coarse_file:
                    coarse_file.write(coarse_mesh_bytes)
                refined_mesh_bytes = _call_ultrashape_service(
                    image_bytes, coarse_mesh_bytes, slot["ultrashape_url"]
                )
                base_name = os.path.splitext(os.path.basename(input_image_path))[0]
                refined_path = os.path.join(
                    ULTRASHAPE_OUTPUT_DIR, f"{base_name}_refined.glb"
                )
                with open(refined_path, "wb") as refined_file:
                    refined_file.write(refined_mesh_bytes)
            else:
                _run_hunyuan(
                    input_image_path, coarse_mesh_path, slot["hunyuan_device"]
                )
                refined_path = _run_ultrashape(
                    input_image_path,
                    coarse_mesh_path,
                    ULTRASHAPE_OUTPUT_DIR,
                    slot["ultrashape_device"],
                )
            backup_path = os.path.join(API_ROOT, "output.glb")
            shutil.copyfile(refined_path, backup_path)
            print(f"[api] backup saved: {backup_path}")
        finally:
            if not KEEP_INTERMEDIATE:
                if os.path.exists(input_image_path):
                    os.remove(input_image_path)
                if os.path.exists(coarse_mesh_path):
                    os.remove(coarse_mesh_path)
    finally:
        _release_slot(slot)

    return refined_path


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _enqueue_task(image_bytes: bytes) -> str:
    task_id = uuid.uuid4().hex
    with TASKS_LOCK:
        TASKS[task_id] = {
            "task_id": task_id,
            "task_status": "PENDING",
            "submit_time": _now_str(),
            "end_time": None,
            "model_url": "",
            "result_url": f"/api/v1/tasks/{task_id}/result",
            "error_message": "",
        }
    TASK_QUEUE.put({"task_id": task_id, "image_bytes": image_bytes})
    return task_id


def _worker_loop() -> None:
    while True:
        task = TASK_QUEUE.get()
        task_id = task["task_id"]
        image_bytes = task["image_bytes"]
        with TASKS_LOCK:
            if task_id in TASKS:
                TASKS[task_id]["task_status"] = "RUNNING"
        try:
            refined_path = _generate_refined_glb(image_bytes)
        except Exception as exc:
            with TASKS_LOCK:
                if task_id in TASKS:
                    TASKS[task_id]["task_status"] = "FAILED"
                    TASKS[task_id]["end_time"] = _now_str()
                    TASKS[task_id]["model_url"] = ""
                    TASKS[task_id]["error_message"] = str(exc)
        else:
            with TASKS_LOCK:
                if task_id in TASKS:
                    TASKS[task_id]["task_status"] = "SUCCEEDED"
                    TASKS[task_id]["end_time"] = _now_str()
                    TASKS[task_id]["model_url"] = refined_path
                    TASKS[task_id]["error_message"] = ""
        finally:
            TASK_QUEUE.task_done()


def _start_worker() -> None:
    global WORKER_THREADS
    if any(thread.is_alive() for thread in WORKER_THREADS):
        return
    WORKER_THREADS = []
    for idx in range(len(SLOTS)):
        thread = threading.Thread(
            target=_worker_loop,
            daemon=True,
            name=f"worker-{idx}",
        )
        thread.start()
        WORKER_THREADS.append(thread)


@app.on_event("startup")
def _startup() -> None:
    _ensure_dirs()
    _hot_start()
    _init_slots()
    _start_worker()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/generate")
async def generate(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    try:
        image_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc

    try:
        refined_path = _generate_refined_glb(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        refined_path,
        filename=os.path.basename(refined_path),
        media_type="model/gltf-binary",
    )


@app.post("/generate_3d")
async def generate_3d(req: GenerateRequest):
    image_bytes = _load_image_from_base64(req.image_base64)

    try:
        refined_path = _generate_refined_glb(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with open(refined_path, "rb") as f:
        model_base64 = base64.b64encode(f.read()).decode("utf-8")

    return {"status": "success", "model_data": model_base64, "format": "glb"}


@app.post("/api/v1/services/aigc/3d-refine/generation")
async def generate_async(req: AsyncGenerateRequest):
    image_bytes = _load_image_from_base64(req.input.image_base64)

    if TASK_QUEUE.full():
        raise HTTPException(status_code=429, detail="Queue is full, try again later")

    task_id = _enqueue_task(image_bytes)
    return {
        "status_code": 200,
        "request_id": uuid.uuid4().hex,
        "code": "",
        "message": "",
        "output": {
            "task_id": task_id,
            "task_status": "PENDING",
            "model_url": "",
            "result_url": f"/api/v1/tasks/{task_id}/result",
        },
        "usage": None,
    }


@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str):
    with TASKS_LOCK:
        task = TASKS.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "status_code": 200,
        "request_id": uuid.uuid4().hex,
        "code": None,
        "message": "",
        "output": {
            "task_id": task_id,
            "task_status": task["task_status"],
            "model_url": task["model_url"],
            "result_url": task["result_url"],
            "submit_time": task["submit_time"],
            "end_time": task["end_time"],
            "error_message": task["error_message"],
        },
        "usage": None,
    }


@app.get("/api/v1/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    with TASKS_LOCK:
        task = TASKS.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["task_status"] != "SUCCEEDED":
        raise HTTPException(status_code=409, detail="Task not ready")

    model_path = task["model_url"]
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(
        model_path,
        filename=os.path.basename(model_path),
        media_type="model/gltf-binary",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10083)
    args = parser.parse_args()

    _ensure_dirs()
    config = uvicorn.Config(app=app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
