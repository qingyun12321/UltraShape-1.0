import argparse
import base64
import io
import os
import queue
import shlex
import shutil
import subprocess
import threading
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

HUNYUAN_ENV = os.environ.get("HUNYUAN_ENV", "anta3d")
HUNYUAN_CUDA_VISIBLE_DEVICES = os.environ.get(
    "HUNYUAN_CUDA_VISIBLE_DEVICES", "4,5,6,7"
)
ULTRASHAPE_ENV = os.environ.get("ULTRASHAPE_ENV", "ultrashape")
ULTRASHAPE_CUDA_VISIBLE_DEVICES = os.environ.get(
    "ULTRASHAPE_CUDA_VISIBLE_DEVICES", "4,5,6,7"
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

JOB_LOCK = threading.Lock()
_CONDA_SH_CACHE = None
TASK_QUEUE_MAXSIZE = 1
TASK_QUEUE: "queue.Queue[Dict[str, object]]" = queue.Queue(maxsize=TASK_QUEUE_MAXSIZE)
TASKS_LOCK = threading.Lock()
TASKS: Dict[str, Dict[str, Optional[str]]] = {}
WORKER_THREAD: Optional[threading.Thread] = None

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


def _ensure_dirs() -> None:
    os.makedirs(HUNYUAN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(ULTRASHAPE_OUTPUT_DIR, exist_ok=True)


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


def _run_hunyuan(image_path: str, output_path: str) -> None:
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
    _run_in_conda(
        HUNYUAN_ENV,
        ["python", "-c", script],
        cwd=HUNYUAN_ROOT,
        extra_env={
            "HUNYUAN_ROOT": HUNYUAN_ROOT,
            "HY3D_MODEL_PATH": MODEL_PATH,
            "INPUT_IMAGE": image_path,
            "OUTPUT_GLB": output_path,
            "CUDA_VISIBLE_DEVICES": HUNYUAN_CUDA_VISIBLE_DEVICES,
        },
    )
    print(f"[hunyuan] coarse mesh saved: {output_path}")


def _run_ultrashape(image_path: str, mesh_path: str, output_dir: str) -> str:
    if not os.path.isdir(ULTRASHAPE_ROOT):
        raise RuntimeError(f"UltraShape root not found: {ULTRASHAPE_ROOT}")

    env = os.environ.copy()
    env.update(
        {
            "ULTRASHAPE_DISABLE_FLASH_ATTN": "1",
            "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF,
            "ULTRASHAPE_CUDA_VISIBLE_DEVICES": ULTRASHAPE_CUDA_VISIBLE_DEVICES,
            "ULTRASHAPE_OCTREE_RES": str(ULTRASHAPE_OCTREE_RES),
            "ULTRASHAPE_STEPS": str(ULTRASHAPE_STEPS),
            "ULTRASHAPE_OUTPUT_DIR": output_dir,
        }
    )

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
    request_id = uuid.uuid4().hex
    input_image_path = os.path.join(INPUT_DIR, f"{request_id}.png")
    coarse_mesh_path = os.path.join(HUNYUAN_OUTPUT_DIR, f"{request_id}.glb")

    print("[api] received request, saving input image")
    _save_image_bytes(image_bytes, input_image_path)

    try:
        with JOB_LOCK:
            _run_hunyuan(input_image_path, coarse_mesh_path)
            refined_path = _run_ultrashape(
                input_image_path, coarse_mesh_path, ULTRASHAPE_OUTPUT_DIR
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
    global WORKER_THREAD
    if WORKER_THREAD and WORKER_THREAD.is_alive():
        return
    WORKER_THREAD = threading.Thread(target=_worker_loop, daemon=True)
    WORKER_THREAD.start()


@app.on_event("startup")
def _startup() -> None:
    _ensure_dirs()
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
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
