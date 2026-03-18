import argparse
import base64
import contextlib
import io
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional
import textwrap
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

API_ROOT = os.path.dirname(os.path.abspath(__file__))
ULTRASHAPE_ROOT = os.path.abspath(os.path.join(API_ROOT, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(ULTRASHAPE_ROOT, ".."))
HUNYUAN_ROOT = os.environ.get(
    "HUNYUAN_ROOT", os.path.join(WORKSPACE_ROOT, "Hunyuan3D-2.1")
)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from memory_orchestration import normalize_memory_action, run_remote_refinement_pipeline
from refine_presets import DEFAULT_PRECISION, resolve_refine_options
from runtime_diagnostics import format_process_exit_status, format_runtime_diagnostics

MODEL_PATH = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
ULTRASHAPE_CKPT = os.environ.get(
    "ULTRASHAPE_CKPT", os.path.join(ULTRASHAPE_ROOT, "checkpoints", "ultrashape_v1.pt")
)
ULTRASHAPE_CONFIG = os.environ.get(
    "ULTRASHAPE_CONFIG",
    os.path.join(ULTRASHAPE_ROOT, "configs", "infer_dit_refine.yaml"),
)
KEEP_INTERMEDIATE = os.environ.get("KEEP_INTERMEDIATE", "0") == "1"
HUNYUAN_CACHE_DIR = os.environ.get(
    "HUNYUAN_CACHE_DIR", os.path.join(HUNYUAN_ROOT, "cache")
)
ULTRASHAPE_CACHE_DIR = os.environ.get(
    "ULTRASHAPE_CACHE_DIR", os.path.join(ULTRASHAPE_ROOT, "cache")
)

HUNYUAN_ENV = os.environ.get("HUNYUAN_ENV", "anta3d")
HUNYUAN_CUDA_VISIBLE_DEVICES = os.environ.get(
    "HUNYUAN_CUDA_VISIBLE_DEVICES", "0"
)
ULTRASHAPE_ENV = os.environ.get("ULTRASHAPE_ENV", "ultrashape")
ULTRASHAPE_CUDA_VISIBLE_DEVICES = os.environ.get(
    "ULTRASHAPE_CUDA_VISIBLE_DEVICES", "0"
)
PYTORCH_CUDA_ALLOC_CONF = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)
ULTRASHAPE_OOM_RETRY = os.environ.get("ULTRASHAPE_OOM_RETRY", "1") == "1"
ULTRASHAPE_OOM_OCTREE_RES = os.environ.get("ULTRASHAPE_OOM_OCTREE_RES", "384")
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
MEMORY_ACTION_TIMEOUT = float(os.environ.get("MEMORY_ACTION_TIMEOUT", "120"))
AUTO_START_LOCAL_SERVICES = os.environ.get("AUTO_START_LOCAL_SERVICES", "0") == "1"
AUTO_START_HUNYUAN = os.environ.get("AUTO_START_HUNYUAN", "1") == "1"
AUTO_START_ULTRASHAPE = os.environ.get("AUTO_START_ULTRASHAPE", "1") == "1"
LOCAL_SERVICE_START_TIMEOUT = float(
    os.environ.get("LOCAL_SERVICE_START_TIMEOUT", "180")
)
HUNYUAN_SERVICE_PYTHON = os.environ.get(
    "HUNYUAN_SERVICE_PYTHON", os.environ.get("PYTHON_BIN", sys.executable)
)
ULTRASHAPE_SERVICE_PYTHON = os.environ.get(
    "ULTRASHAPE_SERVICE_PYTHON", os.environ.get("PYTHON_BIN", sys.executable)
)
OSS_BUCKET = os.environ.get("ULTRASHAPE_OSS_BUCKET", "kokokoni")
OSS_PREFIX = os.environ.get("ULTRASHAPE_OSS_PREFIX", "docker-input&output/ultrashape").strip("/")
OSS_SIGN_EXPIRES = os.environ.get("ULTRASHAPE_OSS_SIGN_EXPIRES", "24h")
TMP_ROOT = os.environ.get("ULTRASHAPE_TMP_ROOT", "/tmp/ultrashape-oss-workspace")
ULTRASHAPE_QUEUE_WORKERS = int(os.environ.get("ULTRASHAPE_QUEUE_WORKERS", "0"))
HUNYUAN_POST_ACTION = normalize_memory_action(
    os.environ.get("HUNYUAN_POST_ACTION"), default="unload"
)
ULTRASHAPE_POST_ACTION = normalize_memory_action(
    os.environ.get("ULTRASHAPE_POST_ACTION"), default="offload"
)

_CONDA_SH_CACHE = None
TASK_QUEUE_MAXSIZE = int(os.environ.get("TASK_QUEUE_MAXSIZE", "8"))
SLOT_QUEUE: "queue.Queue[Dict[str, str]]" = queue.Queue()
SLOTS_LOCK = threading.Lock()
SLOTS: List[Dict[str, str]] = []
MANAGED_SERVICE_LOCK = threading.Lock()
MANAGED_SERVICE_PROCS: List["ManagedServiceProcess"] = []
BOOTSTRAP_LOCK = threading.Lock()
BOOTSTRAP_THREAD: Optional[threading.Thread] = None

app = FastAPI(title="UltraShape Refine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RefineParameters(BaseModel):
    precision: Optional[str] = DEFAULT_PRECISION
    steps: Optional[int] = None
    octree_res: Optional[int] = None
    num_latents: Optional[int] = None
    chunk_size: Optional[int] = None
    seed: Optional[int] = 42
    remove_bg: Optional[bool] = False
    scale: Optional[float] = 0.99


class GenerateRequest(RefineParameters):
    image_base64: str


class AsyncInput(BaseModel):
    image_base64: str


class AsyncParameters(RefineParameters):
    pass


class AsyncGenerateRequest(BaseModel):
    model: Optional[str] = "ultrashape-refine"
    input: AsyncInput
    parameters: Optional[AsyncParameters] = None


@dataclass
class RequestRecord:
    status: str
    created_at: float = field(default_factory=time.time)
    error: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class ManagedServiceProcess:
    name: str
    process: subprocess.Popen


@dataclass
class BootstrapStatus:
    state: str = "starting"
    error: str = ""
    dependencies_ready: bool = False
    slots_ready: bool = False
    workers_ready: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


BOOTSTRAP_STATUS = BootstrapStatus()


class NodeBusyError(RuntimeError):
    """Raised when no execution slot is currently available."""


class ConcurrentTaskQueue:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: list[tuple[str, Any]] = []
        self._active_request_ids: set[str] = set()
        self._records: dict[str, RequestRecord] = {}
        self._workers: list[threading.Thread] = []
        self._running = False

    def start(
        self,
        handler: Any,
        *,
        worker_count: int = 1,
    ) -> None:
        with self._cond:
            if any(worker.is_alive() for worker in self._workers):
                return
            self._running = True
            self._workers = []
            total = max(1, int(worker_count or 1))
            for index in range(total):
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(handler,),
                    daemon=True,
                    name=f"ultrashape-run-queue-worker-{index}",
                )
                self._workers.append(worker)
                worker.start()

    def enqueue(
        self,
        payload: Any,
        *,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        rid = (request_id or "").strip() or str(uuid.uuid4())
        initial_meta = dict(metadata or {})

        with self._cond:
            record = self._records.get(rid)
            if record and record.status in {"pending", "processing"}:
                raise ValueError("request_id already exists in queue")

            self._records[rid] = RequestRecord(status="pending", result=initial_meta)
            self._pending.append((rid, payload))
            position = self._pending_position_unlocked(rid)
            self._cond.notify()
            return rid, position

    def get_queue_status(self, request_id: str | None = None) -> dict[str, Any]:
        with self._cond:
            active_request_ids = sorted(self._active_request_ids)
            payload: dict[str, Any] = {
                "processing": bool(active_request_ids),
                "processing_count": len(active_request_ids),
                "pending": len(self._pending),
                "current_request_id": active_request_ids[0] if active_request_ids else "",
                "processing_request_ids": active_request_ids,
            }
            if request_id is not None:
                rid = request_id.strip()
                payload["status"] = self._records[rid].status if rid in self._records else "unknown"
                payload["position"] = self._position_for_request_unlocked(rid)
            else:
                payload["status"] = (
                    "processing" if active_request_ids else ("pending" if self._pending else "idle")
                )
            return payload

    def get_request_status(self, request_id: str) -> dict[str, Any] | None:
        rid = request_id.strip()
        with self._cond:
            record = self._records.get(rid)
            if not record:
                return None
            result = dict(record.result)
            payload: dict[str, Any] = {
                "request_id": rid,
                "status": record.status,
                "error": record.error,
                "created_at": record.created_at,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "result": result,
            }
            for key, value in result.items():
                if key not in payload:
                    payload[key] = value
            return payload

    def is_idle(self) -> bool:
        with self._cond:
            return not self._active_request_ids and not self._pending

    def _worker_loop(self, handler: Any) -> None:
        while True:
            with self._cond:
                while self._running and not self._pending:
                    self._cond.wait()
                if not self._running:
                    return
                request_id, payload = self._pending.pop(0)
                self._active_request_ids.add(request_id)
                record = self._records[request_id]
                record.status = "processing"
                record.error = ""
                record.started_at = time.time()
                record.finished_at = None

            try:
                result = handler(request_id, payload) or {}
                with self._cond:
                    record = self._records[request_id]
                    record.status = "completed"
                    record.error = ""
                    record.finished_at = time.time()
                    if isinstance(result, dict):
                        merged = dict(record.result)
                        merged.update(result)
                        record.result = merged
            except Exception as exc:
                with self._cond:
                    record = self._records[request_id]
                    record.status = "failed"
                    record.error = str(exc)
                    record.finished_at = time.time()
            finally:
                with self._cond:
                    self._active_request_ids.discard(request_id)

    def _position_for_request_unlocked(self, request_id: str) -> int:
        if not request_id:
            return -1
        if request_id in self._active_request_ids:
            return 0
        return self._pending_position_unlocked(request_id)

    def _pending_position_unlocked(self, request_id: str) -> int:
        for index, (rid, _) in enumerate(self._pending):
            if rid == request_id:
                return index + 1
        return -1


def _model_dump(model: Optional[BaseModel], **kwargs) -> Dict[str, object]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)


def _build_refine_options(params: Optional[RefineParameters] = None) -> Dict[str, object]:
    return resolve_refine_options(_model_dump(params, exclude_none=True))


RUN_QUEUE = ConcurrentTaskQueue()


def _reset_bootstrap_status() -> None:
    global BOOTSTRAP_STATUS
    with BOOTSTRAP_LOCK:
        BOOTSTRAP_STATUS = BootstrapStatus()


def _log_runtime_diagnostics(stage: str, **extra: object) -> None:
    print(
        format_runtime_diagnostics(
            f"refine-api:{stage}",
            extra={key: value for key, value in extra.items() if value is not None},
        ),
        flush=True,
    )


def _log_line(message: str) -> None:
    print(message, flush=True)


def _log_stage(message: str) -> None:
    _log_line(f"[stage] {message}")


def _log_managed_service_exit(
    proc_info: ManagedServiceProcess,
    *,
    prefix: str,
    returncode: int | None = None,
) -> None:
    code = proc_info.process.poll() if returncode is None else returncode
    _log_line(
        f"[stack] {prefix} {proc_info.name} pid={proc_info.process.pid} "
        f"{format_process_exit_status(code)}"
    )


def _update_bootstrap_status(**changes: Any) -> None:
    with BOOTSTRAP_LOCK:
        for key, value in changes.items():
            setattr(BOOTSTRAP_STATUS, key, value)


def _bootstrap_snapshot() -> dict[str, Any]:
    with BOOTSTRAP_LOCK:
        payload = {
            "state": BOOTSTRAP_STATUS.state,
            "error": BOOTSTRAP_STATUS.error,
            "dependencies_ready": BOOTSTRAP_STATUS.dependencies_ready,
            "slots_ready": BOOTSTRAP_STATUS.slots_ready,
            "workers_ready": BOOTSTRAP_STATUS.workers_ready,
            "started_at": BOOTSTRAP_STATUS.started_at,
            "finished_at": BOOTSTRAP_STATUS.finished_at,
        }
    payload["managed_service_failures"] = _managed_service_failures()
    return payload


def _managed_service_failures() -> List[dict[str, Any]]:
    failures: List[dict[str, Any]] = []
    with MANAGED_SERVICE_LOCK:
        for proc_info in MANAGED_SERVICE_PROCS:
            exit_code = proc_info.process.poll()
            if exit_code is None:
                continue
            failures.append(
                {
                    "name": proc_info.name,
                    "exit_code": exit_code,
                    "exit_status": format_process_exit_status(exit_code),
                }
            )
    return failures


def _runtime_is_ready() -> bool:
    snapshot = _bootstrap_snapshot()
    return (
        snapshot["state"] == "ready"
        and snapshot["dependencies_ready"]
        and snapshot["slots_ready"]
        and snapshot["workers_ready"]
        and not snapshot["managed_service_failures"]
    )


def _runtime_live_error() -> str:
    snapshot = _bootstrap_snapshot()
    if snapshot["state"] == "failed":
        return str(snapshot["error"] or "runtime bootstrap failed")
    failures = snapshot.get("managed_service_failures") or []
    if failures:
        first = failures[0]
        return (
            "managed service exited: "
            f"{first['name']} "
            f"({first.get('exit_status') or format_process_exit_status(first['exit_code'])})"
        )
    return ""


def _probe_payload(probe: str) -> dict[str, Any]:
    snapshot = _bootstrap_snapshot()
    return {
        "status": "ok",
        "service": "ultrashape-refine-api",
        "probe": probe,
        "bootstrap_state": snapshot["state"],
        "dependencies_ready": snapshot["dependencies_ready"],
        "slots_ready": snapshot["slots_ready"],
        "workers_ready": snapshot["workers_ready"],
        "error": snapshot["error"],
        "managed_service_failures": snapshot["managed_service_failures"],
    }


def _require_runtime_ready() -> None:
    if _runtime_is_ready():
        return
    payload = _probe_payload("ready")
    raise HTTPException(status_code=503, detail=payload)


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


def _primary_visible_device(value: str, *, fallback: str = "") -> str:
    slots = _split_slots(value) if value else []
    if len(slots) == 1 and "," in slots[0]:
        slots = [item.strip() for item in slots[0].split(",") if item.strip()]
    if slots:
        return slots[0]
    return str(fallback or "").strip()


def _build_managed_service_env(
    base_env: Dict[str, str],
    *,
    service: str,
    visible_device: str,
) -> Dict[str, str]:
    env = dict(base_env)
    device = str(visible_device).strip()
    env["CUDA_VISIBLE_DEVICES"] = device
    if service == "hunyuan":
        env["HUNYUAN_CUDA_VISIBLE_DEVICES"] = device
    elif service == "ultrashape":
        env["ULTRASHAPE_CUDA_VISIBLE_DEVICES"] = device
    else:
        raise ValueError(f"Unsupported managed service: {service}")
    return env


def _parse_service_endpoint(base_url: str) -> Dict[str, object]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"Unsupported service URL scheme: {base_url}")
    if not parsed.hostname:
        raise RuntimeError(f"Service URL missing hostname: {base_url}")
    if parsed.port is None:
        raise RuntimeError(f"Service URL missing port: {base_url}")

    return {
        "base_url": base_url.rstrip("/"),
        "host": parsed.hostname,
        "bind_host": "0.0.0.0" if parsed.hostname in {"127.0.0.1", "localhost"} else parsed.hostname,
        "port": parsed.port,
    }


def _normalize_probe_path(path: str) -> str:
    probe_path = str(path or "").strip() or "/health"
    return probe_path if probe_path.startswith("/") else f"/{probe_path}"


def _wait_for_probe(base_url: str, timeout: float, service_name: str, *, probe_path: str) -> None:
    deadline = time.time() + timeout
    probe_url = f"{base_url.rstrip('/')}{_normalize_probe_path(probe_path)}"
    last_error = ""

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(probe_url, timeout=5) as response:
                if response.status < 400:
                    return
                last_error = f"HTTP {response.status}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)

    raise RuntimeError(
        f"{service_name} did not pass {_normalize_probe_path(probe_path)} within {timeout:.0f}s: {last_error}"
    )


def _spawn_service_process(
    service_name: str,
    python_bin: str,
    script_path: str,
    cwd: str,
    env: Dict[str, str],
) -> subprocess.Popen:
    device_hint = (
        env.get("HUNYUAN_CUDA_VISIBLE_DEVICES")
        or env.get("ULTRASHAPE_CUDA_VISIBLE_DEVICES")
        or env.get("CUDA_VISIBLE_DEVICES")
        or "<inherit>"
    )
    print(
        f"[stack] starting {service_name}: {python_bin} {script_path} "
        f"(cuda_visible_devices={device_hint})",
        flush=True,
    )
    return subprocess.Popen(
        [python_bin, script_path],
        cwd=cwd,
        env=env,
    )


def _managed_service_name(service_kind: str, slot_id: str | int) -> str:
    return f"{service_kind}-{slot_id}"


def _managed_service_matches(
    proc_info: ManagedServiceProcess,
    *,
    service_kind: str | None = None,
    slot_id: str | None = None,
) -> bool:
    if service_kind is not None and not proc_info.name.startswith(f"{service_kind}-"):
        return False
    if slot_id is not None and proc_info.name != _managed_service_name(service_kind or "", slot_id):
        return False
    return True


def _find_managed_service_locked(service_name: str) -> ManagedServiceProcess | None:
    for proc_info in MANAGED_SERVICE_PROCS:
        if proc_info.name == service_name:
            return proc_info
    return None


def _managed_service_enabled(service_kind: str) -> bool:
    if service_kind == "hunyuan":
        return AUTO_START_HUNYUAN
    if service_kind == "ultrashape":
        return AUTO_START_ULTRASHAPE
    raise ValueError(f"Unsupported service kind: {service_kind}")


def _managed_service_specs(service_kind: str) -> List[Dict[str, str]]:
    if service_kind == "hunyuan":
        urls = _split_slots(HUNYUAN_SERVICE_URLS)
        devices = _split_slots(HUNYUAN_CUDA_VISIBLE_DEVICES_LIST)
        if len(urls) != len(devices):
            raise RuntimeError(
                "HUNYUAN_SERVICE_URLS and HUNYUAN_CUDA_VISIBLE_DEVICES_LIST length mismatch"
            )
        specs: List[Dict[str, str]] = []
        for idx, (base_url, device) in enumerate(zip(urls, devices)):
            endpoint = _parse_service_endpoint(base_url)
            specs.append(
                {
                    "service_name": _managed_service_name("hunyuan", idx),
                    "base_url": str(base_url),
                    "visible_device": str(device),
                    "python_bin": HUNYUAN_SERVICE_PYTHON,
                    "script_path": os.path.join(HUNYUAN_ROOT, "api", "hunyuan_server.py"),
                    "cwd": HUNYUAN_ROOT,
                    "cache_dir": HUNYUAN_CACHE_DIR,
                    "host": str(endpoint["bind_host"]),
                    "port": str(endpoint["port"]),
                    "slot_id": str(idx),
                }
            )
        return specs
    if service_kind == "ultrashape":
        urls = _split_slots(ULTRASHAPE_SERVICE_URLS)
        devices = _split_slots(ULTRASHAPE_CUDA_VISIBLE_DEVICES_LIST)
        if len(urls) != len(devices):
            raise RuntimeError(
                "ULTRASHAPE_SERVICE_URLS and ULTRASHAPE_CUDA_VISIBLE_DEVICES_LIST length mismatch"
            )
        specs = []
        for idx, (base_url, device) in enumerate(zip(urls, devices)):
            endpoint = _parse_service_endpoint(base_url)
            specs.append(
                {
                    "service_name": _managed_service_name("ultrashape", idx),
                    "base_url": str(base_url),
                    "visible_device": str(device),
                    "python_bin": ULTRASHAPE_SERVICE_PYTHON,
                    "script_path": os.path.join(ULTRASHAPE_ROOT, "api", "ultrashape_server.py"),
                    "cwd": ULTRASHAPE_ROOT,
                    "cache_dir": ULTRASHAPE_CACHE_DIR,
                    "host": str(endpoint["bind_host"]),
                    "port": str(endpoint["port"]),
                    "slot_id": str(idx),
                }
            )
        return specs
    raise ValueError(f"Unsupported service kind: {service_kind}")


def _build_managed_service_process_env(service_kind: str, spec: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(_build_cache_env(spec["cache_dir"]))
    env = _build_managed_service_env(
        env,
        service=service_kind,
        visible_device=spec["visible_device"],
    )
    if service_kind == "hunyuan":
        env.update(
            {
                "HUNYUAN_ROOT": HUNYUAN_ROOT,
                "HUNYUAN_CACHE_DIR": HUNYUAN_CACHE_DIR,
                "HUNYUAN_SERVICE_HOST": spec["host"],
                "HUNYUAN_SERVICE_PORT": spec["port"],
            }
        )
    elif service_kind == "ultrashape":
        env.update(
            {
                "ULTRASHAPE_CACHE_DIR": ULTRASHAPE_CACHE_DIR,
                "ULTRASHAPE_SERVICE_HOST": spec["host"],
                "ULTRASHAPE_SERVICE_PORT": spec["port"],
            }
        )
    else:
        raise ValueError(f"Unsupported service kind: {service_kind}")
    return env


def _stop_managed_services(
    service_kind: str | None = None,
    *,
    slot_id: str | None = None,
) -> None:
    with MANAGED_SERVICE_LOCK:
        if service_kind is None and slot_id is None:
            procs = list(MANAGED_SERVICE_PROCS)
            MANAGED_SERVICE_PROCS.clear()
        else:
            procs = []
            remaining = []
            for proc_info in MANAGED_SERVICE_PROCS:
                if _managed_service_matches(
                    proc_info,
                    service_kind=service_kind,
                    slot_id=slot_id,
                ):
                    procs.append(proc_info)
                else:
                    remaining.append(proc_info)
            MANAGED_SERVICE_PROCS[:] = remaining

    if not procs:
        return

    for proc_info in procs:
        if proc_info.process.poll() is not None:
            _log_managed_service_exit(proc_info, prefix="service already stopped")
            continue
        _log_line(f"[stack] stopping {proc_info.name} pid={proc_info.process.pid} via terminate")
        proc_info.process.terminate()

    deadline = time.time() + 10
    for proc_info in procs:
        proc = proc_info.process
        exit_code = proc.poll()
        if exit_code is not None:
            _log_managed_service_exit(proc_info, prefix="service stopped", returncode=exit_code)
            continue
        remaining = max(0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
            _log_managed_service_exit(proc_info, prefix="service stopped")
        except subprocess.TimeoutExpired:
            _log_line(f"[stack] killing {proc_info.name} pid={proc.pid} after terminate timeout")
            proc.kill()
            proc.wait(timeout=5)
            _log_managed_service_exit(proc_info, prefix="service killed")


def _start_managed_services(
    service_kind: str,
    *,
    slot_id: str | None = None,
) -> None:
    if not USE_REMOTE_SERVICES or not AUTO_START_LOCAL_SERVICES:
        return
    if not _managed_service_enabled(service_kind):
        return

    with MANAGED_SERVICE_LOCK:
        started: List[ManagedServiceProcess] = []
        try:
            for spec in _managed_service_specs(service_kind):
                if slot_id is not None and spec["slot_id"] != str(slot_id):
                    continue
                existing = _find_managed_service_locked(spec["service_name"])
                if existing is not None:
                    if existing.process.poll() is None:
                        continue
                    _log_managed_service_exit(
                        existing,
                        prefix="removing exited service before restart",
                    )
                    MANAGED_SERVICE_PROCS.remove(existing)
                env = _build_managed_service_process_env(service_kind, spec)
                started.append(
                    ManagedServiceProcess(
                        name=spec["service_name"],
                        process=_spawn_service_process(
                            spec["service_name"],
                            spec["python_bin"],
                            spec["script_path"],
                            spec["cwd"],
                            env,
                        ),
                    )
                )

            MANAGED_SERVICE_PROCS.extend(started)
        except Exception:
            for proc_info in started:
                proc = proc_info.process
                if proc.poll() is None:
                    proc.terminate()
            for proc_info in started:
                proc = proc_info.process
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            for proc_info in started:
                with contextlib.suppress(ValueError):
                    MANAGED_SERVICE_PROCS.remove(proc_info)
            raise


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
        _log_line(f"[scheduler] slots ready: {slot_ids}")


def _acquire_slot(*, block: bool = True, timeout: float | None = None) -> Dict[str, str]:
    try:
        if block:
            if timeout is None:
                slot = SLOT_QUEUE.get()
            else:
                slot = SLOT_QUEUE.get(timeout=timeout)
        else:
            slot = SLOT_QUEUE.get_nowait()
    except queue.Empty as exc:
        raise NodeBusyError("node busy") from exc
    _log_line(f"[scheduler] acquired slot {slot['slot_id']}")
    return slot


def _release_slot(slot: Dict[str, str]) -> None:
    SLOT_QUEUE.put(slot)
    _log_line(f"[scheduler] released slot {slot['slot_id']}")


def _wait_for_slot_services_ready() -> None:
    if not USE_REMOTE_SERVICES:
        return

    seen: set[tuple[str, str]] = set()
    with SLOTS_LOCK:
        slots = list(SLOTS)

    for slot in slots:
        for service_name, url_key in (("hunyuan", "hunyuan_url"), ("ultrashape", "ultrashape_url")):
            base_url = str(slot.get(url_key) or "").strip()
            if not base_url:
                continue
            identity = (service_name, base_url)
            if identity in seen:
                continue
            seen.add(identity)
            _wait_for_probe(
                base_url,
                LOCAL_SERVICE_START_TIMEOUT,
                f"{service_name}:{base_url}",
                probe_path="/ready",
            )


def _ensure_slot_service_ready(service_kind: str, slot: Mapping[str, str]) -> None:
    if not USE_REMOTE_SERVICES:
        return

    slot_id = str(slot.get("slot_id") or "").strip()
    base_url = str(slot.get(f"{service_kind}_url") or "").strip()
    if not base_url:
        return

    if AUTO_START_LOCAL_SERVICES and _managed_service_enabled(service_kind):
        _start_managed_services(service_kind, slot_id=slot_id or None)

    _wait_for_probe(
        base_url,
        LOCAL_SERVICE_START_TIMEOUT,
        f"{service_kind}:{slot_id or base_url}",
        probe_path="/ready",
    )


def _stop_slot_service(service_kind: str, slot: Mapping[str, str]) -> None:
    if not USE_REMOTE_SERVICES or not AUTO_START_LOCAL_SERVICES:
        return
    if not _managed_service_enabled(service_kind):
        return
    slot_id = str(slot.get("slot_id") or "").strip()
    _stop_managed_services(service_kind, slot_id=slot_id or None)


def _post_json(
    url: str,
    payload: Dict[str, object],
    timeout: float,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=data,
        headers=req_headers,
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


def _tmp_root() -> str:
    os.makedirs(TMP_ROOT, exist_ok=True)
    return TMP_ROOT


def _request_workspace(request_id: str) -> str:
    workspace = os.path.join(_tmp_root(), "requests", request_id)
    os.makedirs(workspace, exist_ok=True)
    return workspace


def _safe_filename(name: str) -> str:
    file_name = os.path.basename((name or "").strip()) or "input-image"
    file_name = file_name.replace("\x00", "")
    return file_name


def _normalize_upload_name(name: str, image_bytes: bytes) -> str:
    file_name = _safe_filename(name)
    stem, ext = os.path.splitext(file_name)
    if ext:
        return file_name

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        fmt = (image.format or "").lower()
    except Exception:
        fmt = ""

    guessed_ext = {
        "jpeg": ".jpg",
        "jpg": ".jpg",
        "png": ".png",
        "webp": ".webp",
        "bmp": ".bmp",
    }.get(fmt, ".png")
    return f"{stem or 'input-image'}{guessed_ext}"


def _write_binary(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _normalize_oss_key(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if raw.startswith("oss://"):
        payload = raw[len("oss://") :]
        if "/" not in payload:
            return ""
        bucket, key = payload.split("/", 1)
        if bucket and bucket != OSS_BUCKET:
            raise RuntimeError(f"OSS bucket mismatch: {bucket}")
        return key.lstrip("/")
    return raw.lstrip("/")


def _join_oss_key(*parts: str) -> str:
    chunks = []
    for part in parts:
        p = str(part or "").strip("/")
        if p:
            chunks.append(p)
    return "/".join(chunks)


def _oss_url(oss_key: str) -> str:
    key = _normalize_oss_key(oss_key)
    if not key:
        raise RuntimeError("Empty OSS key")
    return f"oss://{OSS_BUCKET}/{key}"


def _run_ossutil(args: list[str]) -> str:
    timeout_sec = int(os.environ.get("ULTRASHAPE_OSSUTIL_TIMEOUT_SEC", "1800"))
    cmd = ["ossutil", *args]
    print(f"[oss] exec: {' '.join(cmd)} (timeout={timeout_sec}s)")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ossutil timeout after {timeout_sec}s") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ossutil not found in PATH") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "unknown ossutil error"
        raise RuntimeError(f"ossutil {' '.join(args[:2])} failed: {detail}")
    return (result.stdout or "").strip()


def oss_upload_file(local_path: str, oss_key: str) -> None:
    _run_ossutil(["cp", local_path, _oss_url(oss_key), "-f"])


def oss_sign_url(oss_key: str, expires: str = OSS_SIGN_EXPIRES) -> str:
    output = _run_ossutil(["presign", _oss_url(oss_key), "--expires-duration", expires])
    for line in output.splitlines():
        text = line.strip()
        if text.startswith("http://") or text.startswith("https://"):
            return text
    text = output.strip()
    if text:
        return text
    raise RuntimeError("ossutil presign returned empty output")


def _upload_artifact(local_path: str, oss_key: str) -> dict[str, str]:
    oss_upload_file(local_path, oss_key)
    return {
        "oss_key": oss_key,
        "url": oss_sign_url(oss_key),
    }


def _legacy_task_status(status: str) -> str:
    return {
        "pending": "PENDING",
        "processing": "RUNNING",
        "completed": "SUCCEEDED",
        "failed": "FAILED",
    }.get((status or "").strip().lower(), "PENDING")


def _format_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


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
    image_bytes: bytes,
    mesh_bytes: bytes,
    base_url: str,
    params: Optional[RefineParameters] = None,
) -> bytes:
    payload = {
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "mesh_base64": base64.b64encode(mesh_bytes).decode("utf-8"),
    }
    payload.update(_build_refine_options(params))
    response = _post_json(
        f"{base_url}/refine", payload, timeout=SERVICE_TIMEOUT
    )
    mesh_base64 = response.get("mesh_base64")
    if not mesh_base64:
        raise RuntimeError("UltraShape service returned empty mesh")
    return base64.b64decode(mesh_base64)


def _apply_remote_memory_action(base_url: str, action: str, service_name: str) -> None:
    normalized = normalize_memory_action(action, default="none")
    if normalized == "none":
        return

    try:
        _post_json(
            f"{base_url.rstrip('/')}/memory/{normalized}",
            {},
            timeout=MEMORY_ACTION_TIMEOUT,
        )
        print(f"[memory] {service_name} {normalized} completed")
    except Exception as exc:
        detail = f"{service_name} {normalized} failed: {exc}"
        if service_name == "ultrashape":
            print(f"[memory] {detail}")
            return
        raise RuntimeError(detail) from exc


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
    os.makedirs(_tmp_root(), exist_ok=True)
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
    image_path: str,
    mesh_path: str,
    output_dir: str,
    cuda_devices: str,
    params: Optional[RefineParameters] = None,
) -> str:
    if not os.path.isdir(ULTRASHAPE_ROOT):
        raise RuntimeError(f"UltraShape root not found: {ULTRASHAPE_ROOT}")

    refine_options = _build_refine_options(params)
    env = os.environ.copy()
    env.update(
        {
            "ULTRASHAPE_DISABLE_FLASH_ATTN": "1",
            "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF,
            "ULTRASHAPE_CUDA_VISIBLE_DEVICES": cuda_devices,
            "ULTRASHAPE_OCTREE_RES": str(refine_options["octree_res"]),
            "ULTRASHAPE_STEPS": str(refine_options["steps"]),
            "ULTRASHAPE_CHUNK_SIZE": str(refine_options["chunk_size"]),
            "ULTRASHAPE_SCALE": str(refine_options["scale"]),
            "ULTRASHAPE_SEED": str(refine_options["seed"]),
            "ULTRASHAPE_REMOVE_BG": "1" if refine_options["remove_bg"] else "0",
            "ULTRASHAPE_OUTPUT_DIR": output_dir,
        }
    )
    if "num_latents" in refine_options:
        env["ULTRASHAPE_NUM_LATENTS"] = str(refine_options["num_latents"])
    env.update(_build_cache_env(ULTRASHAPE_CACHE_DIR))

    print(
        "[ultrashape] ULTRASHAPE_CUDA_VISIBLE_DEVICES="
        f"{env['ULTRASHAPE_CUDA_VISIBLE_DEVICES']}"
    )
    print(f"[ultrashape] PYTORCH_CUDA_ALLOC_CONF={PYTORCH_CUDA_ALLOC_CONF}")
    print(f"[ultrashape] ULTRASHAPE_OCTREE_RES={env['ULTRASHAPE_OCTREE_RES']}")
    print(f"[ultrashape] ULTRASHAPE_STEPS={env['ULTRASHAPE_STEPS']}")
    print(f"[ultrashape] ULTRASHAPE_CHUNK_SIZE={env['ULTRASHAPE_CHUNK_SIZE']}")
    if "ULTRASHAPE_NUM_LATENTS" in env:
        print(f"[ultrashape] ULTRASHAPE_NUM_LATENTS={env['ULTRASHAPE_NUM_LATENTS']}")

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
            print(
                "[ultrashape] OOM detected, retrying with "
                f"octree_res={retry_env['ULTRASHAPE_OCTREE_RES']}"
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


def _run_refinement_job(
    request_id: str,
    image_bytes: bytes,
    upload_name: str,
    params: Optional[RefineParameters] = None,
    *,
    wait_for_slot: bool = True,
) -> dict[str, Any]:
    _ensure_dirs()
    _init_slots()
    workspace_dir = _request_workspace(request_id)
    file_name = _normalize_upload_name(upload_name, image_bytes)
    input_image_path = os.path.join(workspace_dir, file_name)
    coarse_mesh_path = os.path.join(workspace_dir, "coarse.glb")
    refined_path = os.path.join(workspace_dir, "refined.glb")
    oss_prefix_root = _join_oss_key(OSS_PREFIX, request_id)
    slot = _acquire_slot(block=wait_for_slot)

    _write_binary(input_image_path, image_bytes)
    _log_line(f"[api] request={request_id} workspace={workspace_dir}")

    try:
        if USE_REMOTE_SERVICES:
            refined_mesh_bytes = run_remote_refinement_pipeline(
                image_bytes=image_bytes,
                slot=slot,
                params=params,
                call_hunyuan=_call_hunyuan_service,
                call_ultrashape=_call_ultrashape_service,
                apply_memory_action=_apply_remote_memory_action,
                ensure_service_ready=_ensure_slot_service_ready,
                stop_service=_stop_slot_service,
                log_stage=_log_stage,
                hunyuan_post_action=HUNYUAN_POST_ACTION,
                ultrashape_post_action=ULTRASHAPE_POST_ACTION,
            )
            _write_binary(refined_path, refined_mesh_bytes)
        else:
            _run_hunyuan(input_image_path, coarse_mesh_path, slot["hunyuan_device"])
            refined_path = _run_ultrashape(
                input_image_path,
                coarse_mesh_path,
                workspace_dir,
                slot["ultrashape_device"],
                params,
            )

        input_oss_key = _join_oss_key(oss_prefix_root, "input", file_name)
        output_oss_key = _join_oss_key(oss_prefix_root, "output", "refined.glb")
        input_artifact = _upload_artifact(input_image_path, input_oss_key)
        glb_artifact = _upload_artifact(refined_path, output_oss_key)

        return {
            "request_id": request_id,
            "session_id": request_id,
            "oss_prefix": oss_prefix_root,
            "glb_url": glb_artifact["url"],
            "model_url": glb_artifact["url"],
            "artifacts": {
                "input_image": input_artifact,
                "glb": glb_artifact,
            },
        }
    finally:
        _release_slot(slot)
        if not KEEP_INTERMEDIATE:
            shutil.rmtree(workspace_dir, ignore_errors=True)


def _build_queue_payload(
    image_bytes: bytes,
    *,
    upload_name: str,
    params: Optional[RefineParameters],
) -> dict[str, Any]:
    return {
        "image_bytes": image_bytes,
        "upload_name": upload_name,
        "params": _model_dump(params, exclude_none=True),
    }


def _execute_run_request(request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    image_bytes = payload["image_bytes"]
    upload_name = str(payload.get("upload_name") or "input-image.png")
    params_data = payload.get("params") or {}
    params = RefineParameters(**params_data) if params_data else None
    return _run_refinement_job(request_id, image_bytes, upload_name, params)


def _start_worker() -> None:
    worker_count = ULTRASHAPE_QUEUE_WORKERS if ULTRASHAPE_QUEUE_WORKERS > 0 else len(SLOTS)
    RUN_QUEUE.start(
        _execute_run_request,
        worker_count=max(1, worker_count),
    )
    _update_bootstrap_status(workers_ready=True)


def _start_hot_start_thread() -> None:
    if not HOT_START_ENABLED:
        return
    threading.Thread(target=_hot_start, daemon=True, name="ultrashape-hot-start").start()


def _bootstrap_runtime() -> None:
    try:
        _log_runtime_diagnostics(
            "bootstrap_start",
            use_remote_services=USE_REMOTE_SERVICES,
            hunyuan_visible_devices=_primary_visible_device(
                HUNYUAN_CUDA_VISIBLE_DEVICES_LIST,
                fallback=HUNYUAN_CUDA_VISIBLE_DEVICES,
            ),
            ultrashape_visible_devices=_primary_visible_device(
                ULTRASHAPE_CUDA_VISIBLE_DEVICES_LIST,
                fallback=ULTRASHAPE_CUDA_VISIBLE_DEVICES,
            ),
        )
        _init_slots()
        with SLOTS_LOCK:
            slots = list(SLOTS)
        for slot in slots:
            _ensure_slot_service_ready("hunyuan", slot)
        _update_bootstrap_status(slots_ready=True)
        _update_bootstrap_status(
            state="ready",
            dependencies_ready=True,
            finished_at=time.time(),
        )
        _log_runtime_diagnostics("bootstrap_ready", slots=len(SLOTS))
        _log_line("[bootstrap] runtime is ready to receive requests")
    except Exception as exc:
        error = str(exc)
        _log_runtime_diagnostics("bootstrap_failed", error=error)
        _log_line(f"[bootstrap] runtime bootstrap failed: {error}")
        _update_bootstrap_status(
            state="failed",
            error=error,
            finished_at=time.time(),
        )


def _start_bootstrap_thread() -> None:
    global BOOTSTRAP_THREAD
    with BOOTSTRAP_LOCK:
        if BOOTSTRAP_THREAD is not None and BOOTSTRAP_THREAD.is_alive():
            return
        BOOTSTRAP_THREAD = threading.Thread(
            target=_bootstrap_runtime,
            daemon=True,
            name="ultrashape-runtime-bootstrap",
        )
        BOOTSTRAP_THREAD.start()


@app.on_event("startup")
def _startup() -> None:
    _reset_bootstrap_status()
    _ensure_dirs()
    _log_runtime_diagnostics(
        "startup",
        use_remote_services=USE_REMOTE_SERVICES,
        hot_start_enabled=HOT_START_ENABLED,
        hunyuan_visible_devices=_primary_visible_device(
            HUNYUAN_CUDA_VISIBLE_DEVICES_LIST,
            fallback=HUNYUAN_CUDA_VISIBLE_DEVICES,
        ),
        ultrashape_visible_devices=_primary_visible_device(
            ULTRASHAPE_CUDA_VISIBLE_DEVICES_LIST,
            fallback=ULTRASHAPE_CUDA_VISIBLE_DEVICES,
        ),
    )
    _start_worker()
    _start_hot_start_thread()
    _start_bootstrap_thread()


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop_managed_services()


@app.get("/health")
def health_check():
    return _probe_payload("health")


@app.get("/ready")
def ready_check():
    payload = _probe_payload("ready")
    if _runtime_is_ready():
        return payload
    payload["status"] = "starting"
    return JSONResponse(status_code=503, content=payload)


@app.get("/live")
def live_check():
    payload = _probe_payload("live")
    error = _runtime_live_error()
    if not error:
        return payload
    payload["status"] = "failed"
    payload["error"] = error
    return JSONResponse(status_code=503, content=payload)


def _build_params_from_form(
    precision: str = DEFAULT_PRECISION,
    steps: Optional[int] = None,
    octree_res: Optional[int] = None,
    num_latents: Optional[int] = None,
    chunk_size: Optional[int] = None,
    seed: Optional[int] = 42,
    remove_bg: Optional[bool] = False,
    scale: Optional[float] = 0.99,
) -> RefineParameters:
    return RefineParameters(
        precision=precision,
        steps=steps,
        octree_res=octree_res,
        num_latents=num_latents,
        chunk_size=chunk_size,
        seed=seed,
        remove_bg=remove_bg,
        scale=scale,
    )


def _ensure_image_upload(files: list[UploadFile]) -> UploadFile:
    if not files:
        raise HTTPException(status_code=400, detail="Exactly one image is required")
    if len(files) != 1:
        raise HTTPException(status_code=400, detail="Only one image is supported")

    upload = files[0]
    file_name = _safe_filename(upload.filename)
    is_image = bool(upload.content_type and upload.content_type.startswith("image/"))
    has_known_ext = os.path.splitext(file_name)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if not is_image and not has_known_ext:
        raise HTTPException(status_code=400, detail="Only image uploads are supported")
    return upload


async def _read_upload_bytes(upload: UploadFile) -> bytes:
    try:
        data = await upload.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    return data


def _enqueue_request(
    image_bytes: bytes,
    *,
    upload_name: str,
    params: Optional[RefineParameters],
    request_id: str = "",
) -> tuple[str, int]:
    if TASK_QUEUE_MAXSIZE > 0:
        queue_status = RUN_QUEUE.get_queue_status()
        active_count = int(queue_status.get("pending", 0)) + int(queue_status.get("processing_count", 0))
        if active_count >= TASK_QUEUE_MAXSIZE:
            raise HTTPException(status_code=429, detail="Queue is full, try again later")

    try:
        return RUN_QUEUE.enqueue(
            _build_queue_payload(image_bytes, upload_name=upload_name, params=params),
            request_id=request_id or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/queue_status")
def queue_status(request_id: str = ""):
    rid = request_id.strip()
    return RUN_QUEUE.get_queue_status(rid if rid else None)


@app.get("/request_status")
def request_status(request_id: str):
    rid = request_id.strip()
    if not rid:
        raise HTTPException(status_code=400, detail="request_id is required")
    payload = RUN_QUEUE.get_request_status(rid)
    if not payload:
        raise HTTPException(status_code=404, detail="request_id not found")
    return payload


@app.post("/reconstruct")
async def reconstruct(
    image: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    request_id: str = Form(""),
    time_interval: float = Form(1.0),  # compatible with task-manager hunyuanworld adapter
    frame_selector: str = Form("All"),  # compatible with task-manager hunyuanworld adapter
    show_camera: bool = Form(True),  # accepted for compatibility, unused by ultrashape
    show_mesh: bool = Form(True),  # accepted for compatibility, unused by ultrashape
    filter_sky_bg: bool = Form(False),  # accepted for compatibility, unused by ultrashape
    filter_ambiguous: bool = Form(True),  # accepted for compatibility, unused by ultrashape
    precision: str = Form(DEFAULT_PRECISION),
    steps: Optional[int] = Form(None),
    octree_res: Optional[int] = Form(None),
    num_latents: Optional[int] = Form(None),
    chunk_size: Optional[int] = Form(None),
    seed: Optional[int] = Form(42),
    remove_bg: Optional[bool] = Form(False),
    scale: Optional[float] = Form(0.99),
):
    del time_interval, frame_selector, show_camera, show_mesh, filter_sky_bg, filter_ambiguous
    _require_runtime_ready()
    uploads = [item for item in ([image] if image else []) if item is not None]
    if files:
        uploads.extend(files)
    upload = _ensure_image_upload(uploads)
    image_bytes = await _read_upload_bytes(upload)
    params = _build_params_from_form(
        precision=precision,
        steps=steps,
        octree_res=octree_res,
        num_latents=num_latents,
        chunk_size=chunk_size,
        seed=seed,
        remove_bg=remove_bg,
        scale=scale,
    )
    rid = request_id.strip() or uuid.uuid4().hex
    try:
        return _run_refinement_job(
            rid,
            image_bytes,
            _safe_filename(upload.filename),
            params,
            wait_for_slot=False,
        )
    except NodeBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/run_with_files")
async def run_with_files(
    image: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    request_id: str = Form(""),
    precision: str = Form(DEFAULT_PRECISION),
    steps: Optional[int] = Form(None),
    octree_res: Optional[int] = Form(None),
    num_latents: Optional[int] = Form(None),
    chunk_size: Optional[int] = Form(None),
    seed: Optional[int] = Form(42),
    remove_bg: Optional[bool] = Form(False),
    scale: Optional[float] = Form(0.99),
):
    _require_runtime_ready()
    uploads = [item for item in ([image] if image else []) if item is not None]
    if files:
        uploads.extend(files)
    upload = _ensure_image_upload(uploads)
    image_bytes = await _read_upload_bytes(upload)
    params = _build_params_from_form(
        precision=precision,
        steps=steps,
        octree_res=octree_res,
        num_latents=num_latents,
        chunk_size=chunk_size,
        seed=seed,
        remove_bg=remove_bg,
        scale=scale,
    )
    request_id_out, position = _enqueue_request(
        image_bytes,
        upload_name=_safe_filename(upload.filename),
        params=params,
        request_id=request_id,
    )
    return {
        "status": "queued",
        "request_id": request_id_out,
        "position": position,
    }


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    precision: str = Form(DEFAULT_PRECISION),
    steps: Optional[int] = Form(None),
    octree_res: Optional[int] = Form(None),
    num_latents: Optional[int] = Form(None),
    chunk_size: Optional[int] = Form(None),
    seed: Optional[int] = Form(42),
    remove_bg: Optional[bool] = Form(False),
    scale: Optional[float] = Form(0.99),
):
    _require_runtime_ready()
    upload = _ensure_image_upload([image])
    image_bytes = await _read_upload_bytes(upload)
    params = _build_params_from_form(
        precision=precision,
        steps=steps,
        octree_res=octree_res,
        num_latents=num_latents,
        chunk_size=chunk_size,
        seed=seed,
        remove_bg=remove_bg,
        scale=scale,
    )
    result = _run_refinement_job(uuid.uuid4().hex, image_bytes, _safe_filename(upload.filename), params)
    glb = result["artifacts"]["glb"]
    return RedirectResponse(glb["url"], status_code=307)


@app.post("/generate_3d")
async def generate_3d(req: GenerateRequest):
    _require_runtime_ready()
    image_bytes = _load_image_from_base64(req.image_base64)
    params = RefineParameters(**_model_dump(req, exclude={"image_base64"}))
    result = _run_refinement_job(uuid.uuid4().hex, image_bytes, "input-image.png", params)
    glb = result["artifacts"]["glb"]
    return {
        "status": "success",
        "format": "glb",
        "model_url": glb["url"],
        "oss_key": glb["oss_key"],
    }


@app.post("/api/v1/services/aigc/3d-refine/generation")
async def generate_async(req: AsyncGenerateRequest):
    _require_runtime_ready()
    image_bytes = _load_image_from_base64(req.input.image_base64)
    params = req.parameters or AsyncParameters()
    task_id, _position = _enqueue_request(
        image_bytes,
        upload_name="input-image.png",
        params=params,
    )
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
    payload = RUN_QUEUE.get_request_status(task_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Task not found")
    result = payload.get("result") or {}
    artifacts = result.get("artifacts") or {}
    glb = artifacts.get("glb") or {}

    return {
        "status_code": 200,
        "request_id": uuid.uuid4().hex,
        "code": None,
        "message": "",
        "output": {
            "task_id": task_id,
            "task_status": _legacy_task_status(str(payload.get("status") or "")),
            "glb_url": result.get("glb_url") or glb.get("url", ""),
            "model_url": result.get("model_url") or glb.get("url", ""),
            "result_url": f"/api/v1/tasks/{task_id}/result",
            "submit_time": _format_timestamp(payload.get("created_at")),
            "end_time": _format_timestamp(payload.get("finished_at")),
            "error_message": payload.get("error", ""),
            "session_id": result.get("session_id") or task_id,
            "artifacts": artifacts,
        },
        "usage": None,
    }


@app.get("/api/v1/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    payload = RUN_QUEUE.get_request_status(task_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Task not found")
    if payload.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Task not ready")
    result = payload.get("result") or {}
    artifacts = result.get("artifacts") or {}
    glb = artifacts.get("glb") or {}
    glb_url = glb.get("url")
    if not glb_url:
        raise HTTPException(status_code=404, detail="Model not found")
    return RedirectResponse(glb_url, status_code=307)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10083)
    args = parser.parse_args()

    _ensure_dirs()
    config = uvicorn.Config(app=app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
