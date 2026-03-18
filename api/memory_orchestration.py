from __future__ import annotations

from typing import Any, Callable, Mapping


MemoryAction = str


def normalize_memory_action(value: str | None, *, default: str = "none") -> MemoryAction:
    raw = (value or "").strip().lower()
    if not raw:
        raw = default.strip().lower()

    aliases = {
        "": "none",
        "none": "none",
        "noop": "none",
        "skip": "none",
        "offload": "offload",
        "cpu": "offload",
        "cpu_offload": "offload",
        "unload": "unload",
        "free": "unload",
        "release": "unload",
    }
    normalized = aliases.get(raw)
    if normalized is None:
        raise ValueError(f"Unsupported memory action: {value}")
    return normalized


def run_remote_refinement_pipeline(
    *,
    image_bytes: bytes,
    slot: Mapping[str, str],
    params: Any,
    call_hunyuan: Callable[[bytes, str], bytes],
    call_ultrashape: Callable[[bytes, bytes, str, Any], bytes],
    apply_memory_action: Callable[[str, MemoryAction, str], None],
    ensure_service_ready: Callable[[str, Mapping[str, str]], None] | None = None,
    stop_service: Callable[[str, Mapping[str, str]], None] | None = None,
    log_stage: Callable[[str], None] | None = None,
    hunyuan_post_action: str = "unload",
    ultrashape_post_action: str = "offload",
) -> bytes:
    slot_id = str(slot.get("slot_id") or "?")
    hunyuan_url = str(slot["hunyuan_url"])
    ultrashape_url = str(slot["ultrashape_url"])
    normalized_hunyuan_action = normalize_memory_action(hunyuan_post_action, default="unload")
    normalized_ultrashape_action = normalize_memory_action(ultrashape_post_action, default="offload")

    def emit(service_name: str, action: str, **extra: object) -> None:
        if log_stage is None:
            return
        message = f"slot={slot_id} stage={service_name} action={action}"
        for key, value in extra.items():
            if value is None:
                continue
            message += f" {key}={value}"
        log_stage(message)

    coarse_mesh_bytes = b""
    ultrashape_started = False
    hunyuan_started = False
    try:
        if ensure_service_ready is not None:
            emit("hunyuan", "ensure_ready")
            ensure_service_ready("hunyuan", slot)
            hunyuan_started = True
        emit("hunyuan", "inference_start")
        coarse_mesh_bytes = call_hunyuan(image_bytes, hunyuan_url)
        if normalized_hunyuan_action != "none":
            emit("hunyuan", "memory_action", memory_action=normalized_hunyuan_action)
            apply_memory_action(hunyuan_url, normalized_hunyuan_action, "hunyuan")
        if stop_service is not None and hunyuan_started:
            emit("hunyuan", "stop")
            stop_service("hunyuan", slot)
            hunyuan_started = False
        if ensure_service_ready is not None:
            emit("ultrashape", "ensure_ready")
            ensure_service_ready("ultrashape", slot)
        ultrashape_started = True
        emit("ultrashape", "inference_start")
        return call_ultrashape(image_bytes, coarse_mesh_bytes, ultrashape_url, params)
    finally:
        if stop_service is not None and hunyuan_started:
            emit("hunyuan", "stop")
            stop_service("hunyuan", slot)
        if ultrashape_started and normalized_ultrashape_action != "none":
            emit("ultrashape", "memory_action", memory_action=normalized_ultrashape_action)
            apply_memory_action(ultrashape_url, normalized_ultrashape_action, "ultrashape")
        if stop_service is not None and ultrashape_started:
            emit("ultrashape", "stop")
            stop_service("ultrashape", slot)
