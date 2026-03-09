import os
from typing import Any, Dict, Mapping, Optional


PRESET_ALIASES = {
    "default": "balanced",
    "standard": "balanced",
    "balanced": "balanced",
    "fast": "fast",
    "draft": "fast",
    "low": "fast",
    "legacy": "legacy",
    "backup": "legacy",
    "alpha": "legacy",
    "old-alpha": "legacy",
    "high": "quality",
    "quality": "quality",
    "best": "quality",
}

PRESET_OPTIONS = {
    "fast": {
        "steps": 12,
        "octree_res": 512,
        "num_latents": 8192,
        "chunk_size": 2048,
    },
    "balanced": {
        "steps": 24,
        "octree_res": 768,
        "num_latents": 16384,
        "chunk_size": 2048,
    },
    "legacy": {
        "steps": 12,
        "octree_res": 512,
        "num_latents": 32768,
        "chunk_size": 2048,
    },
    "quality": {
        "steps": 50,
        "octree_res": 1024,
        "num_latents": 32768,
        "chunk_size": 2048,
    },
}


def _env_str(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(name: str) -> Optional[int]:
    value = _env_str(name)
    return int(value) if value is not None else None


def _env_float(name: str) -> Optional[float]:
    value = _env_str(name)
    return float(value) if value is not None else None


def _env_bool(name: str) -> Optional[bool]:
    value = _env_str(name)
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes", "on"}


def normalize_precision(value: Optional[str]) -> str:
    key = (value or _env_str("ULTRASHAPE_PRECISION") or "standard").strip().lower()
    return PRESET_ALIASES.get(key, "balanced")


def resolve_refine_options(
    overrides: Optional[Mapping[str, Any]] = None,
    fallback_num_latents: Optional[int] = None,
) -> Dict[str, Any]:
    overrides = dict(overrides or {})
    normalized_precision = normalize_precision(overrides.get("precision"))

    options: Dict[str, Any] = dict(PRESET_OPTIONS[normalized_precision])
    options.update(
        {
            "precision": normalized_precision,
            "seed": 42,
            "remove_bg": False,
            "scale": 0.99,
        }
    )

    env_overrides = {
        "steps": _env_int("ULTRASHAPE_STEPS"),
        "octree_res": _env_int("ULTRASHAPE_OCTREE_RES"),
        "num_latents": _env_int("ULTRASHAPE_NUM_LATENTS"),
        "chunk_size": _env_int("ULTRASHAPE_CHUNK_SIZE"),
        "seed": _env_int("ULTRASHAPE_SEED"),
        "scale": _env_float("ULTRASHAPE_SCALE"),
        "remove_bg": _env_bool("ULTRASHAPE_REMOVE_BG"),
    }
    for key, value in env_overrides.items():
        if value is not None:
            options[key] = value

    if fallback_num_latents is not None and options.get("num_latents") is None:
        options["num_latents"] = int(fallback_num_latents)

    for key, value in overrides.items():
        if value is None:
            continue
        options[key] = value

    if options.get("num_latents") is None and fallback_num_latents is not None:
        options["num_latents"] = int(fallback_num_latents)

    return options
