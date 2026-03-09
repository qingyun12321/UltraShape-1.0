# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os


def _setup_cache_env() -> None:
    cache_root = os.environ.get("ULTRASHAPE_CACHE_DIR")
    if not cache_root:
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cache_root = os.path.join(package_root, "cache")

    cache_root = os.path.abspath(cache_root)
    hf_home = os.path.join(cache_root, "huggingface")
    torch_home = os.path.join(cache_root, "torch")
    u2net_home = os.path.join(cache_root, "u2net")

    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(torch_home, exist_ok=True)
    os.makedirs(u2net_home, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.environ.setdefault("DIFFUSERS_CACHE", os.path.join(hf_home, "diffusers"))
    os.environ.setdefault("TORCH_HOME", torch_home)
    os.environ.setdefault("HY3DGEN_MODELS", cache_root)
    os.environ.setdefault("U2NET_HOME", u2net_home)


_setup_cache_env()

__all__ = [
    "UltraShapePipeline",
    "FaceReducer",
    "FloaterRemover",
    "DegenerateFaceRemover",
    "MeshSimplifier",
    "ImageProcessorV2",
    "IMAGE_PROCESSORS",
    "DEFAULT_IMAGEPROCESSOR",
]


def __getattr__(name):
    if name == "UltraShapePipeline":
        from .pipelines import UltraShapePipeline
        return UltraShapePipeline
    if name in {"FaceReducer", "FloaterRemover", "DegenerateFaceRemover", "MeshSimplifier"}:
        from .postprocessors import (
            DegenerateFaceRemover,
            FaceReducer,
            FloaterRemover,
            MeshSimplifier,
        )
        return {
            "FaceReducer": FaceReducer,
            "FloaterRemover": FloaterRemover,
            "DegenerateFaceRemover": DegenerateFaceRemover,
            "MeshSimplifier": MeshSimplifier,
        }[name]
    if name in {"ImageProcessorV2", "IMAGE_PROCESSORS", "DEFAULT_IMAGEPROCESSOR"}:
        from .preprocessors import DEFAULT_IMAGEPROCESSOR, IMAGE_PROCESSORS, ImageProcessorV2
        return {
            "ImageProcessorV2": ImageProcessorV2,
            "IMAGE_PROCESSORS": IMAGE_PROCESSORS,
            "DEFAULT_IMAGEPROCESSOR": DEFAULT_IMAGEPROCESSOR,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
