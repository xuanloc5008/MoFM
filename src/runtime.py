"""
Runtime helpers for device selection and platform-specific defaults.
"""
from __future__ import annotations

import copy
import platform
from typing import Dict, List

import torch

_BUILTIN_RUNTIME_PROFILES: Dict[str, Dict] = {
    "default": {
        "train_batch_size": None,
        "val_batch_size": None,
        "num_workers": None,
        "pin_memory": None,
        "persistent_workers": None,
        "prefetch_factor": None,
        "non_blocking": False,
        "use_amp": False,
        "amp_dtype": "float32",
        "allow_tf32": False,
        "matmul_precision": "high",
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
    },
    "mps": {
        "train_batch_size": 4,
        "val_batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "non_blocking": False,
        "use_amp": False,
        "amp_dtype": "float32",
        "allow_tf32": False,
        "matmul_precision": "high",
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
    },
    "cuda": {
        "train_batch_size": None,
        "val_batch_size": None,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "non_blocking": True,
        "use_amp": True,
        "amp_dtype": "bfloat16",
        "allow_tf32": True,
        "matmul_precision": "high",
        "cudnn_benchmark": True,
        "cudnn_deterministic": False,
    },
    "cuda_l40": {
        # Cached topology vectors move contrastive pair mining onto the GPU, so
        # L40/L40S cards can sustain a meaningfully larger batch here.
        "train_batch_size": 24,
        "val_batch_size": 48,
        "num_workers": 8,
        "prefetch_factor": 4,
    },
}

_AMP_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_device(requested: str = "auto", gpu_index: int = 0) -> torch.device:
    """
    Resolve the execution device.

    Priority for ``auto`` is CUDA -> MPS -> CPU.
    """
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_index}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return torch.device(f"cuda:{gpu_index}")

    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_built():
            raise RuntimeError("MPS requested, but this PyTorch build has no MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested, but torch.backends.mps.is_available() is False. "
                "Use an Apple Silicon Python/PyTorch environment and verify Metal is available."
            )
        return torch.device("mps")

    raise ValueError(f"Unsupported device '{requested}'. Expected one of auto/cpu/cuda/mps.")


def get_device_name(device: torch.device, gpu_index: int = 0) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(gpu_index)
    if device.type == "mps":
        return "Apple Silicon MPS"
    return platform.processor() or "CPU"


def pin_memory_for_device(device: torch.device) -> bool:
    """Pinned memory only helps CUDA host-to-device transfers."""
    return device.type == "cuda"


def resolve_num_workers(requested_workers: int, device: torch.device) -> int:
    """
    Choose a DataLoader worker count that is safe for the current platform.

    On macOS with MPS, multiprocessing startup is often slower than the actual
    data loading for this project and can stall or appear hung before epoch 1.
    """
    workers = max(0, int(requested_workers))
    if platform.system() == "Darwin" and device.type == "mps" and workers > 0:
        return 0
    return workers


def resolve_amp_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key not in _AMP_DTYPES:
        raise ValueError(
            f"Unsupported amp_dtype '{name}'. Expected one of {sorted(_AMP_DTYPES)}."
        )
    return _AMP_DTYPES[key]


def resolve_runtime_profile_names(device: torch.device, gpu_index: int = 0) -> List[str]:
    names = ["default", device.type]
    if device.type == "cuda":
        device_name = get_device_name(device, gpu_index).upper()
        if "L40" in device_name:
            names.append("cuda_l40")
    return names


def resolve_runtime_settings(cfg: dict, device: torch.device, gpu_index: int = 0) -> Dict:
    runtime_cfg = cfg.get("runtime", {})
    user_profiles = runtime_cfg.get("profiles", {})
    training_cfg = cfg.get("training", {})

    profile_names = resolve_runtime_profile_names(device, gpu_index)

    settings: Dict = {}
    for name in profile_names:
        settings.update(copy.deepcopy(_BUILTIN_RUNTIME_PROFILES.get(name, {})))
        settings.update(copy.deepcopy(user_profiles.get(name, {})))

    requested_workers = settings.get("num_workers")
    if requested_workers is None:
        requested_workers = training_cfg.get("num_workers", 4)

    train_batch_size = settings.get("train_batch_size")
    if train_batch_size is None:
        train_batch_size = training_cfg.get("batch_size", 8)

    val_batch_size = settings.get("val_batch_size")
    if val_batch_size is None:
        val_batch_size = train_batch_size

    resolved_workers = resolve_num_workers(requested_workers, device)
    pin_memory = settings.get("pin_memory")
    if pin_memory is None:
        pin_memory = pin_memory_for_device(device)

    persistent_workers = settings.get("persistent_workers")
    if persistent_workers is None:
        persistent_workers = resolved_workers > 0
    persistent_workers = bool(persistent_workers and resolved_workers > 0)

    prefetch_factor = settings.get("prefetch_factor")
    if resolved_workers == 0:
        prefetch_factor = None

    use_amp = bool(settings.get("use_amp", False) and device.type == "cuda")
    non_blocking = bool(settings.get("non_blocking", False) and device.type == "cuda")
    allow_tf32 = bool(settings.get("allow_tf32", False) and device.type == "cuda")

    return {
        "profile_names": profile_names,
        "device_name": get_device_name(device, gpu_index),
        "train_batch_size": int(train_batch_size),
        "val_batch_size": int(val_batch_size),
        "requested_num_workers": int(requested_workers),
        "num_workers": int(resolved_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "non_blocking": non_blocking,
        "use_amp": use_amp,
        "amp_dtype": str(settings.get("amp_dtype", "float32")).lower(),
        "allow_tf32": allow_tf32,
        "matmul_precision": str(settings.get("matmul_precision", "high")).lower(),
        "cudnn_benchmark": bool(settings.get("cudnn_benchmark", False) and device.type == "cuda"),
        "cudnn_deterministic": bool(
            settings.get("cudnn_deterministic", False) and device.type == "cuda"
        ),
    }


def configure_torch_runtime(device: torch.device, settings: Dict):
    matmul_precision = settings.get("matmul_precision", "high")
    if matmul_precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(matmul_precision)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = settings.get("allow_tf32", False)
        torch.backends.cudnn.allow_tf32 = settings.get("allow_tf32", False)
        torch.backends.cudnn.benchmark = settings.get("cudnn_benchmark", False)
        torch.backends.cudnn.deterministic = settings.get("cudnn_deterministic", False)


def dataloader_kwargs(runtime_settings: Dict) -> Dict:
    kwargs = {
        "num_workers": runtime_settings["num_workers"],
        "pin_memory": runtime_settings["pin_memory"],
        "persistent_workers": runtime_settings["persistent_workers"],
    }
    if runtime_settings["prefetch_factor"] is not None:
        kwargs["prefetch_factor"] = runtime_settings["prefetch_factor"]
    return kwargs
