"""
Configured cardiac dataset sources.

This module centralizes the train/validation dataset selection so training and
offline preprocessing use the same M&Ms/ACDC split contract.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .acdc import acdc_train_val_split, collect_acdc_slices
from .mnm import collect_mnm1_slices, collect_mnm2_slices


def _as_list(value, default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_source_name(name: str) -> str:
    text = str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "mm1": "mnm1",
        "m&m1": "mnm1",
        "m&ms1": "mnm1",
        "mms1": "mnm1",
        "mnms1": "mnm1",
        "mm2": "mnm2",
        "m&m2": "mnm2",
        "m&ms2": "mnm2",
        "mms2": "mnm2",
        "mnms2": "mnm2",
    }
    return aliases.get(text, text)


def get_preprocessed_root(cfg: Dict) -> Path:
    paths = cfg.get("paths", {})
    output_dir = paths.get("output_dir", "outputs")
    root = paths.get("preprocessed_root")
    if root is None:
        root = paths.get("preprocessed_acdc_root")
    if root is None:
        root = Path(output_dir) / "preprocessed" / "cardiac"
    return Path(root)


def _stamp_slices(
    slices: List[Dict],
    dataset: str,
    split: str = "",
    vendor: str = "Unknown",
) -> List[Dict]:
    for sample in slices:
        sample.setdefault("dataset", dataset)
        sample.setdefault("split", split)
        sample.setdefault("vendor", vendor)
    return slices


def _collect_acdc_training(cfg: Dict) -> List[Dict]:
    data_cfg = cfg.get("data", {})
    train_slices, _ = acdc_train_val_split(
        acdc_root=cfg["paths"]["acdc_root"],
        val_ratio=data_cfg.get("acdc_val_ratio", 0.20),
        seed=cfg.get("seed", 42),
    )
    return _stamp_slices(train_slices, "ACDC", "training", "ACDC")


def _collect_source_slices(cfg: Dict, source: str, role: str) -> List[Dict]:
    data_cfg = cfg.get("data", {})
    paths = cfg.get("paths", {})
    source = _normalize_source_name(source)

    if source == "mnm1":
        default_splits = ["Training", "Validation"] if role == "train" else ["Testing"]
        split_key = "mnm1_train_splits" if role == "train" else "mnm1_val_splits"
        return collect_mnm1_slices(
            paths["mnm1_root"],
            splits=_as_list(data_cfg.get(split_key), default_splits),
            require_gt=True,
        )

    if source == "mnm2":
        return collect_mnm2_slices(paths["mnm2_root"])

    if source == "acdc":
        if role == "train":
            return _collect_acdc_training(cfg)
        split = data_cfg.get("acdc_external_split", "training")
        return _stamp_slices(
            collect_acdc_slices(paths["acdc_root"], split=split),
            "ACDC",
            split,
            "ACDC",
        )

    raise ValueError(f"Unknown dataset source '{source}'. Expected acdc, mnm1, or mnm2.")


def collect_configured_slices(cfg: Dict) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Collect training and validation slices from config.

    Backwards-compatible default:
      no data.train_datasets / data.validation_dataset -> ACDC train/val split.

    Multi-vendor default in configs/config.yaml:
      train_datasets=[mnm1, mnm2], validation_dataset=acdc.
    """
    data_cfg = cfg.get("data", {})
    explicit_train_sources = (
        data_cfg.get("train_datasets")
        or data_cfg.get("training_sources")
        or data_cfg.get("source_datasets")
    )
    explicit_val_source = data_cfg.get("validation_dataset") or data_cfg.get("val_dataset")

    if explicit_train_sources is None and explicit_val_source is None:
        train_slices, val_slices = acdc_train_val_split(
            acdc_root=cfg["paths"]["acdc_root"],
            val_ratio=data_cfg.get("acdc_val_ratio", 0.20),
            seed=cfg.get("seed", 42),
        )
        train_slices = _stamp_slices(train_slices, "ACDC", "training", "ACDC")
        val_slices = _stamp_slices(val_slices, "ACDC", "training_val", "ACDC")
        meta = {
            "train_sources": ["acdc"],
            "validation_source": "acdc_internal_split",
            "validation_split": "training",
        }
        return train_slices, val_slices, meta

    train_sources = [
        _normalize_source_name(source)
        for source in _as_list(explicit_train_sources, ["mnm1", "mnm2"])
    ]
    validation_source = _normalize_source_name(explicit_val_source or "acdc")

    train_slices: List[Dict] = []
    for source in train_sources:
        train_slices.extend(_collect_source_slices(cfg, source, role="train"))

    val_slices = _collect_source_slices(cfg, validation_source, role="val")
    meta = {
        "train_sources": train_sources,
        "validation_source": validation_source,
        "validation_split": data_cfg.get("acdc_external_split", "training")
        if validation_source == "acdc"
        else "",
    }
    return train_slices, val_slices, meta


def summarize_slices(slices: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Compact count summary for logs and preprocessing metadata."""
    return {
        "dataset": dict(Counter(str(s.get("dataset", "Unknown")) for s in slices)),
        "split": dict(Counter(str(s.get("split", "Unknown")) for s in slices)),
        "vendor": dict(Counter(str(s.get("vendor", "Unknown")) for s in slices)),
    }
