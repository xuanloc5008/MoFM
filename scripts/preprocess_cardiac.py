"""
Offline preprocessing for configured cardiac datasets.

Builds deterministic slice caches with topology vectors for the dataset setup
defined in configs/config.yaml. The default config uses:
  train: M&Ms-1 + M&Ms-2
  val:   ACDC external validation

Usage:
  python scripts/preprocess_cardiac.py --config configs/config.yaml --overwrite
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from monai.transforms import Compose
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.sources import (
    collect_configured_slices,
    get_preprocessed_root,
    summarize_slices,
)
from src.data.transforms import get_preprocessing_transforms
from src.topology.persistence import compute_topology_vector


def _safe_token(value) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return token.strip("_") or "unknown"


def _preprocess_sample(sample: dict, transform: Compose, topo_cfg: dict) -> dict:
    processed = transform({
        "image": sample["image"].copy(),
        "label": sample["label"][np.newaxis].copy(),
    })

    label = processed["label"]
    if getattr(label, "ndim", 0) == 3 and label.shape[0] == 1:
        label = label[0]

    topo_vec = compute_topology_vector(
        processed["image"],
        max_dim=int(topo_cfg.get("max_dimension", 1)),
        threshold=float(topo_cfg.get("pd_threshold", 0.02)),
        downsample_size=int(topo_cfg.get("cache_downsample_size", 64)),
        top_k=int(topo_cfg.get("cache_top_k", 8)),
    )

    return {
        "image": np.asarray(processed["image"], dtype=np.float32),
        "label": np.asarray(label, dtype=np.int16),
        "topo_vec": np.asarray(topo_vec, dtype=np.float32),
        "patient_id": sample.get("patient_id", ""),
        "phase": sample.get("phase", ""),
        "slice_idx": int(sample.get("slice_idx", -1)),
        "spacing": np.asarray(sample.get("spacing", [1.5, 1.5, 8.0]), dtype=np.float32),
        "group": sample.get("group", ""),
        "dataset": sample.get("dataset", ""),
        "vendor": sample.get("vendor", "Unknown"),
        "split": sample.get("split", ""),
    }


def _clear_split(split_dir: Path):
    for existing in split_dir.glob("*.npz"):
        existing.unlink()
    index_path = split_dir / "index.json"
    if index_path.exists():
        index_path.unlink()


def _write_split(
    split_name: str,
    slices: list,
    output_dir: Path,
    transform: Compose,
    topo_cfg: dict,
    overwrite: bool,
):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        _clear_split(split_dir)

    index = []
    for i, sample in enumerate(tqdm(slices, desc=f"Preprocess {split_name}", leave=False)):
        item = _preprocess_sample(sample, transform, topo_cfg)
        filename = (
            f"{i:06d}_{_safe_token(item['dataset'])}_"
            f"{_safe_token(item['patient_id'])}_{_safe_token(item['phase'])}_"
            f"s{item['slice_idx']:02d}.npz"
        )
        np.savez_compressed(
            split_dir / filename,
            image=item["image"],
            label=item["label"],
            topo_vec=item["topo_vec"],
            patient_id=np.str_(item["patient_id"]),
            phase=np.str_(item["phase"]),
            slice_idx=np.int32(item["slice_idx"]),
            spacing=item["spacing"],
            group=np.str_(item["group"]),
            dataset=np.str_(item["dataset"]),
            vendor=np.str_(item["vendor"]),
            split=np.str_(item["split"]),
        )
        index.append({
            "file": filename,
            "patient_id": item["patient_id"],
            "phase": item["phase"],
            "slice_idx": item["slice_idx"],
            "group": item["group"],
            "dataset": item["dataset"],
            "vendor": item["vendor"],
            "split": item["split"],
            "topo_dim": int(item["topo_vec"].shape[0]),
        })

    with open(split_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Preprocess configured cardiac slices to disk")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_root = get_preprocessed_root(cfg)
    output_root.mkdir(parents=True, exist_ok=True)

    spatial_size = tuple(cfg["data"]["spatial_size"])
    transform = Compose(get_preprocessing_transforms(spatial_size))
    topo_cfg = cfg.get("topology", {})

    train_slices, val_slices, source_meta = collect_configured_slices(cfg)

    _write_split("train", train_slices, output_root, transform, topo_cfg, overwrite=args.overwrite)
    _write_split("val", val_slices, output_root, transform, topo_cfg, overwrite=args.overwrite)

    meta = {
        "spatial_size": list(spatial_size),
        "seed": int(cfg.get("seed", 42)),
        "source_meta": source_meta,
        "train_slices": len(train_slices),
        "val_slices": len(val_slices),
        "train_summary": summarize_slices(train_slices),
        "val_summary": summarize_slices(val_slices),
        "topo_max_dimension": int(topo_cfg.get("max_dimension", 1)),
        "topo_pd_threshold": float(topo_cfg.get("pd_threshold", 0.02)),
        "topo_cache_downsample_size": int(topo_cfg.get("cache_downsample_size", 64)),
        "topo_cache_top_k": int(topo_cfg.get("cache_top_k", 8)),
    }
    with open(output_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved preprocessed cardiac cache to {output_root}")
    print(f"Train slices: {len(train_slices)} | Val slices: {len(val_slices)}")
    print(f"Train summary: {summarize_slices(train_slices)}")
    print(f"Val summary: {summarize_slices(val_slices)}")


if __name__ == "__main__":
    main()
