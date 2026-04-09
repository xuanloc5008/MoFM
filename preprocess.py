"""
Preprocessing Pipeline for MOFMTDA-TTA.

Steps:
  1. Verify ACDC data structure
  2. Extract 2D slices from NIfTI volumes → save as .npz cache
  3. Precompute ground-truth Persistence Diagrams
  4. (Optional) Precompute MedSAM encoder features

Usage:
    python preprocess.py --data_root ./dataset/acdcdata/ACDC/database/training
    python preprocess.py --data_root ./dataset/acdcdata/ACDC/database/training --cache_encoder --sam_checkpoint ./checkpoints/medsam_vit_b.pth
"""
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple
import json
import pickle
from tqdm import tqdm
from scipy.ndimage import zoom
import warnings

warnings.filterwarnings("ignore")


# ================================================================
# Step 1: Verify data structure
# ================================================================

def verify_data(data_root: str) -> Dict:
    """Verify ACDC data structure and report statistics."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    patients = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("patient")])

    if len(patients) == 0:
        raise ValueError(f"No patient directories found in {root}")

    print(f"Found {len(patients)} patients in {root}")

    stats = {"total_patients": len(patients), "patients": [], "errors": []}

    for pdir in patients:
        pid = pdir.name
        info_file = pdir / "Info.cfg"

        if not info_file.exists():
            stats["errors"].append(f"{pid}: Info.cfg missing")
            continue

        info = parse_info_cfg(info_file)
        ed_frame = info.get("ED", None)
        es_frame = info.get("ES", None)

        if ed_frame is None or es_frame is None:
            stats["errors"].append(f"{pid}: ED/ES not found in Info.cfg")
            continue

        # Check files exist
        ok = True
        for phase, fidx in [("ED", ed_frame), ("ES", es_frame)]:
            img_path = pdir / f"{pid}_frame{fidx:02d}.nii.gz"
            gt_path = pdir / f"{pid}_frame{fidx:02d}_gt.nii.gz"
            if not img_path.exists():
                stats["errors"].append(f"{pid}: {img_path.name} missing")
                ok = False
            if not gt_path.exists():
                stats["errors"].append(f"{pid}: {gt_path.name} missing")
                ok = False

        if ok:
            # Get volume shape
            img_path = pdir / f"{pid}_frame{ed_frame:02d}.nii.gz"
            nii = nib.load(str(img_path))
            shape = nii.shape
            spacing = nii.header.get_zooms()

            stats["patients"].append({
                "id": pid,
                "group": info.get("Group", "unknown"),
                "ed_frame": ed_frame,
                "es_frame": es_frame,
                "shape": list(shape),
                "spacing": [float(s) for s in spacing],
                "n_slices": shape[2] if len(shape) >= 3 else 1,
            })

    print(f"  Valid patients: {len(stats['patients'])}")
    print(f"  Errors: {len(stats['errors'])}")
    if stats["errors"]:
        for e in stats["errors"][:5]:
            print(f"    - {e}")

    # Group distribution
    groups = {}
    for p in stats["patients"]:
        g = p["group"]
        groups[g] = groups.get(g, 0) + 1
    print(f"  Groups: {groups}")

    total_slices = sum(p["n_slices"] * 2 for p in stats["patients"])  # ED + ES
    print(f"  Total 2D slices: {total_slices}")

    return stats


def parse_info_cfg(info_file: Path) -> Dict:
    """Parse ACDC Info.cfg file."""
    info = {}
    with open(info_file, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                # Parse types
                if key in ["ED", "ES", "NbFrame"]:
                    info[key] = int(val)
                elif key in ["Height", "Weight"]:
                    info[key] = float(val)
                else:
                    info[key] = val
    return info


# ================================================================
# Step 2: Extract 2D slices
# ================================================================

def extract_slices(data_root: str, output_dir: str, img_size: Tuple[int, int] = (256, 256),
                   train_ratio: float = 0.7, val_ratio: float = 0.1):
    """Extract 2D slices from NIfTI, normalize, resize, save as .npz cache."""
    root = Path(data_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    patients = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("patient")])

    # Split patients
    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": patients[:n_train],
        "val": patients[n_train:n_train + n_val],
        "test": patients[n_train + n_val:],
    }

    split_stats = {}

    for split_name, split_patients in splits.items():
        split_dir = out / split_name
        split_dir.mkdir(exist_ok=True)

        slice_count = 0
        metadata = []

        for pdir in tqdm(split_patients, desc=f"Extracting {split_name}"):
            pid = pdir.name
            info = parse_info_cfg(pdir / "Info.cfg")
            ed_frame = info["ED"]
            es_frame = info["ES"]

            for phase, fidx in [("ED", ed_frame), ("ES", es_frame)]:
                img_path = pdir / f"{pid}_frame{fidx:02d}.nii.gz"
                gt_path = pdir / f"{pid}_frame{fidx:02d}_gt.nii.gz"

                if not img_path.exists() or not gt_path.exists():
                    continue

                img_nii = nib.load(str(img_path))
                gt_nii = nib.load(str(gt_path))
                img_vol = img_nii.get_fdata()
                gt_vol = gt_nii.get_fdata().astype(np.int32)
                spacing = img_nii.header.get_zooms()

                n_slices = img_vol.shape[2]

                for s in range(n_slices):
                    img_2d = img_vol[:, :, s]
                    gt_2d = gt_vol[:, :, s]
                    z_norm = s / max(n_slices - 1, 1)

                    # Resize
                    img_2d = resize_image(img_2d, img_size)
                    gt_2d = resize_mask(gt_2d, img_size)

                    # Normalize to [0, 1] (MedSAM format)
                    img_2d = normalize_01(img_2d)

                    # Unique ID
                    slice_id = f"{pid}_{phase}_slice{s:02d}"

                    # Save
                    np.savez_compressed(
                        split_dir / f"{slice_id}.npz",
                        image=img_2d.astype(np.float32),
                        mask=gt_2d.astype(np.int8),
                    )

                    metadata.append({
                        "slice_id": slice_id,
                        "patient_id": pid,
                        "phase": phase,
                        "frame_idx": fidx,
                        "slice_idx": s,
                        "z_norm": round(z_norm, 4),
                        "n_slices": n_slices,
                        "group": info.get("Group", "unknown"),
                        "pixel_spacing": [float(spacing[0]), float(spacing[1])],
                        "slice_thickness": float(spacing[2]) if len(spacing) > 2 else 10.0,
                        "original_shape": list(img_vol.shape[:2]),
                    })
                    slice_count += 1

        # Save metadata
        with open(split_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        split_stats[split_name] = {
            "patients": len(split_patients),
            "slices": slice_count,
        }
        print(f"  {split_name}: {len(split_patients)} patients, {slice_count} slices")

    # Save split info
    with open(out / "split_info.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    return split_stats


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape
    return zoom(img, (size[0] / h, size[1] / w), order=3)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    return zoom(mask, (size[0] / h, size[1] / w), order=0).astype(np.int32)


def normalize_01(img: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] — required by MedSAM."""
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


# ================================================================
# Step 3: Precompute Persistence Diagrams
# ================================================================

def precompute_pds(cache_dir: str):
    """Precompute ground-truth PDs for contrastive learning."""
    from data.topology import compute_persistence_diagram

    cache = Path(cache_dir)
    structure_map = {"LV": 1, "Myo": 2, "RV": 3}

    for split in ["train", "val", "test"]:
        split_dir = cache / split
        if not split_dir.exists():
            continue

        pd_dir = split_dir / "persistence_diagrams"
        pd_dir.mkdir(exist_ok=True)

        npz_files = sorted(split_dir.glob("*.npz"))
        print(f"Computing PDs for {split}: {len(npz_files)} slices")

        for npz_path in tqdm(npz_files, desc=f"PDs {split}"):
            data = np.load(npz_path)
            mask = data["mask"]

            pds = {}
            for name, idx in structure_map.items():
                binary = (mask == idx).astype(float)
                if binary.sum() > 0:
                    pd = compute_persistence_diagram(binary)
                    pds[name] = [(float(b), float(d), int(dim)) for b, d, dim in pd]
                else:
                    pds[name] = []

            pd_file = pd_dir / f"{npz_path.stem}_pd.pkl"
            with open(pd_file, "wb") as f:
                pickle.dump(pds, f)

    print("PD precomputation complete.")


# ================================================================
# Step 4: Precompute MedSAM encoder features
# ================================================================

def precompute_encoder_features(cache_dir: str, sam_checkpoint: str,
                                model_type: str = "vit_b",
                                batch_size: int = 8):
    """Cache frozen MedSAM encoder features for all slices."""
    import torch

    cache = Path(cache_dir)

    # Load MedSAM encoder
    print(f"Loading MedSAM encoder from {sam_checkpoint}...")
    try:
        from segment_anything import sam_model_registry
        sam = sam_model_registry[model_type]()
        ckpt = torch.load(sam_checkpoint, map_location="cpu")

        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            encoder_keys = {k.replace("image_encoder.", ""): v
                           for k, v in state.items()
                           if k.startswith("image_encoder.")}
            if encoder_keys:
                sam.image_encoder.load_state_dict(encoder_keys, strict=False)
            else:
                sam.load_state_dict(state, strict=False)

        encoder = sam.image_encoder
    except Exception as e:
        print(f"Failed to load MedSAM: {e}")
        print("Falling back to random encoder for testing.")
        encoder = None
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval()

    for split in ["train", "val", "test"]:
        split_dir = cache / split
        if not split_dir.exists():
            continue

        feat_dir = split_dir / "encoder_features"
        feat_dir.mkdir(exist_ok=True)

        npz_files = sorted(split_dir.glob("*.npz"))
        print(f"Encoding {split}: {len(npz_files)} slices")

        for i in tqdm(range(0, len(npz_files), batch_size), desc=f"Encode {split}"):
            batch_files = npz_files[i:i + batch_size]
            images = []
            for f in batch_files:
                data = np.load(f)
                img = data["image"]  # (H, W), [0, 1]
                # MedSAM expects (3, H, W) — replicate grayscale to 3 channels
                img_3ch = np.stack([img, img, img], axis=0)  # (3, H, W)
                images.append(img_3ch)

            batch_tensor = torch.tensor(np.stack(images), dtype=torch.float32).to(device)

            with torch.no_grad():
                features = encoder(batch_tensor)  # (B, 256, H/16, W/16)

            for j, f in enumerate(batch_files):
                feat = features[j].cpu().numpy()  # (256, 16, 16)
                feat_file = feat_dir / f"{f.stem}_feat.npy"
                np.save(feat_file, feat)

    total_size = sum(
        f.stat().st_size for split in ["train", "val", "test"]
        for f in (cache / split / "encoder_features").glob("*.npy")
        if (cache / split / "encoder_features").exists()
    )
    print(f"Encoder feature caching complete. Total size: {total_size / 1e9:.2f} GB")


# ================================================================
# Step 5: Summary
# ================================================================

def print_summary(cache_dir: str):
    cache = Path(cache_dir)
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        split_dir = cache / split
        if not split_dir.exists():
            continue
        n_slices = len(list(split_dir.glob("*.npz")))
        has_pds = (split_dir / "persistence_diagrams").exists()
        has_feats = (split_dir / "encoder_features").exists()
        n_pds = len(list((split_dir / "persistence_diagrams").glob("*.pkl"))) if has_pds else 0
        n_feats = len(list((split_dir / "encoder_features").glob("*.npy"))) if has_feats else 0

        print(f"\n  {split}:")
        print(f"    Slices:     {n_slices}")
        print(f"    PDs:        {n_pds} {'✓' if n_pds == n_slices else '✗'}")
        print(f"    Features:   {n_feats} {'✓' if n_feats == n_slices else '—'}")

    print()


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="MOFMTDA-TTA Preprocessing")
    parser.add_argument("--data_root", type=str,
                        default="./dataset/acdcdata/ACDC/database/training",
                        help="Path to ACDC training directory")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Output directory for cached data")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--cache_encoder", action="store_true",
                        help="Precompute MedSAM encoder features (requires GPU)")
    parser.add_argument("--sam_checkpoint", type=str,
                        default="./checkpoints/medsam_vit_b.pth")
    parser.add_argument("--skip_pds", action="store_true",
                        help="Skip PD precomputation (faster, PDs computed on-the-fly)")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)

    # Step 1: Verify
    print("\n[Step 1/4] Verifying data structure...")
    stats = verify_data(args.data_root)

    # Step 2: Extract slices
    print("\n[Step 2/4] Extracting 2D slices...")
    extract_slices(args.data_root, args.cache_dir, img_size)

    # Step 3: PDs
    if not args.skip_pds:
        print("\n[Step 3/4] Precomputing Persistence Diagrams...")
        try:
            precompute_pds(args.cache_dir)
        except ImportError:
            print("  SKIPPED: gudhi not installed. PDs will be computed on-the-fly.")
    else:
        print("\n[Step 3/4] Skipped PD precomputation.")

    # Step 4: Encoder features
    if args.cache_encoder:
        print("\n[Step 4/4] Caching MedSAM encoder features...")
        precompute_encoder_features(
            args.cache_dir, args.sam_checkpoint,
            batch_size=args.batch_size)
    else:
        print("\n[Step 4/4] Skipped encoder caching (use --cache_encoder to enable).")

    # Summary
    print_summary(args.cache_dir)
    print("Done! You can now run: python run.py --phase 1")


if __name__ == "__main__":
    main()
