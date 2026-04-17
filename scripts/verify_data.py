"""
verify_data.py
--------------
Pre-flight checker: verifies all three dataset directory structures
BEFORE starting training, giving clear error messages for each issue.

Usage:
    python scripts/verify_data.py --config configs/config.yaml

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""
import argparse
import sys
import os
import configparser
import logging

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check(cond: bool, msg_pass: str, msg_fail: str) -> bool:
    if cond:
        logger.info(f"  {PASS} {msg_pass}")
    else:
        logger.error(f"  {FAIL} {msg_fail}")
    return cond


def warn(cond: bool, msg_ok: str, msg_warn: str) -> bool:
    if cond:
        logger.info(f"  {PASS} {msg_ok}")
    else:
        logger.warning(f"  {WARN} {msg_warn}")
    return cond


def probe_nifti(path: Path) -> Tuple[bool, str]:
    """Try to load a NIfTI file header; return (ok, message)."""
    try:
        img  = nib.load(str(path))
        shp  = img.shape
        sp   = tuple(round(float(x), 2) for x in img.header.get_zooms()[:3])
        return True, f"shape={shp}  spacing={sp}"
    except Exception as exc:
        return False, str(exc)


def visible_csvs(root: Path) -> List[Path]:
    """Skip hidden metadata sidecars such as macOS ._* files."""
    return sorted(
        path for path in root.glob("*.csv")
        if not path.name.startswith(".")
    )


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Load CSVs robustly across UTF-8 variants and legacy encodings."""
    last_exc = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise last_exc if last_exc is not None else RuntimeError(f"Failed to read {path}")


def resolve_acdc_database_dir(root: Path) -> Path:
    """
    Accept either the dataset root (.../ACDC-dataset) or the database directory
    itself (.../ACDC-dataset/database).
    """
    for candidate in (root, root / "database"):
        if not candidate.exists():
            continue
        if candidate.name == "database":
            return candidate
        if (candidate / "training").exists() or (candidate / "testing").exists():
            return candidate
    return root / "database"


def parse_acdc_info_file(info_path: Path) -> dict:
    """Lightweight ACDC Info.cfg parser used by the verifier only."""
    cfg = configparser.RawConfigParser()
    try:
        cfg.read(info_path, encoding="utf-8")
    except configparser.MissingSectionHeaderError:
        with open(info_path, encoding="utf-8", errors="replace") as f:
            lines = [line for line in f if line.strip() and not line.strip().startswith("#")]
        cfg.read_string("[Info]\n" + "".join(lines))

    if "Info" in cfg:
        info = dict(cfg["Info"])
    elif cfg.sections():
        info = dict(cfg[cfg.sections()[0]])
    else:
        info = {}

    return {
        "ed_frame": int(info.get("ed", 1)),
        "es_frame": int(info.get("es", 1)),
        "group": info.get("group", "unknown"),
        "nb_frames": int(info.get("nbframe", info.get("nb_frames", 1))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ACDC verifier
# ─────────────────────────────────────────────────────────────────────────────

def verify_acdc(acdc_root: str) -> bool:
    logger.info("=" * 55)
    logger.info(f"ACDC  →  {acdc_root}")
    logger.info("=" * 55)
    root = Path(acdc_root)
    ok   = True

    # Top-level structure
    ok &= check(root.exists(),
                f"Root exists: {root}",
                f"Root NOT found: {root}")
    db = resolve_acdc_database_dir(root)
    if db == root:
        logger.info(f"  {PASS} Using database root directly: {db}")
    else:
        ok &= check(db.exists(),
                    "database/ subdirectory found",
                    f"database/ NOT found inside {root}")
    if not ok:
        return False

    for split in ("training", "testing"):
        split_dir = db / split
        if not check(split_dir.exists(),
                     f"database/{split}/ found",
                     f"database/{split}/ NOT found"):
            ok = False
            continue

        patient_dirs = sorted(
            p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        logger.info(f"  → {split}: {len(patient_dirs)} patient directories found")
        if not patient_dirs:
            logger.warning(f"  {WARN} No patient subdirectories in {split_dir}")
            continue

        # Probe first 3 patients
        for pdir in patient_dirs[:3]:
            pid      = pdir.name
            info_cfg = pdir / "Info.cfg"

            if not check(info_cfg.exists(),
                         f"{pid}/Info.cfg found",
                         f"{pid}/Info.cfg MISSING"):
                ok = False
                continue

            # Parse Info.cfg
            try:
                info = parse_acdc_info_file(info_cfg)
                ed   = info["ed_frame"]
                es   = info["es_frame"]
                logger.info(f"    {pid}: ED={ed}, ES={es}, "
                            f"group={info['group']}, nb_frames={info['nb_frames']}")
            except Exception as exc:
                logger.error(f"  {FAIL} {pid}/Info.cfg parse error: {exc}")
                ok = False
                continue

            # Check expected NIfTI files
            for phase, fidx in [("ED", ed), ("ES", es)]:
                for suffix in ("", "_gt"):
                    fname = pdir / f"{pid}_frame{fidx:02d}{suffix}.nii.gz"
                    if check(fname.exists(),
                             f"{pid}_frame{fidx:02d}{suffix}.nii.gz found",
                             f"{pid}_frame{fidx:02d}{suffix}.nii.gz MISSING"):
                        nii_ok, nii_msg = probe_nifti(fname)
                        if nii_ok:
                            logger.info(f"      {fname.name}: {nii_msg}")
                        else:
                            logger.error(f"  {FAIL} Cannot read {fname.name}: {nii_msg}")
                            ok = False
                    else:
                        ok = False

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# M&Ms-1 verifier
# ─────────────────────────────────────────────────────────────────────────────

def verify_mnm1(mnm1_root: str) -> bool:
    logger.info("=" * 55)
    logger.info(f"M&Ms-1  →  {mnm1_root}")
    logger.info("=" * 55)
    root = Path(mnm1_root)
    ok   = True

    ok &= check(root.exists(),
                f"Root exists: {root}",
                f"Root NOT found: {root}")
    if not ok:
        return False

    # CSV
    csvs = visible_csvs(root)
    if warn(len(csvs) > 0,
            f"CSV found: {csvs[0].name}" if csvs else "",
            "No CSV found at root — ED/ES will fall back to frame 0/mid"):
        df = read_csv_with_fallback(csvs[0])
        df.columns = [c.strip() for c in df.columns]
        logger.info(f"    CSV columns: {list(df.columns)}")
        for col in ["ED", "ES"]:
            warn(col in df.columns,
                 f"CSV has '{col}' column",
                 f"CSV missing '{col}' column — frame index fallback used")
        for col in ["VendorName", "Vendor"]:
            if col in df.columns:
                vendors = sorted(df[col].dropna().unique())
                logger.info(f"    Vendors in CSV ({col}): {vendors}")
                break

    # Splits
    from src.data.mnm import _MNM1_SPLIT_MAP
    for split_name, rel_path in _MNM1_SPLIT_MAP.items():
        split_dir = root / rel_path
        if split_dir.exists():
            subject_dirs = [
                p for p in split_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
            logger.info(f"  {PASS} {split_name:25s} → {rel_path:30s}  "
                        f"({len(subject_dirs)} subjects)")

            # Probe first subject
            for sdir in subject_dirs[:1]:
                sid    = sdir.name
                img_p  = sdir / f"{sid}_sa.nii.gz"
                gt_p   = sdir / f"{sid}_sa_gt.nii.gz"
                if check(img_p.exists(),
                         f"{sid}_sa.nii.gz found",
                         f"{sid}_sa.nii.gz MISSING"):
                    nii_ok, msg = probe_nifti(img_p)
                    if nii_ok:
                        ndim = nib.load(str(img_p)).ndim
                        logger.info(f"      {img_p.name}: {msg}  (ndim={ndim}, "
                                    f"expect 4 for CINE)")
                        warn(ndim == 4,
                             "Volume is 4-D (H,W,Z,T) as expected",
                             f"Volume is {ndim}-D — expected 4-D CINE volume")
                    else:
                        logger.error(f"  {FAIL} Cannot read {img_p.name}: {msg}")
                        ok = False
                if split_name not in ("Training_Unlabeled",):
                    warn(gt_p.exists(),
                         f"{sid}_sa_gt.nii.gz found",
                         f"{sid}_sa_gt.nii.gz MISSING (labelled split should have GT)")
        else:
            logger.info(f"  — {split_name:25s} → {rel_path}  (not present, skipped)")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# M&Ms-2 verifier
# ─────────────────────────────────────────────────────────────────────────────

def verify_mnm2(mnm2_root: str) -> bool:
    logger.info("=" * 55)
    logger.info(f"MnM2  →  {mnm2_root}")
    logger.info("=" * 55)
    root = Path(mnm2_root)
    ok   = True

    ok &= check(root.exists(),
                f"Root exists: {root}",
                f"Root NOT found: {root}")
    if not ok:
        return False

    # dataset_information.csv
    info_csv = root / "dataset_information.csv"
    if warn(info_csv.exists(),
            "dataset_information.csv found",
            "dataset_information.csv MISSING — vendor info unavailable"):
        df = read_csv_with_fallback(info_csv)
        df.columns = [c.strip() for c in df.columns]
        logger.info(f"    CSV columns: {list(df.columns)}")
        # Show vendor distribution
        for col in ["VendorName", "Vendor", "VENDOR"]:
            if col in df.columns:
                counts = df[col].value_counts().to_dict()
                logger.info(f"    Vendor counts ({col}): {counts}")
                break

    # dataset/ dir
    dataset_dir = root / "dataset"
    ok &= check(dataset_dir.exists(),
                "dataset/ subdirectory found",
                f"dataset/ NOT found in {root}")
    if not ok:
        return False

    subject_dirs = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    logger.info(f"  → dataset/: {len(subject_dirs)} subject directories")

    probed, missing_gt, unexpected_3d = 0, 0, 0
    for sdir in subject_dirs[:5]:
        sid = sdir.name
        for phase, img_name, gt_name in [
            ("ED", f"{sid}_SA_ED.nii.gz",    f"{sid}_SA_ED_gt.nii.gz"),
            ("ES", f"{sid}_SA_ES.nii.gz",    f"{sid}_SA_ES_gt.nii.gz"),
        ]:
            img_p = sdir / img_name
            gt_p  = sdir / gt_name

            if check(img_p.exists(),
                     f"{sid}/{img_name} found",
                     f"{sid}/{img_name} MISSING"):
                nii_ok, msg = probe_nifti(img_p)
                if nii_ok:
                    ndim = nib.load(str(img_p)).ndim
                    logger.info(f"      {img_name}: {msg}  (ndim={ndim}, expect 3)")
                    if ndim != 3:
                        logger.warning(f"  {WARN} Expected 3-D, got {ndim}-D for {img_name}")
                        unexpected_3d += 1
                    probed += 1
                else:
                    logger.error(f"  {FAIL} Cannot read {img_name}: {msg}")
                    ok = False
            else:
                ok = False

            if not warn(gt_p.exists(),
                        f"{sid}/{gt_name} found",
                        f"{sid}/{gt_name} MISSING"):
                missing_gt += 1

        # Confirm CINE / LA files are present but ignored
        cine = list(sdir.glob(f"{sid}_SA_CINE*"))
        la   = list(sdir.glob(f"{sid}_LA_*"))
        if cine:
            logger.info(f"      {PASS} CINE file present ({cine[0].name}) — correctly ignored")
        if la:
            logger.info(f"      {PASS} LA file(s) present — correctly ignored")

    if missing_gt > 0:
        logger.warning(f"  {WARN} {missing_gt} GT files missing across first 5 subjects")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify dataset directory structures")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--acdc",   default=None, help="Override ACDC root path")
    parser.add_argument("--mnm1",   default=None, help="Override M&Ms-1 root path")
    parser.add_argument("--mnm2",   default=None, help="Override MnM2 root path")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    acdc_root = args.acdc or cfg["paths"]["acdc_root"]
    mnm1_root = args.mnm1 or cfg["paths"]["mnm1_root"]
    mnm2_root = args.mnm2 or cfg["paths"]["mnm2_root"]

    results = {}
    results["ACDC"]   = verify_acdc(acdc_root)
    results["M&Ms-1"] = verify_mnm1(mnm1_root)
    results["MnM2"]   = verify_mnm2(mnm2_root)

    logger.info("")
    logger.info("=" * 55)
    logger.info("SUMMARY")
    logger.info("=" * 55)
    all_ok = True
    for name, passed in results.items():
        icon = PASS if passed else FAIL
        logger.info(f"  {icon} {name}")
        all_ok = all_ok and passed

    if all_ok:
        logger.info("\n✅ All checks passed — ready to train.")
    else:
        logger.error("\n❌ Some checks FAILED — fix issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
