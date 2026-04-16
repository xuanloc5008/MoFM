"""
verify_data.py  –  verify all three dataset parsers.

Usage:
    python verify_data.py --data_root /Volumes/Transcend
"""
import sys, argparse, numpy as np
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))

def check_nifti(path):
    try:
        import nibabel as nib
        img = nib.load(path)
        return {"ok": True, "shape": img.shape,
                "dtype": str(img.get_data_dtype()),
                "spacing": tuple(round(float(z),2) for z in img.header.get_zooms())}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def show_sample(s):
    print(f"    patient   : {s['patient']}   phase: {s['phase']}")
    print(f"    pathology : {s.get('pathology','?')}   "
          f"vendor: {s.get('vendor','—')}")
    info = check_nifti(s["image"])
    if info["ok"]:
        ndim = len(info["shape"])
        tag  = "✓ 3-D ready" if ndim == 3 else f"⚠ {ndim}-D (frame will be extracted)"
        print(f"    image     : {info['shape']}  spacing={info['spacing']}  {tag}")
    else:
        print(f"    image     : ✗ {info['error']}")
    if "label" in s:
        li = check_nifti(s["label"])
        if li["ok"]:
            import nibabel as nib
            uniq = sorted(set(np.unique(
                nib.load(s["label"]).get_fdata()).astype(int).tolist()))
            print(f"    label     : {li['shape']}  classes={uniq}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Folder containing ACDC-dataset, M&M1, MnM2")
    args = ap.parse_args()
    root = Path(args.data_root)

    print("="*70)
    print(f"  Dataset Verification   data_root = {root}")
    print("="*70)

    from data.preprocessing import (parse_acdc, parse_mms, parse_mms2,
                                     _load_mms_csv)
    any_ok = False

    # ── ACDC ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n[ACDC]  {root/'ACDC-dataset/database'}")
    samples = parse_acdc(str(root))
    if samples:
        any_ok = True
        pts   = {s["patient"] for s in samples}
        patho = Counter(s.get("pathology") for s in samples)
        print(f"  Patients: {len(pts)}   Samples (ED+ES): {len(samples)}")
        print(f"  Pathology: {dict(patho)}")
        print("  Example:"); show_sample(samples[0])
    else:
        print("  ✗ NOT FOUND")

    # ── M&Ms-1 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n[M&Ms-1]  {root/'M&M1'}")
    if (root / "M&M1").exists():
        any_ok  = True
        csv_m   = _load_mms_csv(root / "M&M1")
        print(f"  CSV entries: {len(csv_m)}")
        total   = 0
        for sp in ["train", "val", "test"]:
            sl = parse_mms(str(root), sp)
            total += len(sl)
            vendors = Counter(s.get("vendor") for s in sl)
            print(f"  {sp:5s}: {len(sl):4d} samples  vendors={dict(vendors)}")
            if sl and sp == "train":
                print("  Example (train):"); show_sample(sl[0])
        patho = Counter(s.get("pathology") for s in
                        parse_mms(str(root),"train") +
                        parse_mms(str(root),"val") +
                        parse_mms(str(root),"test"))
        print(f"  Pathology: {dict(patho)}")
    else:
        print("  ✗ NOT FOUND")

    # ── M&Ms-2 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n[M&Ms-2]  {root/'MnM2'}")
    if (root / "MnM2").exists():
        any_ok  = True
        samples = parse_mms2(str(root))
        pts     = {s["patient"] for s in samples}
        vendors = Counter(s.get("vendor") for s in samples)
        patho   = Counter(s.get("pathology") for s in samples)
        print(f"  Patients: {len(pts)}   Samples: {len(samples)}")
        print(f"  Vendors:   {dict(vendors)}")
        print(f"  Pathology: {dict(patho)}")
        if samples:
            print("  Example:"); show_sample(samples[0])
    else:
        print("  ✗ NOT FOUND")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    if any_ok:
        print("  ✓  Datasets found. Next step:")
        print(f"       python main.py --mode full --data_root {root}")
    else:
        print("  ✗  No datasets found. Check paths:")
        print("       ACDC-dataset/database/training/patient001/...")
        print("       M&M1/Testing/ M&M1/Training/Labeled/...")
        print("       MnM2/dataset/001/...")
    print("═"*70)

if __name__ == "__main__":
    main()
