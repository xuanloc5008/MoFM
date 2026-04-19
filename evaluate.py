"""
evaluate.py
------------
Full evaluation pipeline:

  1. ACDC test set  — segmentation metrics (Dice, HD95, ASD) + clinical metrics
  2. M&Ms-1 test    — domain-shift invariance per scanner vendor
  3. M&Ms-2 test    — domain-shift invariance (different scanner distribution)
  4. Calibration    — ECE, reliability diagram, coverage-accuracy
  5. OOD detection  — uncertainty analysis for in-distribution vs OOD slices

Usage:
  python evaluate.py --config configs/config.yaml \
                     --checkpoint outputs/checkpoints/best_model.pth

All results are saved to outputs/eval/ as CSV and PNG.
"""
import argparse
import logging
import os
import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from src.data.acdc           import collect_acdc_slices, SliceDataset
from src.data.mnm            import collect_mnm1_slices, collect_mnm2_slices, DomainShiftDataset
from src.data.transforms     import get_val_transforms
from src.models.topo_evidential_umamba import build_model
from src.runtime             import (
    configure_torch_runtime,
    dataloader_kwargs,
    resolve_amp_dtype,
    resolve_device,
    resolve_runtime_settings,
)
from src.evaluation.metrics  import (
    compute_segmentation_metrics,
    expected_calibration_error,
    uncertainty_error_correlation,
    coverage_accuracy_curve,
    MetricAggregator,
)
from src.evaluation.clinical_metrics import ClinicalMetricAggregator, bland_altman_stats
from src.visualization.plots import (
    plot_dice_violin,
    plot_uncertainty_map,
    plot_reliability_diagram,
    plot_coverage_accuracy,
    plot_domain_invariance,
    plot_ood_uncertainty_comparison,
    plot_clinical_summary,
    plot_persistence_diagram,
)
from src.topology.persistence import compute_persistence_diagram

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "evaluate.log")),
        ],
    )


@torch.no_grad()
def run_inference_batch(
    model:      torch.nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    num_classes: int,
    non_blocking: bool = False,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    collect_uncertainty: bool = True,
) -> dict:
    """
    Run model inference over a DataLoader.

    Returns dict:
      seg_records       : list of per-slice metric dicts
      dice_records      : list of {dice_c1, dice_c2, dice_c3}
      all_probs         : (N_flat, K)  flat probability array for ECE
      all_labels        : (N_flat,)    flat label array for ECE
      all_uncertainty   : (N_flat,)    flat uncertainty array
      all_is_correct    : (N_flat,)    bool
      sample_tuples     : list of (image, pred_seg, gt_seg, uncertainty) for plots
      clinical_agg      : ClinicalMetricAggregator
    """
    metric_agg   = MetricAggregator(num_classes)
    clinical_agg = ClinicalMetricAggregator()
    dice_records = []

    all_probs      = []
    all_labels     = []
    all_uncertainty= []
    all_is_correct = []
    sample_tuples  = []
    for batch in tqdm(dataloader, desc="Inference", leave=False):
        images  = batch["image"].to(device, non_blocking=non_blocking)
        gt_segs = batch["label"].cpu().numpy()

        with (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp and device.type == "cuda"
            else nullcontext()
        ):
            out = model(images, return_projections=False)
        probs      = out["probs"].float().cpu().numpy()          # B K H W
        seg_maps   = probs.argmax(axis=1)                # B H W
        unc_maps   = out["uncertainty"].float().cpu().numpy().squeeze(1)  # B H W

        for i in range(len(images)):
            pid      = batch.get("patient_id", ["unknown"] * len(images))[i]
            phase    = batch.get("phase",      ["unknown"] * len(images))[i]
            slc_idx  = int(batch.get("slice_idx", [-1] * len(images))[i]) \
                       if "slice_idx" in batch else -1

            if "spacing" in batch:
                sp = batch["spacing"][i]   # tensor (3,) from DataLoader
                sp2d = (float(sp[0]), float(sp[1]))
                sp3d = (float(sp[0]), float(sp[1]), float(sp[2]))
            else:
                sp2d = (1.5, 1.5)
                sp3d = (1.5, 1.5, 8.0)

            pred_seg = seg_maps[i]
            gt_seg   = gt_segs[i]
            unc_map  = unc_maps[i]
            prob_map = probs[i]    # K H W

            # Per-slice segmentation metrics (use physical 2-D spacing)
            m = compute_segmentation_metrics(pred_seg, gt_seg, num_classes, sp2d)
            m["patient_id"] = pid
            m["phase"]      = phase
            metric_agg.update(m)

            dice_records.append({
                "dice_c1": m.get("dice_c1", 0),
                "dice_c2": m.get("dice_c2", 0),
                "dice_c3": m.get("dice_c3", 0),
            })

            # Clinical metric accumulation (use actual 3-D spacing incl. slice thickness)
            clinical_agg.add_slice(pid, phase, slc_idx, pred_seg, gt_seg, sp3d)

            # Flatten for calibration
            flat_probs  = prob_map.transpose(1, 2, 0).reshape(-1, num_classes)  # HW x K
            flat_labels = gt_seg.reshape(-1)
            flat_unc    = unc_map.reshape(-1)
            flat_corr   = (pred_seg == gt_seg).reshape(-1)

            all_probs.append(flat_probs)
            all_labels.append(flat_labels)
            all_uncertainty.append(flat_unc)
            all_is_correct.append(flat_corr)

            # Collect representative samples for visualization
            if len(sample_tuples) < 5:
                img_np = images[i].cpu().numpy().squeeze()
                sample_tuples.append((img_np, pred_seg, gt_seg, unc_map))

    return {
        "metric_agg":      metric_agg,
        "dice_records":    dice_records,
        "all_probs":       np.concatenate(all_probs,      axis=0) if all_probs else np.array([]),
        "all_labels":      np.concatenate(all_labels,     axis=0) if all_labels else np.array([]),
        "all_uncertainty": np.concatenate(all_uncertainty, axis=0) if all_uncertainty else np.array([]),
        "all_is_correct":  np.concatenate(all_is_correct,  axis=0) if all_is_correct else np.array([]),
        "sample_tuples":   sample_tuples,
        "clinical_agg":    clinical_agg,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Topo-Evidential U-Mamba")
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device",     type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["acdc", "mnm1", "mnm2"],
        choices=["acdc", "mnm1", "mnm2"],
        help="Select which evaluation datasets to run.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["output_dir"]
    eval_dir   = os.path.join(output_dir, "eval")
    fig_dir    = os.path.join(output_dir, "figures")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(fig_dir,  exist_ok=True)
    setup_logging(eval_dir)

    device = resolve_device(args.device, gpu_index=args.gpu)
    runtime_settings = resolve_runtime_settings(cfg, device, gpu_index=args.gpu)
    configure_torch_runtime(device, runtime_settings)
    num_classes  = cfg["data"]["num_classes"]
    spatial_size = tuple(cfg["data"]["spatial_size"])
    preprocess_cfg = cfg.get("preprocessing", {})
    context_slices = int(cfg["data"].get("context_slices", cfg["data"].get("in_channels", 1)))
    cfg["runtime_active"] = runtime_settings
    batch_size   = runtime_settings["val_batch_size"]
    num_workers  = runtime_settings["num_workers"]
    ece_bins     = cfg["evaluation"].get("ece_bins", 15)
    amp_dtype    = resolve_amp_dtype(runtime_settings["amp_dtype"])
    if num_workers != runtime_settings["requested_num_workers"]:
        logger.warning(
            f"Overriding num_workers={runtime_settings['requested_num_workers']} "
            f"-> {num_workers} "
            f"for {device.type} on macOS to avoid DataLoader spawn stalls."
        )
    logger.info(f"Device: {device} ({runtime_settings['device_name']})")
    logger.info(
        "Runtime profile: "
        f"{' -> '.join(runtime_settings['profile_names'])} | "
        f"eval_bs={runtime_settings['val_batch_size']} "
        f"workers={runtime_settings['num_workers']} "
        f"amp={'on' if runtime_settings['use_amp'] else 'off'}"
    )
    loader_common = dataloader_kwargs(runtime_settings)

    # ── Load model ──────────────────────────────────────────────────────────
    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    allowed_prefixes = ("topology_encoder.",)
    bad_missing = [k for k in missing if not k.startswith(allowed_prefixes)]
    bad_unexpected = [k for k in unexpected if not k.startswith(allowed_prefixes)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch beyond the experimental topology encoder: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )
    if missing or unexpected:
        logger.warning(
            "Checkpoint loaded with experimental-topology compatibility mode. "
            f"missing={missing}, unexpected={unexpected}"
        )
    model = model.to(device).eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    val_tf = get_val_transforms(spatial_size, preprocess_cfg=preprocess_cfg)
    datasets_to_run = set(args.datasets)
    logger.info(f"Evaluation targets: {', '.join(args.datasets)}")
    acdc_results = None
    acdc_summary = {}

    # ══════════════════════════════════════════════════════════════════════
    #  1. ACDC TEST SET
    # ══════════════════════════════════════════════════════════════════════
    if "acdc" in datasets_to_run:
        logger.info("=== Evaluating ACDC test set ===")
        acdc_test_slices = collect_acdc_slices(
            cfg["paths"]["acdc_root"],
            split="testing",
            preprocess_cfg=preprocess_cfg,
        )
        acdc_test_ds  = SliceDataset(
            acdc_test_slices,
            transforms=val_tf,
            context_slices=context_slices,
        )
        acdc_test_dl  = DataLoader(acdc_test_ds, batch_size=batch_size,
                                   shuffle=False, **loader_common)

        acdc_results  = run_inference_batch(
            model, acdc_test_dl, device, num_classes,
            non_blocking=runtime_settings["non_blocking"],
            use_amp=runtime_settings["use_amp"],
            amp_dtype=amp_dtype,
        )
        acdc_summary  = acdc_results["metric_agg"].summary()

        # Save metrics CSV
        acdc_df = acdc_results["metric_agg"].to_dataframe()
        acdc_df.to_csv(os.path.join(eval_dir, "acdc_test_metrics.csv"), index=False)
        pd.DataFrame([acdc_summary]).to_csv(
            os.path.join(eval_dir, "acdc_test_summary.csv"), index=False
        )
        logger.info(f"ACDC test Dice (mean): {acdc_summary.get('dice_mean_mean', 0):.4f} ± "
                    f"{acdc_summary.get('dice_mean_std', 0):.4f}")
        logger.info(f"ACDC test HD95 (mean): {acdc_summary.get('hd95_mean_mean', 0):.2f}")

        # Plots
        plot_dice_violin(
            acdc_results["dice_records"],
            os.path.join(fig_dir, "acdc_dice_violin.png"),
            title="ACDC Test — Dice per Class",
        )

        for idx, (img, pred, gt, unc) in enumerate(acdc_results["sample_tuples"][:3]):
            plot_uncertainty_map(
                img, pred, gt, unc,
                os.path.join(fig_dir, f"acdc_uncertainty_sample_{idx}.png"),
                title=f"ACDC Test Sample {idx+1}",
            )

        # ── Calibration ─────────────────────────────────────────────────────
        if len(acdc_results["all_probs"]) > 0:
            ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
                acdc_results["all_probs"],
                acdc_results["all_labels"],
                n_bins=ece_bins,
            )
            logger.info(f"ACDC ECE: {ece:.4f}")
            plot_reliability_diagram(
                bin_accs, bin_confs, bin_counts, ece,
                os.path.join(fig_dir, "acdc_reliability_diagram.png"),
            )

            aurrc = uncertainty_error_correlation(
                acdc_results["all_uncertainty"],
                acdc_results["all_is_correct"],
            )
            logger.info(f"ACDC AURRC: {aurrc:.4f}")

            covs, accs = coverage_accuracy_curve(
                acdc_results["all_uncertainty"],
                acdc_results["all_is_correct"],
            )
            plot_coverage_accuracy(
                covs, accs,
                os.path.join(fig_dir, "acdc_coverage_accuracy.png"),
                baseline_acc=acdc_results["all_is_correct"].mean(),
            )

            # Save calibration CSV
            pd.DataFrame({
                "ece": [ece], "aurrc": [aurrc],
            }).to_csv(os.path.join(eval_dir, "acdc_calibration.csv"), index=False)

        # ── Clinical Metrics ────────────────────────────────────────────────
        pred_clinical, gt_clinical = acdc_results["clinical_agg"].compute_all()
        if pred_clinical and gt_clinical:
            pred_df = pd.DataFrame(pred_clinical)
            gt_df   = pd.DataFrame(gt_clinical)
            pred_df.to_csv(os.path.join(eval_dir, "acdc_clinical_pred.csv"), index=False)
            gt_df.to_csv(  os.path.join(eval_dir, "acdc_clinical_gt.csv"),   index=False)

            plot_clinical_summary(
                pred_clinical, gt_clinical,
                os.path.join(fig_dir, "acdc_clinical_summary.png"),
            )

            # Bland-Altman per clinical metric
            for metric, unit in [("LV_EF_pct", "%"), ("LV_EDV_mL", "mL"),
                                  ("LV_ESV_mL", "mL"), ("Myo_mass_g", "g")]:
                common = set(pred_df["patient_id"]) & set(gt_df["patient_id"])
                if not common:
                    continue
                p_vals = pred_df[pred_df["patient_id"].isin(common)][metric].values
                g_vals = gt_df[gt_df["patient_id"].isin(common)][metric].values
                if len(p_vals) > 2:
                    ba = bland_altman_stats(p_vals, g_vals)
                    logger.info(f"{metric}: bias={ba['bias']:+.2f}, "
                                f"LoA=[{ba['loa_lower']:+.2f}, {ba['loa_upper']:+.2f}], "
                                f"r={ba['pearson_r']:.3f}")

        # ── Persistence Diagram example ─────────────────────────────────────
        if acdc_test_slices:
            sample_img = acdc_test_slices[0]["image"].squeeze()
            pds = compute_persistence_diagram(sample_img, max_dim=1)
            plot_persistence_diagram(
                pds,
                os.path.join(fig_dir, "acdc_persistence_diagram.png"),
                title="Persistence Diagram (ACDC Sample)",
            )
    else:
        logger.info("=== Skipping ACDC test set ===")

    # ══════════════════════════════════════════════════════════════════════
    #  2. M&Ms-1 DOMAIN SHIFT
    # ══════════════════════════════════════════════════════════════════════
    if "mnm1" in datasets_to_run:
        logger.info("=== Evaluating M&Ms-1 (domain shift) ===")
        try:
            mnm1_slices = collect_mnm1_slices(
                cfg["paths"]["mnm1_root"],
                splits=["Testing", "Validation"],
                preprocess_cfg=preprocess_cfg,
            )
            mnm1_ds = DomainShiftDataset(
                mnm1_slices,
                transforms=val_tf,
                context_slices=context_slices,
            )
            mnm1_dl = DataLoader(mnm1_ds, batch_size=batch_size,
                                 shuffle=False, **loader_common)

            vendor_groups = mnm1_ds.get_vendor_groups()
            vendor_dice: dict = {}

            # Evaluate per vendor
            for vendor, indices in vendor_groups.items():
                vendor_slices = [mnm1_slices[i] for i in indices]
                v_ds = DomainShiftDataset(
                    vendor_slices,
                    transforms=val_tf,
                    context_slices=context_slices,
                )
                v_dl = DataLoader(v_ds, batch_size=batch_size,
                                  shuffle=False, **loader_common)
                v_res = run_inference_batch(
                    model, v_dl, device, num_classes,
                    non_blocking=runtime_settings["non_blocking"],
                    use_amp=runtime_settings["use_amp"],
                    amp_dtype=amp_dtype,
                )
                v_sum = v_res["metric_agg"].summary()
                vendor_dice[vendor] = [r["dice_mean"] for r in v_res["dice_records"]
                                       if "dice_mean" in r]
                logger.info(f"  M&Ms-1 {vendor}: "
                            f"Dice={v_sum.get('dice_mean_mean', 0):.4f} ± "
                            f"{v_sum.get('dice_mean_std', 0):.4f}")

            mnm1_results = run_inference_batch(
                model, mnm1_dl, device, num_classes,
                non_blocking=runtime_settings["non_blocking"],
                use_amp=runtime_settings["use_amp"],
                amp_dtype=amp_dtype,
            )
            mnm1_df = mnm1_results["metric_agg"].to_dataframe()
            mnm1_df.to_csv(os.path.join(eval_dir, "mnm1_metrics.csv"), index=False)

            if vendor_dice:
                plot_domain_invariance(
                    vendor_dice,
                    os.path.join(fig_dir, "mnm1_domain_invariance.png"),
                    title="M&Ms-1 — Domain Invariance per Scanner Vendor",
                )

            # OOD: compare M&Ms-1 uncertainty vs ACDC (in-distribution)
            if acdc_results is not None:
                mnm1_unc     = mnm1_results["all_uncertainty"]
                acdc_unc_sub = acdc_results["all_uncertainty"][:len(mnm1_unc)]
                if len(mnm1_unc) > 0 and len(acdc_unc_sub) > 0:
                    plot_ood_uncertainty_comparison(
                        in_dist_unc = acdc_unc_sub,
                        ood_unc     = mnm1_unc,
                        save_path   = os.path.join(fig_dir, "domain_shift_uncertainty.png"),
                    )

        except Exception as e:
            logger.warning(f"M&Ms-1 evaluation failed: {e}")
    else:
        logger.info("=== Skipping M&Ms-1 (domain shift) ===")

    # ══════════════════════════════════════════════════════════════════════
    #  3. M&Ms-2 DOMAIN SHIFT
    # ══════════════════════════════════════════════════════════════════════
    if "mnm2" in datasets_to_run:
        logger.info("=== Evaluating M&Ms-2 (domain shift) ===")
        try:
            mnm2_slices = collect_mnm2_slices(
                cfg["paths"]["mnm2_root"],
                preprocess_cfg=preprocess_cfg,
            )
            mnm2_ds = DomainShiftDataset(
                mnm2_slices,
                transforms=val_tf,
                context_slices=context_slices,
            )
            mnm2_dl = DataLoader(mnm2_ds, batch_size=batch_size,
                                 shuffle=False, **loader_common)

            mnm2_results = run_inference_batch(
                model, mnm2_dl, device, num_classes,
                non_blocking=runtime_settings["non_blocking"],
                use_amp=runtime_settings["use_amp"],
                amp_dtype=amp_dtype,
            )
            mnm2_summary = mnm2_results["metric_agg"].summary()
            mnm2_df = mnm2_results["metric_agg"].to_dataframe()
            mnm2_df.to_csv(os.path.join(eval_dir, "mnm2_metrics.csv"), index=False)

            logger.info(f"M&Ms-2 Dice (mean): {mnm2_summary.get('dice_mean_mean', 0):.4f}")

            # Per-vendor breakdown
            vendor_groups_2 = mnm2_ds.get_vendor_groups()
            vendor_dice_2: dict = {}
            for vendor, indices in vendor_groups_2.items():
                v_slices = [mnm2_slices[i] for i in indices]
                v_ds = DomainShiftDataset(
                    v_slices,
                    transforms=val_tf,
                    context_slices=context_slices,
                )
                v_dl = DataLoader(v_ds, batch_size=batch_size, shuffle=False,
                                  **loader_common)
                v_res = run_inference_batch(
                    model, v_dl, device, num_classes,
                    non_blocking=runtime_settings["non_blocking"],
                    use_amp=runtime_settings["use_amp"],
                    amp_dtype=amp_dtype,
                )
                vendor_dice_2[vendor] = [r["dice_mean"] for r in v_res["dice_records"]
                                         if "dice_mean" in r]

            if vendor_dice_2:
                plot_domain_invariance(
                    vendor_dice_2,
                    os.path.join(fig_dir, "mnm2_domain_invariance.png"),
                    title="M&Ms-2 — Domain Invariance per Scanner Vendor",
                )

        except Exception as e:
            logger.warning(f"M&Ms-2 evaluation failed: {e}")
    else:
        logger.info("=== Skipping M&Ms-2 (domain shift) ===")

    # ══════════════════════════════════════════════════════════════════════
    #  4. Final Summary Table
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION SUMMARY")
    logger.info("=" * 60)
    if acdc_summary:
        logger.info("ACDC:")
        for k, v in acdc_summary.items():
            if "_mean" in k and v != float("inf"):
                logger.info(f"  {k}: {v:.4f}")
    else:
        logger.info("ACDC: skipped")
    logger.info("=" * 60)
    logger.info(f"All results saved to: {eval_dir}")
    logger.info(f"All figures  saved to: {fig_dir}")


if __name__ == "__main__":
    main()
