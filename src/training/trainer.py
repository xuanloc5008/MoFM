"""
Training Engine
----------------
Implements the end-to-end training loop for Topo-Evidential U-Mamba.

Each batch:
  1. Forward pass → evidence, projections
  2. Load cached topology vectors for the mini-batch
  3. Compute L_Total = Σ_p L_EDL(p) + β * L_Dice + γ * L_PD-SCon
  4. Backward + optimiser step
  5. Log metrics to TensorBoard

Validation:
  - Forward pass without augmentation
  - Compute Dice, HD95 per class
  - Save best checkpoint by val_dice_mean
"""
import os
import time
import logging
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from ..models.topo_evidential_umamba import TopoEvidentialUMamba
from ..losses.edl_loss import EDLLoss, SoftDiceLoss, get_lambda_t
from ..losses.contrastive_loss import PDSupervisedContrastiveLoss
from ..losses.topology_loss import TopologyAlignmentLoss
from ..evaluation.metrics import compute_segmentation_metrics, MetricAggregator
from ..runtime import resolve_amp_dtype

logger = logging.getLogger(__name__)


class _NullSummaryWriter:
    """Fallback writer when tensorboard is not installed."""

    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


class Trainer:
    def __init__(
        self,
        model:       TopoEvidentialUMamba,
        train_loader: DataLoader,
        val_loader:  DataLoader,
        cfg:         dict,
        device:      torch.device,
        runtime_cfg: dict,
        output_dir:  str,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.runtime_cfg  = runtime_cfg
        self.output_dir   = output_dir

        train_cfg = cfg.get("training", {})
        loss_cfg  = cfg.get("loss",     {})
        data_cfg  = cfg.get("data",     {})
        topo_cfg  = cfg.get("topology", {})
        topo_exp_cfg = cfg.get("topology_experimental", {})

        self.epochs       = train_cfg.get("epochs",        200)
        self.val_interval = train_cfg.get("val_interval",  5)
        self.grad_clip    = train_cfg.get("grad_clip",     1.0)
        self.num_classes  = data_cfg.get("num_classes",    4)

        self.gamma        = loss_cfg.get("gamma",          0.5)
        self.dice_weight  = float(loss_cfg.get("dice_weight", 0.0))
        self.temperature  = loss_cfg.get("temperature",    0.1)
        self.edl_class_weights = loss_cfg.get("edl_class_weights")
        self.dice_include_background = bool(loss_cfg.get("dice_include_background", False))
        self.dice_class_weights = loss_cfg.get("dice_class_weights")
        self.lambda_sched = loss_cfg.get("lambda_t_schedule", {})
        self.topo_threshold = topo_cfg.get("topo_positive_threshold", 0.6)
        self.topo_positive_top_k = topo_cfg.get("topo_positive_top_k")
        self.topology_exp_enabled = bool(topo_exp_cfg.get("enabled", False))
        self.topo_align_weight = float(topo_exp_cfg.get("align_weight", 0.0))
        self.topo_align_start_epoch = int(topo_exp_cfg.get("align_start_epoch", 0))
        self.topo_align_confidence_gated = bool(topo_exp_cfg.get("confidence_gated", True))
        self.non_blocking = runtime_cfg.get("non_blocking", False)
        self.use_amp = runtime_cfg.get("use_amp", False) and device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(runtime_cfg.get("amp_dtype", "float32"))
        grad_scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
        if grad_scaler_cls is not None:
            try:
                self.scaler = grad_scaler_cls("cuda", enabled=self.use_amp)
            except TypeError:
                self.scaler = grad_scaler_cls(enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Loss functions
        self.edl_loss = EDLLoss(class_weights=self.edl_class_weights)
        self.dice_loss = SoftDiceLoss(
            include_background=self.dice_include_background,
            class_weights=self.dice_class_weights,
        )
        self.con_loss = PDSupervisedContrastiveLoss(
            temperature=self.temperature,
            positive_threshold=self.topo_threshold,
            positive_top_k=self.topo_positive_top_k,
        )
        self.topo_align_loss = TopologyAlignmentLoss()

        # Optimiser
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )

        # LR scheduler: linear warmup + cosine anneal
        warmup = train_cfg.get("warmup_epochs", 10)
        remaining = self.epochs - warmup
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=remaining, eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup],
        )

        # Logging
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        if SummaryWriter is None:
            logger.warning("tensorboard is not installed; continuing without TensorBoard logging.")
            self.writer = _NullSummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir)

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir

        self.best_val_dice = 0.0
        self.best_val_metrics: Dict[str, float] = {
            "val_dice_mean": 0.0,
            "val_dice_rv": 0.0,
            "val_dice_myo": 0.0,
            "val_dice_lv": 0.0,
        }
        self.best_train_metrics: Dict[str, float] = {
            "total": float("inf"),
            "edl": float("inf"),
            "dice": float("inf"),
            "con": float("inf"),
            "topo_align": float("inf"),
        }
        self.train_history: Dict[str, list] = {
            "train_loss": [], "train_edl": [], "train_dice": [], "train_con": [], "train_topo_align": [],
            "val_dice_mean": [],
            "val_dice_rv": [], "val_dice_myo": [], "val_dice_lv": [],
            "val_hd95_mean": [], "lr": [],
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Training step
    # ─────────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        agg = {"total": 0.0, "edl": 0.0, "dice": 0.0, "con": 0.0, "topo_align": 0.0}
        n   = 0

        lambda_t = get_lambda_t(
            epoch,
            self.lambda_sched.get("start", 0.0),
            self.lambda_sched.get("end",   1.0),
            self.lambda_sched.get("warmup_epochs", 20),
        )

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=self.non_blocking)
            labels = batch["label"].to(self.device, non_blocking=self.non_blocking)
            topo_vec = batch.get("topo_vec")
            if topo_vec is None:
                raise RuntimeError(
                    "Missing 'topo_vec' in training batch. Rebuild the preprocessed "
                    "dataset with scripts/preprocess_acdc.py and train with "
                    "data.use_preprocessed_acdc=true."
                )
            topo_vec = topo_vec.to(self.device, non_blocking=self.non_blocking)
            barcode_batch = None
            if self.topology_exp_enabled:
                needed = ("barcode_h0", "barcode_h0_count", "barcode_h1", "barcode_h1_count")
                missing = [k for k in needed if k not in batch]
                if missing:
                    raise RuntimeError(
                        "Experimental barcode topology is enabled but the training batch is "
                        f"missing {missing}. Rebuild the cache with "
                        "scripts/preprocess_acdc.py --overwrite."
                    )
                barcode_batch = {
                    "barcode_h0": batch["barcode_h0"].to(self.device, non_blocking=self.non_blocking),
                    "barcode_h0_count": batch["barcode_h0_count"].to(self.device, non_blocking=self.non_blocking),
                    "barcode_h1": batch["barcode_h1"].to(self.device, non_blocking=self.non_blocking),
                    "barcode_h1_count": batch["barcode_h1_count"].to(self.device, non_blocking=self.non_blocking),
                }

            with self._autocast_context():
                # ── Forward ──────────────────────────────────────────────
                out = self.model(images, return_projections=True)
                evidence    = out["evidence"]       # B K H W
                probs       = out["probs"]          # B K H W
                projections = out["projections"]    # B D

                # ── L_EDL ────────────────────────────────────────────────
                l_edl, edl_log = self.edl_loss(evidence, labels, lambda_t=lambda_t)
                l_dice = self.dice_loss(probs, labels) if self.dice_weight > 0 else probs.sum() * 0.0

                # ── L_PD-SCon ────────────────────────────────────────────
                l_con = self.con_loss(projections, topo_vec)

                # ── Experimental barcode alignment ───────────────────────
                topo_align_active = (
                    self.topology_exp_enabled
                    and barcode_batch is not None
                    and self.topo_align_weight > 0.0
                    and epoch >= self.topo_align_start_epoch
                )
                if topo_align_active:
                    topo_embed = self.model.encode_topology(barcode_batch)
                    sample_weights = None
                    if self.topo_align_confidence_gated:
                        # Confident predictions should obey topology more strongly.
                        sample_weights = (
                            1.0 - out["uncertainty"].detach().mean(dim=(1, 2, 3))
                        ).clamp_min(0.0)
                    l_topo_align = self.topo_align_loss(
                        projections,
                        topo_embed,
                        sample_weights=sample_weights,
                    )
                else:
                    l_topo_align = projections.sum() * 0.0

                # ── Total loss ───────────────────────────────────────────
                loss = (
                    l_edl
                    + self.dice_weight * l_dice
                    + self.gamma * l_con
                    + self.topo_align_weight * l_topo_align
                )

            # ── Backward ─────────────────────────────────────────────────
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            B = images.shape[0]
            agg["total"] += loss.item() * B
            agg["edl"]   += l_edl.item() * B
            agg["dice"]  += l_dice.item() * B
            agg["con"]   += l_con.item() * B
            agg["topo_align"] += l_topo_align.item() * B
            n += B

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "edl":  f"{l_edl.item():.4f}",
                "dice": f"{l_dice.item():.4f}",
                "con":  f"{l_con.item():.4f}",
                "talign": f"{l_topo_align.item():.4f}",
            })

        return {k: v / n for k, v in agg.items()}

    # ─────────────────────────────────────────────────────────────────────────
    #  Validation step
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        metric_agg = MetricAggregator(self.num_classes)

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            images  = batch["image"].to(self.device, non_blocking=self.non_blocking)
            labels  = batch["label"].cpu().numpy()
            spacing = (1.5, 1.5)   # default spacing for 2-D slice

            with self._autocast_context():
                seg_map, _ = self.model.predict(images)
            seg_map = seg_map.cpu().numpy()

            for i in range(len(images)):
                if "spacing" in batch:
                    sp = batch["spacing"][i]
                    sp2d = (float(sp[0]), float(sp[1]))
                else:
                    sp2d = (1.5, 1.5)
                m = compute_segmentation_metrics(
                    seg_map[i], labels[i], self.num_classes, sp2d
                )
                metric_agg.update(m)

        summary = metric_agg.summary()
        return {
            "val_dice_mean": summary.get("dice_mean_mean", 0.0),
            "val_hd95_mean": summary.get("hd95_mean_mean", float("inf")),
            "val_dice_rv":   summary.get("dice_c1_mean",   0.0),
            "val_dice_myo":  summary.get("dice_c2_mean",   0.0),
            "val_dice_lv":   summary.get("dice_c3_mean",   0.0),
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Main training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        logger.info(f"Starting training: {self.epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")

        for epoch in range(self.epochs):
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch(epoch)
            self.best_train_metrics["total"] = min(self.best_train_metrics["total"], train_metrics["total"])
            self.best_train_metrics["edl"] = min(self.best_train_metrics["edl"], train_metrics["edl"])
            self.best_train_metrics["dice"] = min(self.best_train_metrics["dice"], train_metrics["dice"])
            self.best_train_metrics["con"] = min(self.best_train_metrics["con"], train_metrics["con"])
            self.best_train_metrics["topo_align"] = min(
                self.best_train_metrics["topo_align"], train_metrics["topo_align"]
            )

            # Scheduler step
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            # Validate
            val_metrics = {}
            if (epoch + 1) % self.val_interval == 0 or epoch == self.epochs - 1:
                val_metrics = self._validate()

                # Save best model
                vdice = val_metrics.get("val_dice_mean", 0.0)
                for key in self.best_val_metrics:
                    self.best_val_metrics[key] = max(
                        self.best_val_metrics[key],
                        val_metrics.get(key, 0.0),
                    )
                if vdice > self.best_val_dice:
                    self.best_val_dice = vdice
                    self._save_checkpoint(epoch, val_metrics, best=True)

            # Save periodic checkpoint
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(epoch, val_metrics)

            # Logging
            elapsed = time.time() - t0
            log_msg = (
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"Loss: {train_metrics['total']:.4f} "
                f"(EDL: {train_metrics['edl']:.4f}, Dice: {train_metrics['dice']:.4f}, "
                f"Con: {train_metrics['con']:.4f}, TAlign: {train_metrics['topo_align']:.4f}) | "
                f"Best Loss: {self.best_train_metrics['total']:.4f} "
                f"(EDL: {self.best_train_metrics['edl']:.4f}, "
                f"Dice: {self.best_train_metrics['dice']:.4f}, "
                f"Con: {self.best_train_metrics['con']:.4f}, "
                f"TAlign: {self.best_train_metrics['topo_align']:.4f}) | "
                f"LR: {lr:.2e} | {elapsed:.1f}s"
            )
            if val_metrics:
                log_msg += (
                    f" | Dice: {val_metrics.get('val_dice_mean', 0):.4f} "
                    f"(RV:{val_metrics.get('val_dice_rv',0):.3f} "
                    f"Myo:{val_metrics.get('val_dice_myo',0):.3f} "
                    f"LV:{val_metrics.get('val_dice_lv',0):.3f})"
                    f" | Best Dice: {self.best_val_metrics['val_dice_mean']:.4f} "
                    f"(RV:{self.best_val_metrics['val_dice_rv']:.3f} "
                    f"Myo:{self.best_val_metrics['val_dice_myo']:.3f} "
                    f"LV:{self.best_val_metrics['val_dice_lv']:.3f})"
                )
            logger.info(log_msg)

            # TensorBoard
            self.writer.add_scalar("Train/Loss_Total",      train_metrics["total"], epoch)
            self.writer.add_scalar("Train/Loss_EDL",        train_metrics["edl"],   epoch)
            self.writer.add_scalar("Train/Loss_Dice",       train_metrics["dice"],  epoch)
            self.writer.add_scalar("Train/Loss_Contrastive",train_metrics["con"],   epoch)
            self.writer.add_scalar("Train/Loss_TopologyAlign", train_metrics["topo_align"], epoch)
            self.writer.add_scalar("Train/LR",              lr,                     epoch)
            for k, v in val_metrics.items():
                if v != float("inf"):
                    self.writer.add_scalar(f"Val/{k}", v, epoch)

            # History
            self.train_history["train_loss"].append(train_metrics["total"])
            self.train_history["train_edl"].append(train_metrics["edl"])
            self.train_history["train_dice"].append(train_metrics["dice"])
            self.train_history["train_con"].append(train_metrics["con"])
            self.train_history["train_topo_align"].append(train_metrics["topo_align"])
            self.train_history["val_dice_mean"].append(val_metrics.get("val_dice_mean", None))
            self.train_history["val_dice_rv"].append(val_metrics.get("val_dice_rv", None))
            self.train_history["val_dice_myo"].append(val_metrics.get("val_dice_myo", None))
            self.train_history["val_dice_lv"].append(val_metrics.get("val_dice_lv", None))
            self.train_history["val_hd95_mean"].append(val_metrics.get("val_hd95_mean", None))
            self.train_history["lr"].append(lr)

        self.writer.close()
        logger.info(
            "Training complete. "
            f"Best Loss: {self.best_train_metrics['total']:.4f} "
            f"(EDL: {self.best_train_metrics['edl']:.4f} "
            f"Dice: {self.best_train_metrics['dice']:.4f} "
            f"Con: {self.best_train_metrics['con']:.4f} "
            f"TAlign: {self.best_train_metrics['topo_align']:.4f}) | "
            f"Best val Dice: {self.best_val_metrics['val_dice_mean']:.4f} "
            f"(RV:{self.best_val_metrics['val_dice_rv']:.3f} "
            f"Myo:{self.best_val_metrics['val_dice_myo']:.3f} "
            f"LV:{self.best_val_metrics['val_dice_lv']:.3f})"
        )
        return self.train_history

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _save_checkpoint(self, epoch: int, metrics: dict, best: bool = False):
        fname = "best_model.pth" if best else f"checkpoint_ep{epoch+1}.pth"
        path  = os.path.join(self.ckpt_dir, fname)
        torch.save({
            "epoch":       epoch + 1,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "metrics":     metrics,
            "best_dice":   self.best_val_dice,
            "best_train_metrics": self.best_train_metrics,
            "best_val_metrics": self.best_val_metrics,
            "cfg":         self.cfg,
        }, path)
        if best:
            logger.info(f"  ✓ Saved best model (Dice: {self.best_val_dice:.4f}) → {path}")
