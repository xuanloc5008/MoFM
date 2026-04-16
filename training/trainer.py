"""
training/trainer.py
3-phase curriculum training engine for DUT-Mamba-MoE.
Phase 1: Experts only (seg + EDL, σ fixed)
Phase 2: Routing + topology awakening
Phase 3: Bayesian auto-balancing (σ unfrozen)
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import cfg
from models.architecture import DUTMambaMoE
from training.losses import (
    EDLLoss, Betti3DLoss, TopologicalContrastiveLoss,
    SegmentationLoss, RoutingLoss, HomoscedasticLoss, GTTECLoss
)
from ttec.ttec_engine import GTTECGating, TopologicalUncertaintyScorer
from evaluation.metrics import compute_metrics


class CurriculumTrainer:
    """
    3-phase curriculum trainer.
    Supports multi-seed training for reproducibility.
    """

    def __init__(
        self,
        model: DUTMambaMoE,
        dataloaders: Dict,
        output_dir: str,
        device: torch.device,
        seed: int = 42,
    ):
        self.model       = model.to(device)
        self.dataloaders = dataloaders
        self.output_dir  = Path(output_dir)
        self.device      = device
        self.seed        = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss functions
        self.edl_loss    = EDLLoss()
        self.topo_loss   = Betti3DLoss(
            resolution=cfg.topology.cubical_resolution
        )
        self.tcl_loss    = TopologicalContrastiveLoss()
        self.seg_loss    = SegmentationLoss(cfg.data.num_classes)
        self.route_loss  = RoutingLoss()
        self.homo_loss   = HomoscedasticLoss()
        self.gate_loss   = GTTECLoss()

        # TU scorer for post-training
        self.tu_scorer = TopologicalUncertaintyScorer()

        # Metrics history
        self.history = {"train": [], "val": []}

        # Best checkpoint
        self.best_dice = 0.0

    # ─────────────────────────────────────────────────────────────
    # Phase 1: Expert burn-in
    # ─────────────────────────────────────────────────────────────

    def train_phase1(self):
        """
        Train each expert independently.
        Only seg + EDL losses, σ frozen at 1.
        Routing network NOT active.
        """
        print(f"\n{'='*60}")
        print(f"PHASE 1: Expert Burn-in ({cfg.training.phase1_epochs} epochs)")
        print(f"{'='*60}")

        # Freeze routing
        for p in self.model.routing.parameters():
            p.requires_grad_(False)
        # Freeze log_sigma2 in Phase 1
        self.model.log_sigma2.requires_grad_(False)

        # Separate optimisers per expert
        expert_opts = [
            AdamW(
                [p for p in exp.parameters() if p.requires_grad],
                lr=cfg.training.lr_expert,
                weight_decay=cfg.training.weight_decay,
            )
            for exp in self.model.experts
        ]
        schedulers = [
            CosineAnnealingLR(opt, T_max=cfg.training.phase1_epochs)
            for opt in expert_opts
        ]

        for epoch in range(1, cfg.training.phase1_epochs + 1):
            self.model.train()
            epoch_losses = []
            for batch in tqdm(
                self.dataloaders["train"],
                desc=f"P1 Ep {epoch}/{cfg.training.phase1_epochs}",
                leave=False
            ):
                image = batch["image"].to(self.device)
                label = batch["label"].to(self.device).long()
                if label.ndim == 5:
                    label = label.squeeze(1)

                y_oh = F.one_hot(label, num_classes=cfg.data.num_classes)
                y_oh = y_oh.permute(0, 4, 1, 2, 3).float()

                total_loss = torch.tensor(0.0, device=self.device)
                for m, (expert, opt) in enumerate(
                    zip(self.model.experts, expert_opts)
                ):
                    opt.zero_grad()
                    out, hiddens = expert(image)

                    l_seg = self.seg_loss(out["logits"], label)
                    l_edl = self.edl_loss(out["alpha"], y_oh, epoch,
                                          cfg.training.phase1_epochs)
                    loss = l_seg + l_edl
                    loss.backward()
                    nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
                    opt.step()
                    total_loss = total_loss + loss.detach()

                epoch_losses.append(total_loss.item() / len(self.model.experts))

            for sch in schedulers:
                sch.step()

            if epoch % 10 == 0:
                val_metrics = self._validate(phase=1)
                print(f"  P1 Ep {epoch:3d} | loss={np.mean(epoch_losses):.4f} "
                      f"| Dice_avg={val_metrics['dice_avg']:.4f}")

        self._save_checkpoint("phase1_complete")

    # ─────────────────────────────────────────────────────────────
    # Phase 2: Routing + topology awakening
    # ─────────────────────────────────────────────────────────────

    def train_phase2(self):
        """
        Freeze experts, train routing network.
        Introduce TCL and ramp up Topo3D loss linearly.
        """
        print(f"\n{'='*60}")
        print(f"PHASE 2: Routing + Topology ({cfg.training.phase2_epochs} epochs)")
        print(f"{'='*60}")

        # Freeze experts
        for exp in self.model.experts:
            for p in exp.parameters():
                p.requires_grad_(False)
        # Unfreeze routing
        for p in self.model.routing.parameters():
            p.requires_grad_(True)

        opt = AdamW(
            list(self.model.routing.parameters()),
            lr=cfg.training.lr_routing,
            weight_decay=cfg.training.weight_decay,
        )
        sch = CosineAnnealingLR(opt, T_max=cfg.training.phase2_epochs)
        TOPO_WARMUP = 20   # epochs to ramp Topo loss

        for epoch in range(1, cfg.training.phase2_epochs + 1):
            self.model.train()
            topo_weight = min(1.0, epoch / TOPO_WARMUP)
            epoch_losses = []

            for batch in tqdm(
                self.dataloaders["train"],
                desc=f"P2 Ep {epoch}/{cfg.training.phase2_epochs}",
                leave=False
            ):
                image = batch["image"].to(self.device)
                label = batch["label"].to(self.device).long()
                if label.ndim == 5:
                    label = label.squeeze(1)

                opt.zero_grad()
                with torch.no_grad():
                    # Get frozen expert outputs
                    expert_outs_list = []
                    for exp in self.model.experts:
                        out, _ = exp(image)
                        expert_outs_list.append(out)

                weights = self.model.routing(image, expert_outs_list)

                # Routing loss with topo penalty
                vacuities = [o["vacuity"] for o in expert_outs_list]
                topo_vals = []
                for m, s in enumerate(self.model.structures):
                    tl = self.topo_loss(expert_outs_list[m]["p_hat"], s)
                    topo_vals.append(tl.item() if isinstance(tl, torch.Tensor)
                                     else tl)

                l_route = self.route_loss(weights, vacuities, topo_vals)

                # TCL loss (Phase 2: introduce)
                l_tcl = torch.tensor(0.0, device=self.device)
                if "image_aug" in batch:
                    img_aug = batch["image_aug"].to(self.device)
                    for exp in self.model.experts:
                        _, h_orig = exp(image)
                        _, h_aug  = exp(img_aug)
                        l_tcl = l_tcl + self.tcl_loss(h_orig, h_aug)
                    l_tcl = l_tcl / len(self.model.experts)

                loss = l_route + cfg.training.lambda_pd * l_tcl * topo_weight
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.routing.parameters(), 1.0)
                opt.step()
                epoch_losses.append(loss.item())

            sch.step()
            if epoch % 10 == 0:
                val_metrics = self._validate(phase=2)
                print(f"  P2 Ep {epoch:3d} | loss={np.mean(epoch_losses):.4f} "
                      f"| Dice_avg={val_metrics['dice_avg']:.4f} "
                      f"| TER={val_metrics.get('ter', 0.0):.4f}")

        self._save_checkpoint("phase2_complete")

    # ─────────────────────────────────────────────────────────────
    # Phase 3: Bayesian auto-balancing
    # ─────────────────────────────────────────────────────────────

    def train_phase3(self):
        """
        Unfreeze σ parameters. Joint optimisation of routing + σ.
        All experts remain frozen.
        """
        print(f"\n{'='*60}")
        print(f"PHASE 3: Bayesian Auto-Balancing ({cfg.training.phase3_epochs} epochs)")
        print(f"{'='*60}")

        # Unfreeze σ
        self.model.log_sigma2.requires_grad_(True)

        opt = AdamW(
            list(self.model.routing.parameters()) +
            [self.model.log_sigma2],
            lr=cfg.training.lr_routing,
            weight_decay=cfg.training.weight_decay,
        )
        sch = CosineAnnealingLR(opt, T_max=cfg.training.phase3_epochs)

        for epoch in range(1, cfg.training.phase3_epochs + 1):
            self.model.train()
            epoch_losses = []

            for batch in tqdm(
                self.dataloaders["train"],
                desc=f"P3 Ep {epoch}/{cfg.training.phase3_epochs}",
                leave=False,
            ):
                image = batch["image"].to(self.device)
                label = batch["label"].to(self.device).long()
                if label.ndim == 5:
                    label = label.squeeze(1)
                y_oh = F.one_hot(label, num_classes=cfg.data.num_classes)
                y_oh = y_oh.permute(0, 4, 1, 2, 3).float()

                opt.zero_grad()
                with torch.no_grad():
                    expert_outs_list = []
                    for exp in self.model.experts:
                        out, _ = exp(image)
                        expert_outs_list.append(out)

                weights = self.model.routing(image, expert_outs_list)

                # Compute all component losses
                sigma = self.model.get_sigma()   # (M, 4)
                total_loss = torch.tensor(0.0, device=self.device)

                for m, (s, out) in enumerate(
                    zip(self.model.structures, expert_outs_list)
                ):
                    l_edl  = self.edl_loss(out["alpha"], y_oh, epoch,
                                           cfg.training.phase3_epochs)
                    l_topo = self.topo_loss(out["p_hat"], s)
                    l_seg  = self.seg_loss(out["logits"], label)
                    losses_m = {
                        "edl": l_edl, "topo": l_topo,
                        "seg": l_seg, "tcl": torch.tensor(0.0, device=self.device),
                    }
                    l_m = self.homo_loss(losses_m, self.model.log_sigma2[m])
                    total_loss = total_loss + l_m

                vacuities = [o["vacuity"] for o in expert_outs_list]
                topo_vals = [0.0] * len(self.model.experts)
                l_route = self.route_loss(weights, vacuities, topo_vals)
                total_loss = total_loss + l_route

                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.routing.parameters()) +
                    [self.model.log_sigma2], 1.0
                )
                opt.step()
                epoch_losses.append(total_loss.item())

            sch.step()
            if epoch % 5 == 0:
                sigma_vals = self.model.get_sigma().detach().cpu()
                val_metrics = self._validate(phase=3)
                print(f"  P3 Ep {epoch:3d} | loss={np.mean(epoch_losses):.4f} "
                      f"| Dice_avg={val_metrics['dice_avg']:.4f}")
                print(f"    σ: {sigma_vals.numpy().round(3)}")

        self._save_checkpoint("phase3_complete")

    # ─────────────────────────────────────────────────────────────
    # Train G_TTEC on annotation data
    # ─────────────────────────────────────────────────────────────

    def train_gttec(
        self,
        annotation_features: torch.Tensor,   # (N, 7)
        annotation_labels: torch.Tensor,      # (N,) long {0,1,2}
        gating: GTTECGating,
        n_epochs: int = 100,
    ) -> GTTECGating:
        """
        Train G_TTEC after main model converges.
        Uses TTEC annotation study data as supervision.
        """
        print(f"\n{'='*60}")
        print(f"TRAINING G_TTEC ({n_epochs} epochs)")
        print(f"{'='*60}")

        gating = gating.to(self.device)
        feat  = annotation_features.to(self.device)
        labs  = annotation_labels.to(self.device)

        opt = AdamW(gating.parameters(), lr=cfg.training.lr_gate,
                    weight_decay=cfg.training.weight_decay)
        loss_fn = GTTECLoss().to(self.device)

        best_loss = float("inf")
        for ep in range(1, n_epochs + 1):
            gating.train()
            opt.zero_grad()
            q = gating(feat)
            loss = loss_fn(q, labs)
            loss.backward()
            opt.step()
            if ep % 20 == 0:
                acc = (q.argmax(1) == labs).float().mean().item()
                print(f"  G_TTEC Ep {ep:3d} | loss={loss.item():.4f} | acc={acc:.4f}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(gating.state_dict(),
                           self.output_dir / "gttec_best.pt")

        gating.load_state_dict(
            torch.load(self.output_dir / "gttec_best.pt",
                       map_location=self.device)
        )
        return gating

    # ─────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, phase: int = 3) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in self.dataloaders["val"]:
            image = batch["image"].to(self.device)
            label = batch["label"].to(self.device).long()
            if label.ndim == 5:
                label = label.squeeze(1)

            if phase == 1:
                out, _ = self.model.experts[0](image)
                p_hat = out["p_hat"]
            else:
                result = self.model(image, phase=phase)
                p_hat = result["p_bar"]

            pred = p_hat.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

        preds  = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(preds, labels, cfg.data.num_classes)
        self.history["val"].append(metrics)

        # Save best
        if metrics["dice_avg"] > self.best_dice:
            self.best_dice = metrics["dice_avg"]
            self._save_checkpoint("best")

        return metrics

    # ─────────────────────────────────────────────────────────────
    # Full curriculum
    # ─────────────────────────────────────────────────────────────

    def run_full_curriculum(self):
        self.train_phase1()
        self.train_phase2()
        self.train_phase3()
        # Accumulate TU scorer from training set
        print("\nAccumulating TU Fréchet means from training data...")
        self._accumulate_tu()
        print("Training complete.")
        return self.history

    def _accumulate_tu(self):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloaders["train"], desc="TU accumulation"):
                image = batch["image"].to(self.device)
                result = self.model(image, phase=3)
                for hidden_list in result["hidden_states"]:
                    flat_h = [h.detach().cpu() for h in hidden_list]
                    self.tu_scorer.update_from_batch(flat_h)
        self.tu_scorer.finalise()
        torch.save(
            self.tu_scorer.frechet_means,
            self.output_dir / "frechet_means.pt"
        )

    def _save_checkpoint(self, tag: str):
        path = self.output_dir / f"checkpoint_{tag}_seed{self.seed}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "history": self.history,
            "best_dice": self.best_dice,
            "sigma": self.model.get_sigma().detach().cpu(),
        }, path)
        print(f"  → Checkpoint saved: {path.name}")
