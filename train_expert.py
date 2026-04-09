"""
Phase 1: Train M anatomy-specialised Experts.
Each Expert: SAM (frozen) + AdaLoRA + 4-branch decoder.
Loss weights learned via uncertainty weighting. Rank learned via AdaLoRA.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm

from config import Config
from data.dataset import ACDCDataset, collate_fn
from models.sam_expert import build_sam_expert
from models.losses import ExpertLoss


def train_one_expert(expert_id: int, config: Config, seed: int = 42):
    """Train a single Expert."""
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary = config.expert_assignments[expert_id]
    print(f"\n{'='*60}")
    print(f"Phase 1: Training Expert {expert_id} ({primary}-specialist), seed={seed}")
    print(f"{'='*60}")

    # Dataset (from preprocessed cache)
    train_ds = ACDCDataset(
        config.train.cache_dir, split="train",
        augment=True, tpa_enabled=config.aug.tpa_enabled,
        use_cached_features=config.train.cache_encoder_features,
    )
    val_ds = ACDCDataset(
        config.train.cache_dir, split="val",
        augment=False,
        use_cached_features=config.train.cache_encoder_features,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.train.expert_batch_size,
        shuffle=True, num_workers=config.data.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.train.expert_batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        collate_fn=collate_fn,
    )

    # Build Expert
    expert = build_sam_expert(
        sam_checkpoint=config.sam.checkpoint,
        model_type=config.sam.model_type,
        adalora_config=config.adalora,
        expert_id=expert_id,
        primary_structure=primary,
        num_classes=config.data.num_classes,
        img_size=config.data.img_size[0],
        embed_dim=config.loss.contrastive_embed_dim,
        use_cached_features=config.train.cache_encoder_features,
        use_medsam=config.sam.use_medsam,
    ).to(device)

    # Loss
    criterion = ExpertLoss(config.loss, config.edl, config.data.num_classes)

    # Optimizer: include uncertainty weighting params
    params = [
        {"params": expert.seg_branch.parameters(), "lr": config.train.expert_lr},
        {"params": expert.edl_branch.parameters(), "lr": config.train.expert_lr},
        {"params": expert.contrastive_branch.parameters(), "lr": config.train.expert_lr},
        {"params": expert.log_sigma2.parameters(), "lr": config.train.expert_lr * 10},  # Faster for sigma
    ]
    # Add AdaLoRA params if encoder exists and has trainable params
    if expert.encoder is not None:
        encoder_params = [p for p in expert.encoder.parameters() if p.requires_grad]
        if encoder_params:
            params.append({"params": encoder_params, "lr": config.train.expert_lr * 0.1})

    optimizer = optim.AdamW(params, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.expert_epochs, eta_min=1e-6)

    # Training loop
    best_val_dice = 0
    history = []

    for epoch in range(config.train.expert_epochs):
        expert.train()
        epoch_losses = {}
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"E{expert_id} Epoch {epoch+1}", leave=False):
            images = batch["image"].to(device)
            batch["mask"] = batch["mask"].to(device)

            outputs = expert(images)

            # Compute PD similarities for contrastive loss (if available)
            pd_sims = None
            B = images.shape[0]
            if train_ds.has_pds and B > 1:
                pd_sims = torch.zeros(B, B, device=device)
                for i in range(B):
                    for j in range(i + 1, B):
                        sim = train_ds.get_pd_similarity(batch["idx"][i], batch["idx"][j])
                        pd_sims[i, j] = sim
                        pd_sims[j, i] = sim

            losses = criterion(outputs, batch, epoch, pd_similarities=pd_sims)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v.item() if torch.is_tensor(v) else v)

        scheduler.step()

        # Epoch summary
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        sigma_vals = expert.get_sigma_values()
        elapsed = time.time() - t0

        # Validation
        val_dice = validate_expert(expert, val_loader, device)

        print(f"  Epoch {epoch+1}/{config.train.expert_epochs} "
              f"[{elapsed:.1f}s] "
              f"loss={avg_losses['total']:.4f} "
              f"val_dice={val_dice:.4f} "
              f"σ_LV={sigma_vals['LV']:.3f} "
              f"σ_Myo={sigma_vals['Myo']:.3f} "
              f"σ_RV={sigma_vals['RV']:.3f}")

        history.append({
            "epoch": epoch + 1,
            "losses": avg_losses,
            "sigma": sigma_vals,
            "val_dice": val_dice,
        })

        # Save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_dir = Path(config.train.checkpoint_dir) / f"seed_{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": expert.state_dict(),
                "epoch": epoch + 1,
                "val_dice": val_dice,
                "sigma": sigma_vals,
                "config": {
                    "expert_id": expert_id,
                    "primary": primary,
                }
            }, ckpt_dir / f"expert_{expert_id}_best.pth")

    # Save final config (learned values)
    print(f"\n  Expert {expert_id} ({primary}) training complete.")
    print(f"  Best val Dice: {best_val_dice:.4f}")
    print(f"  Final σ: {expert.get_sigma_values()}")
    print(f"  Final effective weights: {expert.get_effective_weights()}")

    return expert, history


def validate_expert(expert, val_loader, device):
    """Quick validation: average Dice over all structures."""
    from utils.metrics import dice_score
    expert.eval()
    all_dice = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            outputs = expert(images)
            preds = outputs["p_hat"].argmax(dim=1).cpu().numpy()

            for b in range(preds.shape[0]):
                for c in [1, 2, 3]:
                    all_dice.append(dice_score(preds[b], masks[b], c))

    return sum(all_dice) / max(len(all_dice), 1)


def train_all_experts(config: Config, seed: int = 42):
    """Train all M experts sequentially."""
    experts = []
    histories = []

    for expert_id in range(config.gating.num_experts):
        expert, history = train_one_expert(expert_id, config, seed)
        experts.append(expert)
        histories.append(history)

    return experts, histories


if __name__ == "__main__":
    config = Config()
    train_all_experts(config, seed=42)
