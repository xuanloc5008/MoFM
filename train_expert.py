"""
Phase 1: Train M general-purpose Experts.
Each Expert: MedSAM (frozen) + AdaLoRA + 4-branch decoder.
All experts have identical loss. Diversity comes from different random seeds.
Specialisation emerges in Phase 2 (gating), NOT here.
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
    """Train a single general-purpose Expert."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Phase 1: Training Expert {expert_id} (general-purpose), seed={seed}")
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

    # Build Expert — general purpose, no primary structure assignment
    expert = build_sam_expert(
        sam_checkpoint=config.sam.checkpoint,
        model_type=config.sam.model_type,
        adalora_config=config.adalora,
        expert_id=expert_id,
        primary_structure="LV",  # Dummy — σ init is symmetric now
        num_classes=config.data.num_classes,
        img_size=config.data.img_size[0],
        embed_dim=config.loss.contrastive_embed_dim,
        use_cached_features=config.train.cache_encoder_features,
        use_medsam=config.sam.use_medsam,
    ).to(device)

    # Symmetric σ init — all structures start equal
    for name in ["LV", "Myo", "RV"]:
        expert.log_sigma2[name].data.fill_(0.0)  # σ = 1.0 for all

    # Loss — identical for all experts, no specialisation
    criterion = ExpertLoss(config.loss, config.edl, config.data.num_classes)

    # Optimizer
    params = [
        {"params": list(expert.seg_branch.parameters()) +
                   list(expert.edl_branch.parameters()) +
                   list(expert.contrastive_branch.parameters()),
         "lr": config.train.expert_lr},
        {"params": list(expert.log_sigma2.parameters()),
         "lr": config.train.expert_lr * 10},
    ]
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
        val_dice, val_per_struct = validate_expert(expert, val_loader, device)

        print(f"  Epoch {epoch+1}/{config.train.expert_epochs} "
              f"[{elapsed:.1f}s] "
              f"loss={avg_losses['total']:.4f} "
              f"val: LV={val_per_struct['LV']:.3f} "
              f"Myo={val_per_struct['Myo']:.3f} "
              f"RV={val_per_struct['RV']:.3f} "
              f"avg={val_dice:.4f} | "
              f"σ: LV={sigma_vals['LV']:.3f} "
              f"Myo={sigma_vals['Myo']:.3f} "
              f"RV={sigma_vals['RV']:.3f}")

        history.append({
            "epoch": epoch + 1,
            "losses": avg_losses,
            "sigma": sigma_vals,
            "val_dice": val_dice,
            "val_per_struct": val_per_struct,
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
                "val_per_struct": val_per_struct,
                "sigma": sigma_vals,
                "effective_weights": expert.get_effective_weights(),
                "config": {
                    "expert_id": expert_id,
                    "seed": seed,
                }
            }, ckpt_dir / f"expert_{expert_id}_best.pth")

    print(f"\n  Expert {expert_id} (seed={seed}) training complete.")
    print(f"  Best val Dice: {best_val_dice:.4f}")
    print(f"  Final σ: {expert.get_sigma_values()}")
    print(f"  Final effective weights: {expert.get_effective_weights()}")

    return expert, history


def validate_expert(expert, val_loader, device):
    """Validation: per-structure Dice."""
    from utils.metrics import dice_score
    expert.eval()
    struct_dice = {"LV": [], "Myo": [], "RV": []}
    struct_map = {"LV": 1, "Myo": 2, "RV": 3}

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            outputs = expert(images)
            preds = outputs["p_hat"].argmax(dim=1).cpu().numpy()

            for b in range(preds.shape[0]):
                for name, c in struct_map.items():
                    struct_dice[name].append(dice_score(preds[b], masks[b], c))

    avg = {k: sum(v) / max(len(v), 1) for k, v in struct_dice.items()}
    overall = sum(avg.values()) / 3
    return overall, avg


def train_all_experts(config: Config, seed: int = 42):
    """Train all M experts — each with a different seed for diversity.
    
    All experts are general-purpose (identical loss, identical data).
    Diversity comes solely from different random initialisations.
    Specialisation emerges later in the gating network (Phase 2).
    """
    experts = []
    histories = []
    expert_seeds = config.loss.expert_seeds

    for expert_id in range(config.gating.num_experts):
        expert_seed = expert_seeds[expert_id] if expert_id < len(expert_seeds) else seed + expert_id * 100
        expert, history = train_one_expert(expert_id, config, expert_seed)
        experts.append(expert)
        histories.append(history)

    # Report diversity: compare per-structure dice across experts
    print(f"\n{'='*60}")
    print("Expert Diversity Summary")
    print(f"{'='*60}")
    for i, h in enumerate(histories):
        best = max(h, key=lambda x: x["val_dice"])
        ps = best.get("val_per_struct", {})
        print(f"  Expert {i} (seed={expert_seeds[i] if i < len(expert_seeds) else '?'}): "
              f"LV={ps.get('LV', 0):.3f} Myo={ps.get('Myo', 0):.3f} RV={ps.get('RV', 0):.3f}")

    return experts, histories


if __name__ == "__main__":
    config = Config()
    train_all_experts(config, seed=42)
