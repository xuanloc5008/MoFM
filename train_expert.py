"""
Phase 1: Train M general-purpose UNet Experts (end-to-end).
Each Expert: ResNet34-UNet + 4-branch output (Seg, EDL, Contrastive, Topo).
All experts have identical loss. Diversity comes from different random seeds.
Specialisation emerges in Phase 2 (gating), NOT here.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm

from config import Config
from data.dataset import ACDCDataset, collate_fn
from models.sam_expert import build_expert
from models.losses import ExpertLoss


def train_one_expert(expert_id: int, config: Config, seed: int = 42):
    """Train a single UNet Expert end-to-end."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Phase 1: Training Expert {expert_id} (UNet end-to-end), seed={seed}")
    print(f"{'='*60}")

    # Dataset — no cached features needed, images loaded directly
    train_ds = ACDCDataset(
        config.train.cache_dir, split="train",
        augment=True, tpa_enabled=config.aug.tpa_enabled,
        use_cached_features=False,
    )
    val_ds = ACDCDataset(
        config.train.cache_dir, split="val",
        augment=False, use_cached_features=False,
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

    # Build UNet Expert (~26M params for ResNet34)
    expert = build_expert(
        num_classes=config.data.num_classes,
        img_size=config.data.img_size[0],
        encoder_name=config.backbone.encoder_name,
        embed_dim=config.loss.contrastive_embed_dim,
        in_channels=1,
    ).to(device)

    # Loss — identical for all experts
    criterion = ExpertLoss(config.loss, config.edl, config.data.num_classes)

    # Optimizer — end-to-end, encoder + decoder + σ
    optimizer = optim.AdamW([
        {"params": [p for n, p in expert.named_parameters()
                    if "log_sigma2" not in n],
         "lr": config.train.expert_lr},
        {"params": list(expert.log_sigma2.parameters()),
         "lr": config.train.expert_lr * 10},
    ], weight_decay=config.train.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.expert_epochs, eta_min=1e-6)

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

            # PD similarities for contrastive loss
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

        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        sigma_vals = expert.get_sigma_values()
        elapsed = time.time() - t0

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
            "epoch": epoch + 1, "losses": avg_losses, "sigma": sigma_vals,
            "val_dice": val_dice, "val_per_struct": val_per_struct,
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_dir = Path(config.train.checkpoint_dir) / f"seed_{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": expert.state_dict(),
                "epoch": epoch + 1, "val_dice": val_dice,
                "val_per_struct": val_per_struct, "sigma": sigma_vals,
                "config": {"expert_id": expert_id, "seed": seed},
            }, ckpt_dir / f"expert_{expert_id}_best.pth")

    print(f"\n  Expert {expert_id} (seed={seed}) complete. Best Dice: {best_val_dice:.4f}")
    print(f"  Final σ: {expert.get_sigma_values()}")
    return expert, history


def validate_expert(expert, val_loader, device):
    from utils.metrics import dice_score
    expert.eval()
    struct_dice = {"LV": [], "Myo": [], "RV": []}

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            preds = expert(images)["p_hat"].argmax(dim=1).cpu().numpy()
            for b in range(preds.shape[0]):
                for name, c in [("LV", 1), ("Myo", 2), ("RV", 3)]:
                    struct_dice[name].append(dice_score(preds[b], masks[b], c))

    avg = {k: sum(v) / max(len(v), 1) for k, v in struct_dice.items()}
    return sum(avg.values()) / 3, avg


def train_all_experts(config: Config, seed: int = 42):
    """Train M experts — each with a different seed for diversity."""
    experts, histories = [], []
    expert_seeds = config.loss.expert_seeds

    for eid in range(config.gating.num_experts):
        s = expert_seeds[eid] if eid < len(expert_seeds) else seed + eid * 100
        expert, history = train_one_expert(eid, config, s)
        experts.append(expert)
        histories.append(history)

    print(f"\n{'='*60}\nExpert Diversity Summary\n{'='*60}")
    for i, h in enumerate(histories):
        best = max(h, key=lambda x: x["val_dice"])
        ps = best.get("val_per_struct", {})
        print(f"  Expert {i} (seed={expert_seeds[i] if i < len(expert_seeds) else '?'}): "
              f"LV={ps.get('LV',0):.3f} Myo={ps.get('Myo',0):.3f} RV={ps.get('RV',0):.3f}")

    return experts, histories


if __name__ == "__main__":
    config = Config()
    train_all_experts(config)
