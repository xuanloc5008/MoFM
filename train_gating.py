"""
Phase 2: Train Topology-Penalised Gating Network.
All Expert weights are frozen. Only the gating Mini-UNet is trained.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm

from config import Config
from data.dataset import ACDCDataset, collate_fn
from models.gating import GatingNetwork, GatingLoss
from models.sam_expert import SAMExpert


def train_gating(experts: list, config: Config, seed: int = 42):
    """Train the gating network given frozen experts."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Phase 2: Training Gating Network, seed={seed}")
    print(f"{'='*60}")

    # Freeze all experts
    for expert in experts:
        expert.eval()
        for p in expert.parameters():
            p.requires_grad = False

    # Dataset (from preprocessed cache)
    train_ds = ACDCDataset(
        config.train.cache_dir, split="train", augment=False,
        use_cached_features=config.train.cache_encoder_features,
    )
    val_ds = ACDCDataset(
        config.train.cache_dir, split="val", augment=False,
        use_cached_features=config.train.cache_encoder_features,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.train.gating_batch_size,
        shuffle=True, num_workers=config.data.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.train.gating_batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        collate_fn=collate_fn,
    )

    # Gating network
    gating = GatingNetwork(
        num_experts=config.gating.num_experts,
        num_classes=config.data.num_classes,
        base_channels=config.gating.gating_channels,
        cbam_reduction=config.gating.cbam_reduction,
    ).to(device)

    criterion = GatingLoss(config.gating.route_penalty_weight)
    optimizer = optim.AdamW(gating.parameters(), lr=config.train.gating_lr,
                            weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.gating_epochs, eta_min=1e-6)

    best_val_dice = 0
    history = []

    for epoch in range(config.train.gating_epochs):
        gating.train()
        epoch_losses = {}
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Gating Epoch {epoch+1}", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward through all frozen experts
            with torch.no_grad():
                expert_outputs = [expert(images) for expert in experts]

            # Gating
            weights = gating(images, expert_outputs)
            ensemble = gating.ensemble_predict(weights, expert_outputs)

            # Loss
            batch["mask"] = masks
            losses = criterion(ensemble, expert_outputs, weights, masks)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(gating.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v.item())

        scheduler.step()

        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        elapsed = time.time() - t0

        # Validation
        val_dice = validate_gating(gating, experts, val_loader, device)

        print(f"  Epoch {epoch+1}/{config.train.gating_epochs} "
              f"[{elapsed:.1f}s] "
              f"loss={avg_losses['total']:.4f} "
              f"dice={avg_losses.get('dice', 0):.4f} "
              f"route={avg_losses.get('route_penalty', 0):.4f} "
              f"val_dice={val_dice:.4f}")

        history.append({
            "epoch": epoch + 1,
            "losses": avg_losses,
            "val_dice": val_dice,
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_dir = Path(config.train.checkpoint_dir) / f"seed_{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": gating.state_dict(),
                "epoch": epoch + 1,
                "val_dice": val_dice,
            }, ckpt_dir / "gating_best.pth")

    print(f"\n  Gating training complete. Best val Dice: {best_val_dice:.4f}")
    return gating, history


def validate_gating(gating, experts, val_loader, device):
    from utils.metrics import dice_score
    gating.eval()
    all_dice = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            expert_outputs = [e(images) for e in experts]
            weights = gating(images, expert_outputs)
            ensemble = gating.ensemble_predict(weights, expert_outputs)
            preds = ensemble["prediction"].cpu().numpy()

            for b in range(preds.shape[0]):
                for c in [1, 2, 3]:
                    all_dice.append(dice_score(preds[b], masks[b], c))

    return sum(all_dice) / max(len(all_dice), 1)


if __name__ == "__main__":
    print("Run via run.py --phase 2")
