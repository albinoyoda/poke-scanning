"""Fine-tune MobileNetV3-Small for Pokemon card embedding via contrastive learning.

Trains the model so augmented views of the *same* card map to nearby
embeddings while different cards are pushed apart (NT-Xent / InfoNCE loss).

Usage:
    uv run python scripts/finetune_model.py [--epochs 12] [--batch-size 64]

Outputs:
    data/mobilenet_v3_small.onnx      (overwritten with fine-tuned model)

Requires dev dependencies: torch, torchvision.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("data")


def _collect_card_images(data_dir: Path) -> list[Path]:
    """Collect all reference card image paths."""
    cards_dir = data_dir / "metadata" / "cards"
    if not cards_dir.exists():
        print(f"Error: {cards_dir} not found", file=sys.stderr)
        sys.exit(1)
    card_files = sorted(cards_dir.glob("*.json"))
    paths: list[Path] = []
    for card_file in card_files:
        with open(card_file, encoding="utf-8") as f:
            cards = json.load(f)
        set_id = card_file.stem
        images_dir = data_dir / "images" / set_id
        for card in cards:
            card_id = card["id"]
            safe_id = re.sub(r'[<>:"/\\|?*]', "_", card_id)
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                p = images_dir / f"{safe_id}{ext}"
                if p.exists() and p.stat().st_size > 0:
                    paths.append(p)
                    break
    return paths


# ---------------------------------------------------------------------------
# Augmentation — simulate photo-like conditions
# ---------------------------------------------------------------------------


def _build_augmentation() -> transforms.Compose:
    """Build aggressive augmentation pipeline mimicking real card photos."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.65, 0.80)),
            transforms.RandomRotation(15),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ]
    )


class CardPairDataset(Dataset):
    """Yields two augmented views of each card image (SimCLR-style).

    All images are pre-loaded and resized to 256x256 uint8 numpy arrays
    to eliminate disk I/O during training while keeping memory manageable
    (~3.75 GB for 20k images vs ~11 GB for raw file bytes).
    """

    CACHE_SIZE = 256  # Pre-resize images to this dimension

    def __init__(self, image_paths: list[Path], transform: transforms.Compose) -> None:
        self.transform = transform
        # Pre-load images resized to 256x256 as numpy uint8 arrays
        print("  Pre-loading and resizing images into memory...")
        t0 = time.perf_counter()
        self.images: list[np.ndarray] = []
        failed = 0
        for i, p in enumerate(image_paths):
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((self.CACHE_SIZE, self.CACHE_SIZE), Image.BILINEAR)
                self.images.append(np.asarray(img, dtype=np.uint8))
            except Exception:  # noqa: BLE001
                failed += 1
            if (i + 1) % 5000 == 0:
                print(f"    {i + 1}/{len(image_paths)} loaded")
        elapsed = time.perf_counter() - t0
        total_mb = (
            len(self.images) * self.CACHE_SIZE * self.CACHE_SIZE * 3 / (1024 * 1024)
        )
        print(
            f"    Done: {len(self.images)} images ({total_mb:.0f} MB) in {elapsed:.1f}s"
        )
        if failed:
            print(f"    Warning: {failed} images failed to load")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.fromarray(self.images[idx])
        return self.transform(img), self.transform(img)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

EMBED_DIM = 576


class CardEmbeddingModel(nn.Module):
    """MobileNetV3-Small backbone + projection head for contrastive learning."""

    def __init__(self, proj_dim: int = 128) -> None:
        super().__init__()
        backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        # Keep everything up to and including avgpool
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        # Projection head for contrastive training (discarded at export)
        self.projector = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(EMBED_DIM, proj_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features (used at export time)."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 576)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features + project (used during training)."""
        feat = self.forward_features(x)
        return self.projector(feat)


# ---------------------------------------------------------------------------
# NT-Xent (InfoNCE) Loss
# ---------------------------------------------------------------------------


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Compute NT-Xent loss for a batch of positive pairs.

    z1, z2: (B, D) L2-normalised projection outputs.
    Each z1[i], z2[i] is a positive pair; all other combinations are negatives.
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ]
    )
    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_dir: Path,
    output_path: Path,
    *,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    temperature: float = 0.07,
    warmup_epochs: int = 2,
) -> None:
    """Fine-tune MobileNetV3-Small with contrastive learning."""
    print("Collecting card images...")
    image_paths = _collect_card_images(data_dir)
    print(f"  {len(image_paths)} reference images")

    augmentation = _build_augmentation()
    dataset = CardPairDataset(image_paths, augmentation)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    model = CardEmbeddingModel()
    model.train()

    # Two-rate optimiser: lower LR for pretrained backbone, higher for projector
    param_groups = [
        {"params": model.features.parameters(), "lr": backbone_lr},
        {"params": model.avgpool.parameters(), "lr": backbone_lr},
        {"params": model.projector.parameters(), "lr": lr},
    ]
    optimiser = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Cosine annealing scheduler
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

    n_loader = len(loader)
    print(f"\nTraining for {epochs} epochs ({n_loader} batches/epoch)...")

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        epoch_start = time.perf_counter()

        # Freeze backbone for warmup epochs
        freeze_backbone = epoch < warmup_epochs
        for p in model.features.parameters():
            p.requires_grad = not freeze_backbone

        for batch_idx, (view1, view2) in enumerate(loader):
            z1 = F.normalize(model(view1), dim=1)
            z2 = F.normalize(model(view2), dim=1)
            loss = nt_xent_loss(z1, z2, temperature=temperature)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.perf_counter() - epoch_start
                eta = elapsed / (batch_idx + 1) * (n_loader - batch_idx - 1)
                batch_loss = running_loss / n_batches
                print(
                    f"    [{batch_idx + 1}/{n_loader}] "
                    f"loss={batch_loss:.4f} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )

        epoch_elapsed = time.perf_counter() - epoch_start
        avg_loss = running_loss / max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        status = " (backbone frozen)" if freeze_backbone else ""
        remaining = (epochs - epoch - 1) * epoch_elapsed
        print(
            f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} "
            f"lr={current_lr:.6f} time={epoch_elapsed:.0f}s "
            f"remaining~{remaining:.0f}s{status}"
        )

    # Export to ONNX — backbone features only (drop projector)
    print(f"\nExporting fine-tuned model to {output_path}...")
    model.eval()

    class FeatureExtractor(nn.Module):
        """Thin wrapper that only calls forward_features."""

        def __init__(self, base: CardEmbeddingModel) -> None:
            super().__init__()
            self.features = base.features
            self.avgpool = base.avgpool

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            return torch.flatten(x, 1)

    export_model = FeatureExtractor(model)
    export_model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Model size: {size_mb:.1f} MB")

    # Verify
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path))
    result = session.run(None, {"input": dummy.numpy()})
    print(f"  Output shape: {result[0].shape}")
    print("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV3-Small for card embedding"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/mobilenet_v3_small.onnx",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        temperature=args.temperature,
        warmup_epochs=args.warmup_epochs,
    )


if __name__ == "__main__":
    main()
