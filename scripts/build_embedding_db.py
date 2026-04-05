"""Build the FAISS embedding index from downloaded reference card images.

Usage:
    uv run python scripts/build_embedding_db.py [--data-dir DATA_DIR]

Reads card metadata + images from the data directory, computes CNN
embeddings via ONNX Runtime, and stores them in a FAISS index with a
JSON metadata sidecar.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from card_reco.embedder import EMBEDDING_DIM, CardEmbedder
from card_reco.faiss_index import DEFAULT_INDEX_PATH, DEFAULT_META_PATH, CardIndex
from card_reco.models import CardRecord

DEFAULT_DATA_DIR = Path("data")


def find_card_image(card: dict, data_dir: Path) -> Path | None:
    """Locate the downloaded image for a card."""
    card_id = card["id"]
    set_id = card.get("set", {}).get("id", card_id.split("-")[0])
    images_dir = data_dir / "images" / set_id
    safe_card_id = re.sub(r'[<>:"/\\|?*]', "_", card_id)

    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = images_dir / f"{safe_card_id}{ext}"
        if path.exists() and path.stat().st_size > 0:
            return path

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS embedding index")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Base data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=str(DEFAULT_INDEX_PATH),
        help=f"Output FAISS index path (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default=str(DEFAULT_META_PATH),
        help=f"Output metadata JSON path (default: {DEFAULT_META_PATH})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to ONNX model (default: data/mobilenet_v3_small.onnx)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation (default: 32)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load sets metadata
    sets_file = data_dir / "metadata" / "sets.json"
    if not sets_file.exists():
        print(
            f"Error: Sets file not found: {sets_file}\n"
            "Run scripts/download_reference_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(sets_file, encoding="utf-8") as f:
        sets_data = json.load(f)
    sets_by_id = {s["id"]: s for s in sets_data}

    # Collect all card metadata files
    cards_dir = data_dir / "metadata" / "cards"
    if not cards_dir.exists():
        print(f"Error: Cards directory not found: {cards_dir}", file=sys.stderr)
        sys.exit(1)

    card_files = sorted(cards_dir.glob("*.json"))
    print(f"Found {len(card_files)} set metadata files.")

    # First pass: collect all cards with valid images.
    all_cards: list[CardRecord] = []
    all_image_paths: list[Path] = []

    for card_file in card_files:
        with open(card_file, encoding="utf-8") as f:
            cards = json.load(f)
        set_id = card_file.stem
        set_info = sets_by_id.get(set_id, {})
        set_name = set_info.get("name", set_id)

        for card in cards:
            img_path = find_card_image(card, data_dir)
            if img_path is None:
                continue
            all_cards.append(
                CardRecord(
                    id=card["id"],
                    name=card.get("name", "Unknown"),
                    set_id=set_id,
                    set_name=set_name,
                    number=card.get("number", ""),
                    rarity=card.get("rarity", ""),
                    image_path=str(img_path),
                )
            )
            all_image_paths.append(img_path)

    print(f"Found {len(all_cards)} cards with images.")

    # Load embedder.
    print("Loading CNN model...")
    embedder = CardEmbedder(args.model_path)

    # Compute embeddings in batches.
    all_embeddings = np.empty((len(all_cards), EMBEDDING_DIM), dtype=np.float32)
    batch_size = args.batch_size
    failed = 0

    for start in tqdm(range(0, len(all_cards), batch_size), desc="Embedding"):
        end = min(start + batch_size, len(all_cards))
        batch_images = []
        for i in range(start, end):
            try:
                pil_img = Image.open(all_image_paths[i]).convert("RGB")
                batch_images.append(np.asarray(pil_img, dtype=np.uint8))
            except Exception as e:
                print(f"  Warning: {all_cards[i].id}: {e}", file=sys.stderr)
                # Use zero vector for failed images.
                batch_images.append(np.zeros((224, 224, 3), dtype=np.uint8))
                failed += 1

        # embed_pil expects PIL, but we have RGB numpy arrays already.
        # Use _preprocess → session.run directly via embed_batch-like logic.
        embeddings = np.stack(
            [embedder.embed_pil(Image.fromarray(img)) for img in batch_images]
        )
        all_embeddings[start:end] = embeddings

    # Build and save index.
    print(f"\nBuilding FAISS index ({len(all_cards)} vectors, {EMBEDDING_DIM} dims)...")
    CardIndex.build(
        all_embeddings,
        all_cards,
        index_path=args.index_path,
        meta_path=args.meta_path,
    )

    index_size = Path(args.index_path).stat().st_size / (1024 * 1024)
    meta_size = Path(args.meta_path).stat().st_size / (1024 * 1024)

    print("Done!")
    print(f"  Cards indexed: {len(all_cards)}")
    print(f"  Failed: {failed}")
    print(f"  Index: {args.index_path} ({index_size:.1f} MB)")
    print(f"  Metadata: {args.meta_path} ({meta_size:.1f} MB)")


if __name__ == "__main__":
    main()
