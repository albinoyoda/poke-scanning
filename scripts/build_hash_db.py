"""Build the perceptual hash database from downloaded reference card images.

Usage:
    python scripts/build_hash_db.py [--data-dir DATA_DIR] [--db-path DB_PATH]

Reads card metadata + images from the data directory, computes perceptual hashes,
and stores them in a SQLite database. Resumable — skips cards already in the DB.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Add src to path so we can import card_reco
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from card_reco.database import HashDatabase
from card_reco.hasher import compute_hashes_pil

DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_PATH = Path("data") / "card_hashes.db"


def find_card_image(card: dict, data_dir: Path) -> Path | None:
    """Locate the downloaded image for a card."""
    card_id = card["id"]
    set_id = card.get("set", {}).get("id", card_id.split("-")[0])
    images_dir = data_dir / "images" / set_id
    # Apply the same sanitization used by download_reference_data.py
    safe_card_id = re.sub(r'[<>:"/\\|?*]', "_", card_id)

    # Try common extensions
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = images_dir / f"{safe_card_id}{ext}"
        if path.exists() and path.stat().st_size > 0:
            return path

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build perceptual hash database")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Base data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Output database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild from scratch (ignore existing DB entries)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    db_path = Path(args.db_path)

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

    # Open DB
    with HashDatabase(db_path) as db:
        existing_count = db.count() if not args.rebuild else 0
        if existing_count > 0:
            print(f"Database already has {existing_count} cards. Skipping those.")

        # Get existing IDs to skip
        existing_ids: set[str] = set()
        if not args.rebuild and existing_count > 0:
            existing_ids = {c.id for c in db.get_all_cards()}

        total_processed = 0
        total_skipped = 0
        total_failed = 0
        failed_sets: dict[str, int] = {}

        for card_file in card_files:
            with open(card_file, encoding="utf-8") as f:
                cards = json.load(f)

            set_id = card_file.stem
            set_info = sets_by_id.get(set_id, {})
            set_name = set_info.get("name", set_id)
            set_failed = 0

            for card in tqdm(cards, desc=f"Hashing {set_id}", leave=False):
                card_id = card["id"]

                if card_id in existing_ids:
                    total_skipped += 1
                    continue

                img_path = find_card_image(card, data_dir)
                if img_path is None:
                    total_failed += 1
                    set_failed += 1
                    continue

                try:
                    pil_image = Image.open(img_path).convert("RGB")
                    hashes = compute_hashes_pil(pil_image)

                    db.insert_card(
                        card_id=card_id,
                        name=card.get("name", "Unknown"),
                        set_id=set_id,
                        set_name=set_name,
                        number=card.get("number", ""),
                        rarity=card.get("rarity", ""),
                        image_path=str(img_path),
                        hashes=hashes,
                    )
                    total_processed += 1

                    # Checkpoint every 500 cards
                    if total_processed % 500 == 0:
                        db.commit()

                except Exception as e:
                    print(
                        f"  Warning: Failed to hash {card_id}: {e}",
                        file=sys.stderr,
                    )
                    total_failed += 1
                    set_failed += 1

            if set_failed > 0:
                failed_sets[set_id] = set_failed

            db.commit()

        print("\nDone!")
        print(f"  Processed: {total_processed}")
        print(f"  Skipped (already in DB): {total_skipped}")
        print(f"  Failed: {total_failed}")
        print(f"  Total in DB: {db.count()}")
        print(f"  Database: {db_path.resolve()}")

        if failed_sets:
            print(f"\n  Sets with missing images ({len(failed_sets)} sets):")
            for sid, count in sorted(failed_sets.items(), key=lambda x: -x[1]):
                print(f"    {sid}: {count} cards")
            print(
                "\n  Hint: run scripts/download_reference_data.py"
                " to download missing images."
            )


if __name__ == "__main__":
    main()
