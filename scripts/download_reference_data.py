"""Download Pokemon TCG card metadata and images from PokemonTCG/pokemon-tcg-data.

Usage:
    python scripts/download_reference_data.py [--data-dir DATA_DIR] [--small]

Downloads card JSON metadata from the GitHub repo, then downloads card images
from images.pokemontcg.io. Resumable — skips already-downloaded files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master"
SETS_URL = f"{GITHUB_RAW_BASE}/sets/en.json"
CARDS_URL_TEMPLATE = f"{GITHUB_RAW_BASE}/cards/en/{{set_id}}.json"

DEFAULT_DATA_DIR = Path("data")
METADATA_DIR_NAME = "metadata"
IMAGES_DIR_NAME = "images"

REQUEST_TIMEOUT = 30
DOWNLOAD_DELAY = 0.05  # seconds between image downloads to be respectful


def download_sets(data_dir: Path) -> list[dict]:
    """Download the sets index."""
    meta_dir = data_dir / METADATA_DIR_NAME
    meta_dir.mkdir(parents=True, exist_ok=True)

    sets_file = meta_dir / "sets.json"
    if sets_file.exists():
        print(f"Sets file already exists: {sets_file}")
        with open(sets_file, encoding="utf-8") as f:
            return json.load(f)

    print("Downloading sets index...")
    resp = requests.get(SETS_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    sets_data = resp.json()

    with open(sets_file, "w", encoding="utf-8") as f:
        json.dump(sets_data, f, indent=2)

    print(f"Downloaded {len(sets_data)} sets.")
    return sets_data


def download_cards_for_set(set_id: str, data_dir: Path) -> list[dict]:
    """Download card metadata for a single set."""
    meta_dir = data_dir / METADATA_DIR_NAME / "cards"
    meta_dir.mkdir(parents=True, exist_ok=True)

    cards_file = meta_dir / f"{set_id}.json"
    if cards_file.exists():
        with open(cards_file, encoding="utf-8") as f:
            return json.load(f)

    url = CARDS_URL_TEMPLATE.format(set_id=set_id)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    cards_data = resp.json()

    with open(cards_file, "w", encoding="utf-8") as f:
        json.dump(cards_data, f, indent=2)

    return cards_data


def download_card_image(
    card: dict, data_dir: Path, use_small: bool = False
) -> Path | None:
    """Download a single card image. Returns the local path or None on failure."""
    images = card.get("images", {})
    url = images.get("small" if use_small else "large") or images.get("small")
    if not url:
        return None

    set_id = card.get("set", {}).get("id", card.get("id", "unknown").split("-")[0])
    card_id = card["id"]
    ext = Path(url).suffix or ".png"

    img_dir = data_dir / IMAGES_DIR_NAME / set_id
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"{card_id}{ext}"

    if img_path.exists() and img_path.stat().st_size > 0:
        return img_path

    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(resp.content)
        return img_path
    except requests.RequestException as e:
        print(f"  Warning: Failed to download {card_id}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Pokemon TCG reference data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Base data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Download small images instead of large (~1 GB vs ~5-10 GB)",
    )
    parser.add_argument(
        "--sets",
        type=str,
        nargs="*",
        default=None,
        help="Only download specific set IDs (default: all sets)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Step 1: Download sets
    sets_data = download_sets(data_dir)
    set_ids = [s["id"] for s in sets_data]

    if args.sets:
        set_ids = [sid for sid in args.sets if sid in set_ids]
        if not set_ids:
            print(
                f"Error: None of the specified sets found. Available: {[s['id'] for s in sets_data[:10]]}...",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"\nWill process {len(set_ids)} sets.")

    # Step 2: Download card metadata for each set
    all_cards: list[dict] = []
    print("\nDownloading card metadata...")
    for set_id in tqdm(set_ids, desc="Sets"):
        try:
            cards = download_cards_for_set(set_id, data_dir)
            # Attach set info to each card if not present
            set_info = next((s for s in sets_data if s["id"] == set_id), {})
            for card in cards:
                if "set" not in card:
                    card["set"] = set_info
            all_cards.extend(cards)
        except requests.RequestException as e:
            print(f"  Warning: Failed to download set {set_id}: {e}", file=sys.stderr)

    print(f"Total cards: {len(all_cards)}")

    # Step 3: Download card images
    print(f"\nDownloading card images ({'small' if args.small else 'large'})...")
    downloaded = 0
    skipped = 0
    failed = 0

    for card in tqdm(all_cards, desc="Images"):
        result = download_card_image(card, data_dir, use_small=args.small)
        if result is None:
            failed += 1
        elif result.stat().st_size > 0:
            downloaded += 1
        else:
            skipped += 1
        time.sleep(DOWNLOAD_DELAY)

    print(f"\nDone! Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    print(f"Data directory: {data_dir.resolve()}")


if __name__ == "__main__":
    main()
