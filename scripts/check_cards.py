"""Check whether required reference cards exist in the hash database."""

from __future__ import annotations

import sys

from card_reco.database import HashDatabase

NEEDED: list[str] = [
    "sv3pt5-146",
    "base1-4",
    "base2-10",
    "base4-10",
    "base4-19",
    "base5-21",
    "base6-1",
    "gym1-12",
    "neo1-8",
    "ex2-99",
    "neo1-7",
]


def main() -> None:
    """Print the presence status of each required card."""
    try:
        db = HashDatabase()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Error opening database: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        for cid in NEEDED:
            card = db.get_card_by_id(cid)
            status = f"FOUND: {card.name}" if card else "MISSING"
            print(f"  {cid:>12}  {status}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
