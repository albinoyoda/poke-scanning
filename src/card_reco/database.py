from __future__ import annotations

import sqlite3
from pathlib import Path

from card_reco.models import CardHashes, CardRecord

DEFAULT_DB_PATH = Path("data") / "card_hashes.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    set_id      TEXT NOT NULL,
    set_name    TEXT NOT NULL,
    number      TEXT NOT NULL,
    rarity      TEXT NOT NULL DEFAULT '',
    image_path  TEXT NOT NULL,
    ahash       TEXT NOT NULL,
    phash       TEXT NOT NULL,
    dhash       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cards_set_id ON cards(set_id);
CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(name);
"""

_CARD_COLUMNS = (
    "id, name, set_id, set_name, number, rarity, image_path, ahash, phash, dhash"
)


def _row_to_card(row: sqlite3.Row) -> CardRecord:
    """Convert a SQLite Row to a CardRecord."""
    return CardRecord(
        id=row["id"],
        name=row["name"],
        set_id=row["set_id"],
        set_name=row["set_name"],
        number=row["number"],
        rarity=row["rarity"],
        image_path=row["image_path"],
        ahash=row["ahash"],
        phash=row["phash"],
        dhash=row["dhash"],
    )


class HashDatabase:
    """SQLite-backed database of reference card hashes."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> HashDatabase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def insert_card(
        self,
        card_id: str,
        name: str,
        set_id: str,
        set_name: str,
        number: str,
        rarity: str,
        image_path: str,
        hashes: CardHashes,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cards
                (id, name, set_id, set_name, number, rarity, image_path,
                 ahash, phash, dhash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card_id,
                name,
                set_id,
                set_name,
                number,
                rarity,
                image_path,
                hashes.ahash,
                hashes.phash,
                hashes.dhash,
            ),
        )

    def commit(self) -> None:
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM cards").fetchone()
        return int(row[0])

    def get_all_cards(self) -> list[CardRecord]:
        rows = self._conn.execute(f"SELECT {_CARD_COLUMNS} FROM cards").fetchall()
        return [_row_to_card(r) for r in rows]

    def get_all_ids(self) -> set[str]:
        """Return the set of all card IDs in the database."""
        rows = self._conn.execute("SELECT id FROM cards").fetchall()
        return {r["id"] for r in rows}

    def get_card_by_id(self, card_id: str) -> CardRecord | None:
        row = self._conn.execute(
            f"SELECT {_CARD_COLUMNS} FROM cards WHERE id = ?",
            (card_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_card(row)
