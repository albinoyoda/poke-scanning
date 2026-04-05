"""Find set IDs matching target names in the sets metadata."""

from __future__ import annotations

import json

with open("data/metadata/sets.json", encoding="utf-8") as f:
    sets: list[dict[str, str]] = json.load(f)

targets: list[str] = [
    "jungle",
    "base set 2",
    "team rocket",
    "legendary collection",
    "gym heroes",
    "neo genesis",
    "sandstorm",
    "scarlet",
    "151",
    "base set",
]
for s in sets:
    name_lower: str = s["name"].lower()
    for t in targets:
        if t in name_lower:
            print(f"{s['id']:>12}  {s['name']}")
            break
