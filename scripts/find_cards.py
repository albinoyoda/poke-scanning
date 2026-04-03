"""Find card IDs for annotation cards."""
import json

# Map annotation set names to actual set IDs
SETS = {
    "base2": "base2",    # Jungle → Scyther
    "base4": "base4",    # Base Set 2 → Mewtwo, Wigglytuff
    "base5": "base5",    # Team Rocket → Dark Charizard
    "base6": "base6",    # Legendary Collection → Alakazam
    "gym1": "gym1",      # Gym Heroes → Rocket's Moltres
    "neo1": "neo1",      # Neo Genesis → Kingdra
    "ex2": "ex2",        # Sandstorm → Typhlosion EX
}

search_terms = {
    "base2": [("scyther", "10")],
    "base4": [("mewtwo", "10"), ("wigglytuff", "19")],
    "base5": [("dark charizard", "21")],
    "base6": [("alakazam", "1")],
    "gym1": [("rocket's moltres", "12"), ("moltres", "12")],
    "neo1": [("kingdra", "8")],
    "ex2": [("typhlosion", "99")],
}

for set_id, terms in search_terms.items():
    cards = json.load(open(f"data/metadata/cards/{set_id}.json"))
    for name_term, number in terms:
        for c in cards:
            if name_term in c["name"].lower() and c.get("number") == number:
                print(f"{set_id:>6}  {c['id']:>20}  {c['name']}  #{c.get('number','')}")
                break

# Also check for Jumpluff (unknown set in annotation)
for set_file in ["neo1", "neo2", "neo3", "neo4"]:
    try:
        cards = json.load(open(f"data/metadata/cards/{set_file}.json"))
        for c in cards:
            if "jumpluff" in c["name"].lower():
                print(f"{set_file:>6}  {c['id']:>20}  {c['name']}  #{c.get('number','')}")
    except FileNotFoundError:
        pass
