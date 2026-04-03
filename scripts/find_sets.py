import json

with open("data/metadata/sets.json") as f:
    sets = json.load(f)

targets = ["jungle", "base set 2", "team rocket", "legendary collection", 
           "gym heroes", "neo genesis", "sandstorm", "scarlet", "151", "base set"]
for s in sets:
    name_lower = s["name"].lower()
    for t in targets:
        if t in name_lower:
            print(f"{s['id']:>12}  {s['name']}")
            break
