from card_reco.database import HashDatabase
db = HashDatabase()
needed = ['sv3pt5-146','base1-4','base2-10','base4-10','base4-19','base5-21','base6-1','gym1-12','neo1-8','ex2-99','neo1-7']
for cid in needed:
    c = db.get_card_by_id(cid)
    status = "FOUND: " + c.name if c else "MISSING"
    print(f"  {cid:>12}  {status}")
db.close()
