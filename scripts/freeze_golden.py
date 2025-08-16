import os, json, glob, requests, pathlib
API=os.getenv("API","http://localhost:8080/analyze")
OUT="data/golden_expected.json"
golden={}
for d in ["data/golden/positives","data/golden/negatives"]:
    if not os.path.isdir(d): continue
    for p in sorted(glob.glob(os.path.join(d,"*.*"))):
        fn=pathlib.Path(p).name
        with open(p,"rb") as f:
            r=requests.post(API, files={"file": (fn, f, "image/jpeg")})
        r.raise_for_status()
        j=r.json()
        labels=sorted({x["damage_type"] for x in j.get("findings",[])})
        golden[fn]={"count": len(j.get("findings",[])), "labels": labels}
os.makedirs("data", exist_ok=True)
json.dump(golden, open(OUT,"w"), indent=2)
print(f"Wrote {OUT} with {len(golden)} entries.")
