import os, json, requests, pathlib
API=os.getenv("API","http://localhost:8080/analyze")
GOLD="data/golden_expected.json"

def test_golden():
    exp=json.load(open(GOLD))
    misses=[]
    for fn,expect in exp.items():
        p = ("data/golden/positives/"+fn if os.path.exists("data/golden/positives/"+fn)
             else "data/golden/negatives/"+fn)
        with open(p,"rb") as f:
            r=requests.post(API, files={"file": (fn, f, "image/jpeg")})
        r.raise_for_status()
        got=r.json()
        got_labels=sorted({x["damage_type"] for x in got.get("findings",[])})
        got_count=len(got.get("findings",[]))
        if not set(got_labels).issuperset(set(expect["labels"])) or abs(got_count-expect["count"])>1:
            misses.append((fn, expect, {"labels":got_labels,"count":got_count}))
    assert not misses, f"Regressions on {len(misses)} images (first 5): {misses[:5]}"
