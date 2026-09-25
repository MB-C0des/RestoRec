import csv
from pathlib import Path
from statistics import mean

here = Path(__file__).resolve().parent
path = next(here.rglob("retrieval_evaluation_results.csv"))
with path.open(encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

def mix(row):
    ids = row["retrieved_ids"].split(";")
    return tuple(sum(i.startswith(s + ":") for i in ids) for s in ("TikTok", "Reddit", "Places"))

print("Source mix (TikTok, Reddit, Places) per method:")
for method in ("Dense", "BM25", "Hybrid"):
    mixes = [mix(r) for r in rows if r["method"] == method]
    print(f"  {method}: {sorted(set(mixes))}")

affected = {}
for r in rows:
    if r["method"] == "Dense" and mix(r) != (3, 2, 3):
        affected[r["question"]] = mix(r)

print(f"\nAffected dense queries: {len(affected)}")
for question, m in affected.items():
    print(f"  - {question} {m}")

metrics = ["precision_at_k", "recall_at_k", "reciprocal_rank", "ndcg_at_k"]
for label, keep in (("All 20 queries", lambda q: True),
                    ("Unaffected queries only", lambda q: q not in affected)):
    print(f"\n{label}")
    print("  Method   P@8    R@8    MRR    nDCG@8")
    for method in ("Dense", "BM25", "Hybrid"):
        subset = [r for r in rows if r["method"] == method and keep(r["question"])]
        values = [mean(float(r[m]) for r in subset) for m in metrics]
        print(f"  {method:<7}" + "".join(f"{v:7.3f}" for v in values) + f"   (n={len(subset)})")