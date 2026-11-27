"""Precision and recall of the relevance gate against evals/gate_labels.json (LLM labels against CURATE_SCOPE).

    python -m evals.gate_eval evals/runs/gate_pool_2026-11-27.csv              # tuning split, old vs new, errors
    python -m evals.gate_eval evals/runs/gate_pool_2026-11-27.csv --heldout    # held-out third: report only
    python -m evals.gate_eval evals/runs/gate_pool_2026-11-27.csv --library    # current manifest papers the new gate drops

Positive = in scope. Precision = share of gate passes that are in scope; recall = share of in-scope papers that
pass. The sample is stratified on the old gate, so "weighted" re-weights each stratum to its size in the pool.
Prints titles and matched terms only, never abstracts.
"""
import argparse
import csv
import json
from collections import Counter

from src import config, curate, manifest
from evals.gate_label import LABELS, strata

# The gate before fix/relevance-gate, frozen here for the comparison.
OLD_INCLUDE = {"resistance", "strength", "hypertrophy", "weightlifting", "1rm", "one-repetition", "periodization",
               "training volume"}
OLD_EXCLUDE = {"patients", "patient", "disease", "hiv", "arthritis", "osteoarthritis", "rheumatoid", "cancer",
               "diabetes", "clinical", "metformin", "ibuprofen", "therapy", "mtorc1", "hydroxylation",
               "signaling pathway", "gene expression", "in vitro", "rodent", "mice", "murine", "knockout"}
GATES = {"old": curate.RelevanceFilter(OLD_INCLUDE, OLD_EXCLUDE, (), ()), "new": curate.RelevanceFilter(*curate.GATE_RULES)}


def metrics(items: list[dict], gate, weights: dict | None = None) -> dict:
    count = Counter()
    for it in items:
        passed = gate.check(it["text"]).passed
        count[("tp" if it["label"] == "in" else "fp") if passed else ("fn" if it["label"] == "in" else "tn")] += \
            (weights or {}).get(it["stratum"], 1)
    precision = count["tp"] / ((count["tp"] + count["fp"]) or 1)
    recall = count["tp"] / ((count["tp"] + count["fn"]) or 1)
    return {"precision": round(precision, 3), "recall": round(recall, 3), **{k: round(count[k], 1) for k in
                                                                              ("tp", "fp", "fn", "tn")}}


def errors(items: list[dict], gate) -> list[str]:
    out = []
    for it in items:
        result = gate.check(it["text"])
        if result.passed != (it["label"] == "in"):
            kind = "FALSE PASS" if result.passed else "FALSE REJECT"
            out.append(f"{kind:12s} {it['title'][:95]}\n{'':13s}terms +{result.include} -{result.exclude}\n"
                       f"{'':13s}label: {it['population']}: {it['reason']}")
    return out


def load(pool_path: str) -> tuple[list[dict], dict]:
    pool = {r["openalex_id"]: r for r in csv.DictReader(open(pool_path))}
    items = [{**it, "text": pool[it["openalex_id"]]["text"]} for it in json.loads(LABELS.read_text())["items"]]
    sizes, sampled = Counter(strata(r) for r in pool.values()), Counter(it["stratum"] for it in items)
    return items, {s: sizes[s] / sampled[s] for s in sampled}


def library(pool_path: str) -> None:
    """Which included manifest papers the new gate would drop. Dry run: nothing is written."""
    pool = {r["openalex_id"]: r for r in csv.DictReader(open(pool_path))}
    for r in manifest.included(manifest.load()):
        result = GATES["new"].check(pool[r["openalex_id"]]["text"])
        if not result.passed:
            print(f"drop {r['paper_id']:20s} {r['title'][:70]}\n     +{result.include} -{result.exclude}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--heldout", action="store_true", help="report on the held-out third (never tune on it)")
    ap.add_argument("--library", action="store_true")
    args = ap.parse_args()
    if args.library:
        return library(args.pool)
    items, weights = load(args.pool)
    split = "heldout" if args.heldout else "tuning"
    chosen = [it for it in items if it["split"] == split]
    print(f"{split}: {len(chosen)} items, {sum(it['label'] == 'in' for it in chosen)} in scope")
    for name, gate in GATES.items():
        print(f"{name:4s} raw {metrics(chosen, gate)}\n     weighted {metrics(chosen, gate, weights)}")
    for name, gate in GATES.items():
        errs = errors(chosen, gate)
        print(f"\n--- {name} gate: {len(errs)} errors ---")
        print("\n".join(errs))


if __name__ == "__main__":
    main()
