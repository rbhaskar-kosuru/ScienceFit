"""Label a stratified sample of curate candidates against config.CURATE_SCOPE with the configured LLM (gpt-oss-120b).

    python -m evals.gate_label evals/runs/gate_pool_2026-11-27.csv           # sample (once), then label; resumable
    python -m evals.gate_label evals/runs/gate_pool_2026-11-27.csv --relabel # re-ask items a scope change may flip

One fresh-context call per candidate. Labels go to evals/gate_labels.json after every call (IDs, titles, labels:
no abstracts), so a stop on Groq's daily limit loses nothing: run the same command again to resume. Tokens are
counted per day; a run stops at DAILY_BUDGET (Groq free tier: 200K/day).

Scope v2 (2026-11-27) added bodyweight/band/kettlebell/suspension loads and nutrition with muscle outcomes. It only
widens "in", so an earlier "in" stands; --relabel re-asks the earlier "out" items that weren't clinical or animal.
"""
import csv
import json
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

from src import config
from src.llm import LLMClient, LLMError
from src.redact import redact

SEED = 42
LABELS = config.ROOT / "evals" / "gate_labels.json"
SAMPLE = config.ROOT / "evals" / "runs" / "gate_sample.json"  # sampled IDs + strata (no text, but kept with the pool)
STRATA = {"clear_reject": 42, "near_reject": 70, "near_pass": 60, "clear_pass": 80}  # 252: what ~150K tokens buys
HELD_OUT = 1 / 3
DAILY_BUDGET = 190_000  # Groq free tier is 200K/day; leaves room for the day's other calls
SCOPE_VERSION = 2
NOT_FLIPPABLE = re.compile(r"animal|rats?\b|mice|mouse|rodent|in vitro|cell", re.I)  # reasons scope v2 can't change
MAX_CHARS = 1000  # abstract cut: keeps the budget; design and population are stated early
POPULATIONS = {"youth", "adult", "older", "mixed", "clinical"}
EXTRA = {"reasoning_effort": "low", "max_completion_tokens": 700}

SYSTEM = (
    "Screen a paper for a resistance-training library. SCOPE: " + config.CURATE_SCOPE + "\n"
    "Reviews/meta-analyses of in-scope studies are in. 'Clinical trial' or 'clinical relevance' alone is not a "
    "clinical population; healthy people with obesity, sarcopenia or frailty are not patients. No abstract: use "
    "the title. Bodyweight, band, kettlebell and suspension training count as resistance training. Nutrition or "
    "supplement studies in healthy people are in if they measure muscle mass, lean mass, strength or muscle protein "
    "synthesis, even without training; diet trials measuring only fat or body weight are out. "
    'JSON only: {"label":"in|out","population":"youth|adult|older|mixed|clinical","reason":"<=15 words"}'
)


def strata(row: dict) -> str:
    count = lambda s: len([t for t in s.split("; ") if t])
    near = count(row["gate_include"]) == 1 or count(row["gate_exclude"]) == 1
    return ("near_" if near else "clear_") + ("pass" if row["gate_pass"] == "True" else "reject")


def draw_sample(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    picked = []
    for name, n in STRATA.items():
        group = sorted((r for r in rows if strata(r) == name), key=lambda r: r["openalex_id"])
        picked += [{"openalex_id": r["openalex_id"], "stratum": name} for r in rng.sample(group, min(n, len(group)))]
    ids = sorted(p["openalex_id"] for p in picked)
    held = set(random.Random(SEED + 1).sample(ids, round(len(ids) * HELD_OUT)))
    for p in picked:
        p["split"] = "heldout" if p["openalex_id"] in held else "tuning"
    return picked


def parse(answer: str) -> dict:
    m = re.search(r"\{.*\}", answer, re.S)
    data = json.loads(m.group(0)) if m else {}
    label, population = str(data.get("label", "")).lower(), str(data.get("population", "")).lower()
    if label not in {"in", "out"} or population not in POPULATIONS:
        raise ValueError("unparseable label")
    return {"label": label, "population": population, "reason": str(data.get("reason", ""))[:200]}


def needs_relabel(item: dict) -> bool:
    """An earlier-scope "out" that the wider scope might turn "in": not clinical, not animal or cell work."""
    return (item.get("scope_version", 1) < SCOPE_VERSION and item["label"] == "out"
            and item["population"] != "clinical" and not NOT_FLIPPABLE.search(item["reason"]))


def ask(client: LLMClient, row: dict, labels: dict) -> dict | None:
    abstract = row["text"][len(row["title"]):].strip()[:MAX_CHARS] or "(no abstract)"
    prompt = f"TITLE: {row['title']}\nABSTRACT: {abstract}"
    for _ in range(2):  # one retry for an unparseable reply
        try:
            return parse(client.generate(SYSTEM, prompt))
        except ValueError:
            continue
        finally:
            spent = int(client.last_usage.get("total_tokens") or 0)
            labels["tokens_used"] += spent
            day = labels.setdefault("tokens_by_day", {})
            day[str(date.today())] = day.get(str(date.today()), 0) + spent
    return None


def ask_patiently(client: LLMClient, row: dict, labels: dict, waits: int = 10, sleep=time.sleep) -> dict | None:
    """A per-minute 429 can ask for a longer wait than LLMClient retries; wait it out. The daily limit: give up."""
    for attempt in range(waits + 1):
        try:
            return ask(client, row, labels)
        except LLMError as e:
            if "HTTP 429" not in str(e) or "per day" in str(e) or attempt == waits:
                raise
            sleep(30)


def over_budget(labels: dict) -> bool:
    if labels.get("tokens_by_day", {}).get(str(date.today()), 0) >= DAILY_BUDGET:
        print(f"Today's budget of {DAILY_BUDGET} tokens reached; run again tomorrow to resume.")
        return True
    return False


def main(pool_path: str, relabel: bool = False) -> int:
    rows = {r["openalex_id"]: r for r in csv.DictReader(open(pool_path))}
    if SAMPLE.exists():
        sample = json.loads(SAMPLE.read_text())
    else:
        sample = draw_sample(list(rows.values()))
        SAMPLE.write_text(json.dumps(sample, indent=1))
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {"tokens_used": 0, "items": []}
    client = LLMClient(extra_params=EXTRA)
    save = lambda: LABELS.write_text(json.dumps(labels, indent=1, ensure_ascii=False))
    if relabel:
        todo = [it for it in labels["items"] if needs_relabel(it)]
        flipped = 0
        for item in todo:
            if over_budget(labels):
                break
            try:
                result = ask_patiently(client, rows[item["openalex_id"]], labels)
            except LLMError as e:
                print(f"Paused: {redact(e)[:200]}. Run the same command again to resume.")
                break
            if result:
                flipped += result["label"] != item["label"]
                item.update({"label_scope_v1": item["label"], **result, "scope_version": SCOPE_VERSION})
                save()
        left = sum(needs_relabel(it) for it in labels["items"])
        print(f"Re-labelled {len(todo) - left} of {len(todo)} ({flipped} changed). Tokens used: {labels['tokens_used']}.")
        return 0 if not left else 1
    done = {item["openalex_id"] for item in labels["items"]}
    for s in sample:
        if s["openalex_id"] in done:
            continue
        if over_budget(labels):
            break
        row = rows[s["openalex_id"]]
        try:
            result = ask_patiently(client, row, labels)
        except LLMError as e:
            print(f"Paused: {redact(e)[:200]}. Run the same command again to resume.")
            break
        if result is None:
            print(f"  unparseable twice: {s['openalex_id']}")
            continue
        labels["items"].append({"openalex_id": s["openalex_id"], "title": row["title"], "stratum": s["stratum"],
                                "split": s["split"], **result, "scope_version": SCOPE_VERSION})
        save()
        if len(labels["items"]) % 20 == 0:
            print(f"  {len(labels['items'])} labelled, {labels['tokens_used']} tokens")
    n = len(labels["items"])
    save()
    print(f"Labelled {n} of {len(sample)}. Tokens used: {labels['tokens_used']}.")
    return 0 if n == len(sample) else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], relabel="--relabel" in sys.argv[2:]))
