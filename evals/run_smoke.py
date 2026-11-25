"""Smoke eval: run each question in smoke.json through RAG().ask() and report refusals and citations.

    mkdir -p evals/runs
    python -m evals.run_smoke > evals/runs/new.txt
    python -m evals.run_smoke --model MODEL --base-url URL --prompt-version v1   # override .env / defaults

A case passes if it must refuse and did, or if it answered and cited at least one expected paper.
It also shows the sections of cited chunks and flags answers cited only from introduction chunks, which
describe other studies (a pass/fail it doesn't change). The excerpts the model saw follow each answer, so keep
outputs in evals/runs/ (gitignored: no paper text in the repo). A question that errors or comes back empty is
flagged, and the last line says how many of the questions were answered. Generation uses LLM_TEMPERATURE and LLM_SEED.
"""
import argparse
import json
import re
from pathlib import Path

CASES = Path(__file__).with_name("smoke.json")
EXCERPT = re.compile(r'<excerpt id="([^"]+)"(?: section="([^"]+)")?( secondhand="true")?>\n(.*?)\n</excerpt>', re.S)


class Recorder:
    """Passes generate() calls to the client and keeps the last prompt, so the report shows what the model saw."""
    def __init__(self, client):
        self.client = client
        self.prompt = ""

    def generate(self, system: str, prompt: str) -> str:
        self.prompt = prompt
        return self.client.generate(system, prompt)


def excerpts(prompt: str) -> list[dict]:
    return [{"id": pid, "section": section or "unknown", "secondhand": bool(mark), "text": text}
            for pid, section, mark, text in EXCERPT.findall(prompt)]


def check(case: dict, res: dict) -> dict:
    retrieved = [s["paper_id"] for s in res["sources"]]
    cited = [s["paper_id"] for s in res["sources"] if s["cited"]]
    cited_sections = {s["paper_id"]: [c["section"] for c in s.get("chunks", [])] for s in res["sources"] if s["cited"]}
    all_sections = [sec for secs in cited_sections.values() for sec in secs]
    expected = set(case.get("expect_any", []))
    if case.get("must_refuse"):
        passed = res["refused"]
    else:
        passed = not res["refused"] and bool(expected & set(cited))
    return {"passed": passed, "refused": res["refused"], "retrieved": retrieved, "cited": cited,
            "unverified": res["unverified_citations"], "expected_retrieved": bool(expected & set(retrieved)),
            "cited_sections": cited_sections,
            "intro_only": bool(all_sections) and all(sec == "introduction" for sec in all_sections)}


def report(case: dict, res: dict, result: dict, shown: list[dict]) -> str:
    expect = "must refuse" if case.get("must_refuse") else " | ".join(case["expect_any"])
    lines = [
        f"{case['id']}  {'PASS' if result['passed'] else 'FAIL'}  {case['question']}",
        f"    expected:   {expect}",
        f"    refused:    {'yes' if result['refused'] else 'no'}",
    ]
    if not case.get("must_refuse"):
        lines.append(f"    expected paper retrieved: {'yes' if result['expected_retrieved'] else 'no'}")
    lines += [
        f"    retrieved:  {', '.join(result['retrieved']) or '-'}",
        f"    cited:      {', '.join(result['cited']) or '-'}",
        f"    sections:   " + ("; ".join(f"{pid}: {', '.join(secs)}" for pid, secs in result["cited_sections"].items()) or "-"),
        f"    unverified: {', '.join(result['unverified']) or '-'}",
    ]
    if result["intro_only"]:
        lines.append("    WARNING:    cited only from introduction chunks (background, not the papers' own findings)")
    lines += [
        "    answer:     " + res["answer"].strip().replace("\n", "\n                "),
        f"    excerpts:   {len(shown)}",
    ]
    for e in shown:
        lines += [f"      [{e['id']}] section: {e['section']}{' secondhand' if e['secondhand'] else ''}", "        " + e["text"].replace("\n", "\n        ")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from src import config
    from src.llm import LLMClient
    from src.redact import redact
    from src.rag import RAG  # the reranker loads in RAG(); kept out of import so tests can use check()

    ap = argparse.ArgumentParser(description="Run the smoke eval.")
    ap.add_argument("--model", help="default: LLM_MODEL")
    ap.add_argument("--base-url", help="default: LLM_BASE_URL")
    ap.add_argument("--prompt-version", default=config.LLM_PROMPT_VERSION, help="default: LLM_PROMPT_VERSION")
    args = ap.parse_args(argv)

    recorder = Recorder(LLMClient(base_url=args.base_url, model=args.model))
    print(f"LLM_MODEL: {recorder.client.model}\nPROMPT_VERSION: {args.prompt_version}\n")
    rag = RAG(generate=recorder.generate, prompt_version=args.prompt_version)
    cases = json.loads(CASES.read_text())
    passed, failed = 0, []
    for case in cases:
        recorder.prompt = ""  # a refusal sends no prompt
        try:
            res = rag.ask(case["question"])
            problem = None if res["answer"].strip() else "empty answer"
        except Exception as e:  # one failed question must not hide the others
            problem = f"{type(e).__name__}: {redact(e)}"
        if problem:
            failed.append(case["id"])
            print(f"{case['id']}  ERROR  {case['question']}\n    {problem}\n", flush=True)
            continue
        result = check(case, res)
        passed += result["passed"]
        print(report(case, res, result, excerpts(recorder.prompt)) + "\n", flush=True)
    print(f"Answered {len(cases) - len(failed)} of {len(cases)}" + (f"  NOT ANSWERED: {', '.join(failed)}" if failed else ""))
    print(f"Passed {passed}/{len(cases)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
