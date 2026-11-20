"""Smoke eval: run each question in smoke.json through RAG().ask() and report refusals and citations.

    python -m evals.run_smoke > evals/baseline.txt

A case passes if it must refuse and did, or if it answered and cited at least one expected paper.
It also shows the sections of cited chunks and flags answers cited only from introduction chunks, which
describe other studies (a pass/fail it doesn't change). Generation uses LLM_TEMPERATURE and LLM_SEED.
"""
import json
from pathlib import Path

CASES = Path(__file__).with_name("smoke.json")


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


def report(case: dict, res: dict, result: dict) -> str:
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
    ]
    return "\n".join(lines)


def main() -> None:
    from src.rag import RAG  # loads the reranker; kept out of import so tests can use check()

    rag, cases = RAG(), json.loads(CASES.read_text())
    passed = 0
    for case in cases:
        res = rag.ask(case["question"])
        result = check(case, res)
        passed += result["passed"]
        print(report(case, res, result) + "\n")
    print(f"Passed {passed}/{len(cases)}")


if __name__ == "__main__":
    main()
