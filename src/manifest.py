"""Curation manifest: the metadata bridge between curate.py and ingest.py."""
import json
from pathlib import Path

from . import config


def load(path: Path = config.MANIFEST_PATH) -> list[dict]:
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    return [r for r in records if r.get("paper_id")]  # pre-v2 entries have no stable ID


def save(records: list[dict], path: Path = config.MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def merge(old: list[dict], new: list[dict]) -> list[dict]:
    by_id = {r["paper_id"]: r for r in old}
    by_id.update({r["paper_id"]: r for r in new})
    return sorted(by_id.values(), key=lambda r: r.get("score", 0), reverse=True)


def by_file(records: list[dict]) -> dict[str, dict]:
    return {r["pdf_file"]: r for r in records if r.get("pdf_file")}
