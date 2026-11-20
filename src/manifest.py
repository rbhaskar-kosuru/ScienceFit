"""Curation manifest: the metadata bridge between curate.py and ingest.py.

Every paper curation has judged is listed, with `included` and, when excluded, a `reason`.
"""
import os
import json
from pathlib import Path

from . import config


def load(path: Path = config.MANIFEST_PATH) -> list[dict]:
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    return [r for r in records if r.get("paper_id")]  # pre-v2 entries have no stable ID


def save(records: list[dict], path: Path = config.MANIFEST_PATH) -> None:
    """Write to a temp file, then swap it in, so a crash mid-write never leaves a half-written manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    os.replace(tmp, path)


def included(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("included")]


def excluded(records: list[dict]) -> list[dict]:
    return [r for r in records if not r.get("included")]
