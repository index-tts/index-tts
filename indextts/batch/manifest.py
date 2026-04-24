"""Manifest I/O and progress tracking for batch inference.

Manifest format (JSONL, one object per line):

    {"id": "sample_0001", "text": "Hello, world.", "ref_audio": "/abs/ref.wav"}

Optional keys: ``out_path`` (overrides the default ``<output_dir>/<id>.wav``).

Progress file (JSONL, appended atomically by the runner):

    {"id": "sample_0001", "status": "done",     "out_path": "...", "seconds": 1.23}
    {"id": "sample_0002", "status": "skipped",  "reason": "truncated_output"}
    {"id": "sample_0003", "status": "failed",   "error": "CUDA out of memory"}
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Iterable, Iterator, Optional
from urllib.parse import quote


@dataclasses.dataclass
class Job:
    id: str
    text: str
    ref_audio: str
    out_path: Optional[str] = None  # resolved by the runner if None

    def resolved_out_path(self, output_dir: str) -> str:
        if self.out_path:
            return self.out_path
        safe_id = quote(self.id, safe="._-") or "job"
        return os.path.join(output_dir, f"{safe_id}.wav")


def _validate(obj: dict, line_no: int) -> Job:
    for key in ("id", "text", "ref_audio"):
        if key not in obj:
            raise ValueError(f"manifest line {line_no}: missing required key {key!r}")
        if not isinstance(obj[key], str):
            raise ValueError(f"manifest line {line_no}: {key!r} must be a string")
    out_path = obj.get("out_path")
    if out_path is not None and not isinstance(out_path, str):
        raise ValueError(f"manifest line {line_no}: 'out_path' must be a string if present")
    return Job(id=obj["id"], text=obj["text"], ref_audio=obj["ref_audio"], out_path=out_path)


def read_manifest(path: str) -> Iterator[Job]:
    """Yield Job records from a JSONL manifest. Blank lines are ignored."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"manifest line {line_no}: invalid JSON ({e})") from e
            yield _validate(obj, line_no)


def load_processed_ids(progress_path: str) -> set[str]:
    """Return the set of job IDs already recorded in a progress file.

    Any line whose ``status`` is ``done`` or ``skipped`` is considered processed
    and will be filtered out on resume. ``failed`` entries are NOT filtered,
    so retries happen on the next run.
    """
    processed: set[str] = set()
    if not os.path.exists(progress_path):
        return processed
    with open(progress_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("status") in ("done", "skipped") and "id" in obj:
                processed.add(obj["id"])
    return processed


def filter_remaining(jobs: Iterable[Job], processed: set[str]) -> list[Job]:
    return [j for j in jobs if j.id not in processed]
