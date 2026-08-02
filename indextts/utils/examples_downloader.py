"""
Example file downloader that fetches example audio files from HuggingFace
Spaces or ModelScope Studio, depending on the detected network environment.

The example files are hosted at:
- HuggingFace: https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo
- ModelScope: https://modelscope.cn/studio/IndexTeam/IndexTTS-2-Demo

File names are determined from ``examples/cases.jsonl``.
"""

import json
import logging
import os
from typing import List, Set

import requests

from indextts.utils.network_detection import need_proxy

logger = logging.getLogger(__name__)

# Project root (indextts/utils/../../ = project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXAMPLES_DIR = os.path.join(_PROJECT_ROOT, "examples")
_TESTS_DIR = os.path.join(_PROJECT_ROOT, "tests")

# Remote repository configuration
_HF_RAW_URL = "https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/resolve/main"
_MS_RAW_URL = "https://modelscope.cn/studio/IndexTeam/IndexTTS-2-Demo/resolve/master"
# Additional files not listed in cases.jsonl but needed by the code
_EXTRA_FILES = [
    "voice_01.wav",  # used in infer.py and infer_v2.py __main__ blocks
]

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "IndexTTS/2.0"})


def _download_file(
    url: str,
    local_path: str,
    timeout: int = 60,
    min_size: int = 0,
    max_bytes: int = 0,
) -> None:
    """
    Download a file from a URL to a local path with validation.

    Raises RuntimeError if the server returns an error or non-binary content.
    When *max_bytes* > 0 the download stops after that many bytes (useful for
    reachability checks that don't need the full file).
    """
    resp = _SESSION.get(url, timeout=timeout, stream=True)
    try:
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"Server returned HTTP {resp.status_code} for {url}")
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError(
                f"Server returned HTML instead of binary file for {url} "
                f"(Content-Type: {content_type}). The URL may be invalid."
            )

        # Write to a temp file first, then rename atomically
        tmp_path = local_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                received = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    received += len(chunk)
                    if max_bytes and received >= max_bytes:
                        break
            if min_size and os.path.getsize(tmp_path) < min_size:
                raise RuntimeError(
                    f"Downloaded file is suspiciously small "
                    f"({os.path.getsize(tmp_path)} bytes) for {url}"
                )
            os.replace(tmp_path, local_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    finally:
        resp.close()


def get_required_files() -> List[str]:
    """
    Parse ``examples/cases.jsonl`` to determine which example files are needed.

    Returns a list of file names (without directory prefix).
    """
    cases_path = os.path.join(_EXAMPLES_DIR, "cases.jsonl")

    files: Set[str] = set(_EXTRA_FILES)

    if os.path.exists(cases_path):
        with open(cases_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("prompt_audio", "emo_audio"):
                    if key in case and case[key]:
                        files.add(case[key])

    return sorted(files)


def ensure_examples_available(force: bool = False) -> None:
    """
    Ensure all example files are available locally.
    Downloads missing files from the appropriate remote source.

    Call this at startup before using example files.
    """
    required = get_required_files()
    if not required:
        return

    os.makedirs(_EXAMPLES_DIR, exist_ok=True)
    base_url = _MS_RAW_URL if need_proxy() else _HF_RAW_URL

    for filename in required:
        local_path = os.path.join(_EXAMPLES_DIR, filename)
        if os.path.exists(local_path) and not force:
            continue
        url = f"{base_url}/examples/{filename}"
        try:
            _download_file(url, local_path, min_size=100, timeout=120)
        except Exception as e:
            logger.warning(f"Failed to download {filename}: {e}")


def download_test_sample(force: bool = False) -> str:
    """
    Download the test sample audio file (``tests/sample_prompt.wav``).

    Returns the local path if the file is available.
    Raises RuntimeError on failure.
    """
    os.makedirs(_TESTS_DIR, exist_ok=True)
    local_path = os.path.join(_TESTS_DIR, "sample_prompt.wav")

    if os.path.exists(local_path) and not force:
        return local_path

    base_url = _MS_RAW_URL if need_proxy() else _HF_RAW_URL
    url = f"{base_url}/examples/voice_01.wav"

    _download_file(url, local_path, min_size=100, timeout=120)
    return local_path

# Alias for backward compatibility (used by tests)
ensure_test_sample_available = download_test_sample
