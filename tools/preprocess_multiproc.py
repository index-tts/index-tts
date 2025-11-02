#!/usr/bin/env python3
"""
Launch multiple preprocessing workers in parallel while preserving the ability
to resume from previous runs. Each worker operates on a disjoint shard of the
manifest and writes outputs to a temporary worker directory; once finished, the
results are merged back into the main output directory and the worker assets are
removed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import time


MANIFEST_TRAIN = "train_manifest.jsonl"
MANIFEST_VAL = "val_manifest.jsonl"
STATS_FILE = "stats.json"
FEATURE_SUBDIRS = ["codes", "condition", "emo_vec", "text_ids"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IndexTTS2 preprocessing across multiple worker processes."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Source dataset manifest JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece model path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/config_finetune.yaml"),
        help="Model config YAML (default: checkpoints/config_finetune.yaml).",
    )
    parser.add_argument(
        "--gpt-checkpoint",
        type=Path,
        default=Path("checkpoints/gpt.pth"),
        help="Base GPT checkpoint (default: checkpoints/gpt.pth).",
    )
    parser.add_argument("--language", type=str, default="ja", help="Language hint for normaliser.")
    parser.add_argument("--device", type=str, default="cuda", help="Device string (default: cuda).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per worker.")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker threads per process.")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel preprocessing processes to launch.",
    )
    parser.add_argument(
        "--launch-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between launching each worker process (default: 0).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Validation ratio passed to preprocess_data.py (default: 0.01).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Forward --skip-existing to each worker.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples to process overall (0 = all remaining).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to preprocess_data.py after a '--'.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Directory to use for Hugging Face caches/offline assets (default: ./hf_cache).",
    )
    return parser.parse_args()


def cache_has_required_assets(cache_dir: Path) -> bool:
    required = [
        cache_dir / "models--facebook--seamless-m4t-medium",
        cache_dir / "models--amphion--MaskGCT",
    ]
    return all(path.exists() for path in required)


def merge_manifest_shards(base_path: Path, pattern: str, target: Path) -> None:
    shard_files = sorted(base_path.glob(pattern))
    if not shard_files:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as dst:
        for shard in shard_files:
            with shard.open("r", encoding="utf-8") as src:
                shutil.copyfileobj(src, dst)
            shard.unlink()


def move_feature_tree(source_dir: Path, dest_dir: Path) -> None:
    for sub in FEATURE_SUBDIRS:
        src_sub = source_dir / sub
        if not src_sub.exists():
            continue
        dst_sub = dest_dir / sub
        dst_sub.mkdir(parents=True, exist_ok=True)
        for path in src_sub.iterdir():
            target_path = dst_sub / path.name
            if target_path.exists():
                # Already present; assume prior run completed this sample.
                path.unlink()
            else:
                shutil.move(str(path), str(target_path))


def consolidate_previous_shards(output_dir: Path) -> None:
    # Merge stray worker manifests if the previous run exited early.
    merge_manifest_shards(output_dir, "train_manifest.worker_*.jsonl", output_dir / MANIFEST_TRAIN)
    merge_manifest_shards(output_dir, "val_manifest.worker_*.jsonl", output_dir / MANIFEST_VAL)
    for worker_dir in sorted(output_dir.glob("worker_*")):
        merge_worker_results(worker_dir, output_dir)


def load_processed_ids(manifest_paths: Iterable[Path]) -> set[str]:
    ids: set[str] = set()
    for path in manifest_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = record.get("id")
                if sample_id:
                    ids.add(sample_id)
    return ids


def remaining_manifest_entries(
    manifest: Path,
    processed_ids: set[str],
    max_samples: int,
) -> List[str]:
    remaining: List[str] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("id")
            if not sample_id or sample_id in processed_ids:
                continue
            remaining.append(line)
            if max_samples and len(remaining) >= max_samples:
                break
    return remaining


def write_chunks(
    lines: Sequence[str],
    num_chunks: int,
    work_dir: Path,
    prefix: str,
) -> List[tuple[Path, int]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[tuple[Path, int]] = []
    if not lines:
        return chunks
    chunk_size = math.ceil(len(lines) / num_chunks)
    for idx in range(num_chunks):
        start = idx * chunk_size
        if start >= len(lines):
            break
        end = min(len(lines), start + chunk_size)
        chunk_path = work_dir / f"{prefix}_chunk_{idx:02d}.jsonl"
        with chunk_path.open("w", encoding="utf-8") as handle:
            handle.writelines(lines[start:end])
        chunks.append((chunk_path, end - start))
    return chunks


def launch_worker(
    chunk_manifest: Path,
    worker_output: Path,
    args: argparse.Namespace,
    hf_env: Dict[str, str],
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-u",
        "tools/preprocess_data.py",
        "--manifest",
        str(chunk_manifest),
        "--output-dir",
        str(worker_output),
        "--tokenizer",
        str(args.tokenizer),
        "--config",
        str(args.config),
        "--gpt-checkpoint",
        str(args.gpt_checkpoint),
        "--language",
        args.language,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--val-ratio",
        str(args.val_ratio),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.extra_args:
        cmd.append("--")
        cmd.extend(args.extra_args)
    env = os.environ.copy()
    for key, value in hf_env.items():
        env.setdefault(key, value)
    return subprocess.Popen(cmd, env=env)


def append_and_remove(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as dst, source.open("r", encoding="utf-8") as src:
        shutil.copyfileobj(src, dst)
    source.unlink()


def merge_worker_results(worker_dir: Path, main_output: Path) -> None:
    append_and_remove(worker_dir / MANIFEST_TRAIN, main_output / MANIFEST_TRAIN)
    append_and_remove(worker_dir / MANIFEST_VAL, main_output / MANIFEST_VAL)
    move_feature_tree(worker_dir, main_output)
    stats_path = worker_dir / STATS_FILE
    if stats_path.exists():
        stats_path.unlink()
    shutil.rmtree(worker_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_cache_dir = (
        args.hf_cache_dir.expanduser().resolve()
        if args.hf_cache_dir
        else (Path.cwd() / "hf_cache")
    )
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    hf_env = {
        "HF_HOME": str(hf_cache_dir),
        "HF_HUB_CACHE": str(hf_cache_dir),
        "HF_DATASETS_CACHE": str(hf_cache_dir),
    }
    if cache_has_required_assets(hf_cache_dir):
        hf_env["HF_HUB_OFFLINE"] = "1"
        hf_env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        print("[preprocess_multiproc] HF cache missing SeamlessM4T/MaskGCT weights; running in online mode to populate cache.")

    consolidate_previous_shards(output_dir)

    processed_ids = load_processed_ids(
        [
            output_dir / MANIFEST_TRAIN,
            output_dir / MANIFEST_VAL,
        ]
    )

    manifest_path = args.manifest.expanduser().resolve()
    remaining_lines = remaining_manifest_entries(manifest_path, processed_ids, args.max_samples)
    if not remaining_lines:
        print("No remaining samples. Nothing to do.")
        return

    num_processes = min(max(1, args.num_processes), len(remaining_lines))
    chunk_root = manifest_path.parent.parent / "_preprocess_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_prefix = manifest_path.parent.name
    chunks = write_chunks(remaining_lines, num_processes, chunk_root, chunk_prefix)
    if not chunks:
        print("Failed to create chunk manifests.")
        return

    total_samples = sum(count for _, count in chunks)
    print(
        f"[preprocess_multiproc] Remaining samples: {len(remaining_lines)} "
        f"(assigned {total_samples} across {len(chunks)} workers)."
    )
    for idx, (chunk_path, count) in enumerate(chunks):
        print(f"[preprocess_multiproc] Worker {idx:02d} -> {chunk_path} ({count} samples)")

    processes: List[subprocess.Popen] = []
    worker_dirs: List[Path] = []
    try:
        for idx, (chunk_path, count) in enumerate(chunks):
            worker_dir = output_dir / f"worker_{idx:02d}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[preprocess_multiproc] Launching worker {idx:02d} "
                f"({count} samples) -> dir {worker_dir}"
            )
            proc = launch_worker(chunk_path, worker_dir, args, hf_env)
            processes.append(proc)
            worker_dirs.append(worker_dir)
            if idx < len(chunks) - 1 and args.launch_delay > 0:
                print(
                    f"[preprocess_multiproc] Sleeping {args.launch_delay:.2f}s before launching next worker..."
                )
                time.sleep(args.launch_delay)

        return_codes = [proc.wait() for proc in processes]
        if any(code != 0 for code in return_codes):
            for idx, code in enumerate(return_codes):
                if code != 0:
                    print(f"Worker {idx} exited with code {code}", file=sys.stderr)
            raise RuntimeError("One or more workers failed. Check logs above.")

        for worker_dir in worker_dirs:
            merge_worker_results(worker_dir, output_dir)

    finally:
        for chunk_path, _ in chunks:
            if chunk_path.exists():
                chunk_path.unlink()
        try:
            if chunk_root.exists() and not any(chunk_root.iterdir()):
                chunk_root.rmdir()
        except OSError:
            pass

    print("All workers completed successfully.")


if __name__ == "__main__":
    main()

