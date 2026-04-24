"""Main-process orchestration for multi-GPU batch inference.

Usage::

    python -m indextts.batch \
        --manifest jobs.jsonl \
        --output-dir out/ \
        --model-dir checkpoints/ \
        --cfg checkpoints/config.yaml \
        --gpus 0,1,2,3,4,5,6,7 \
        --workers-per-gpu 1 \
        --fp16

Resume is automatic: rerunning with the same ``--progress-file`` skips any
sample that already has a ``done`` or ``skipped`` record.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
import signal
import sys
import time
from typing import List

from .manifest import Job, filter_remaining, load_processed_ids, read_manifest
from .worker import SHUTDOWN, WorkerConfig, worker_main


def _parse_gpus(spec: str) -> List[int]:
    spec = spec.strip()
    if not spec:
        raise argparse.ArgumentTypeError("--gpus cannot be empty")
    ids: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--gpus: invalid GPU id {part!r}"
            ) from e
    if not ids:
        raise argparse.ArgumentTypeError("--gpus parsed to empty list")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("--gpus contains duplicates")
    return ids


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="indextts.batch",
        description="Offline multi-GPU batch inference for IndexTTS2 (voice cloning).",
    )
    p.add_argument("--manifest", required=True,
                   help="Path to input JSONL manifest.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write output .wav files.")
    p.add_argument("--model-dir", default="checkpoints",
                   help="IndexTTS2 model directory (default: checkpoints).")
    p.add_argument("--cfg", default=None,
                   help="Path to config.yaml (default: <model-dir>/config.yaml).")
    p.add_argument("--gpus", type=_parse_gpus, default=[0],
                   help="Comma-separated list of GPU ids, e.g. '0,1,2,3,4,5,6,7'.")
    p.add_argument("--workers-per-gpu", type=int, default=1,
                   help="Number of worker processes (model replicas) per GPU.")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use fp16 inference (default: enabled).")
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   help="Disable fp16 inference.")
    p.add_argument("--cuda-kernel", dest="cuda_kernel", action="store_true", default=True,
                   help="Enable BigVGAN custom CUDA kernel (default: enabled).")
    p.add_argument("--no-cuda-kernel", dest="cuda_kernel", action="store_false",
                   help="Disable BigVGAN custom CUDA kernel.")
    p.add_argument("--torch-compile", action="store_true", default=False,
                   help="Enable torch.compile for s2mel (slow startup; off by default).")
    p.add_argument("--max-text-tokens-per-segment", type=int, default=120,
                   help="Text tokens per GPT generation segment.")
    p.add_argument("--max-mel-tokens", type=int, default=1500,
                   help="Mel tokens per segment before GPT truncates.")
    p.add_argument("--retries", type=int, default=2,
                   help="Number of retries on transient errors (e.g. CUDA OOM).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files instead of skipping.")
    p.add_argument("--progress-file", default=None,
                   help="Path to progress JSONL (default: <output-dir>/progress.jsonl).")
    p.add_argument("--queue-size", type=int, default=0,
                   help="Max in-flight jobs in the shared queue (0 = 4 * total_workers).")
    return p.parse_args(argv)


def _build_worker_configs(args) -> List[WorkerConfig]:
    cfg_path = args.cfg or os.path.join(args.model_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config not found: {cfg_path}")
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"model dir not found: {args.model_dir}")

    configs: List[WorkerConfig] = []
    for gpu_id in args.gpus:
        for slot in range(args.workers_per_gpu):
            configs.append(WorkerConfig(
                gpu_id=gpu_id,
                slot=slot,
                model_dir=args.model_dir,
                cfg_path=cfg_path,
                output_dir=args.output_dir,
                use_fp16=args.fp16,
                use_cuda_kernel=args.cuda_kernel,
                use_torch_compile=args.torch_compile,
                max_text_tokens_per_segment=args.max_text_tokens_per_segment,
                max_mel_tokens=args.max_mel_tokens,
                retries=args.retries,
                overwrite=args.overwrite,
            ))
    return configs


def _worker_launcher(cfg: WorkerConfig, job_q, progress_q) -> None:
    # CUDA_VISIBLE_DEVICES must be set before the child imports torch. With the
    # ``spawn`` start method, no CUDA context is inherited, so setting it here
    # (before worker_main imports torch) is sufficient.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Ignore SIGINT in workers; the main process handles Ctrl-C and drains via
    # shutdown sentinels. Without this, Ctrl-C in the terminal would kill
    # workers mid-inference because SIGINT is delivered to the process group.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker_main(cfg, job_q, progress_q)


def _spawn_worker(ctx: mp.context.BaseContext, cfg: WorkerConfig, job_q, progress_q) -> mp.Process:
    p = ctx.Process(
        target=_worker_launcher,
        args=(cfg, job_q, progress_q),
        name=f"indextts-worker-gpu{cfg.gpu_id}-slot{cfg.slot}",
        daemon=False,
    )
    p.start()
    return p


def _format_event_for_log(event: dict) -> str:
    # Keep event order stable so log files diff nicely.
    ordered_keys = ["id", "status", "out_path", "seconds", "reason", "error", "gpu", "slot"]
    ordered = {k: event[k] for k in ordered_keys if k in event}
    for k, v in event.items():
        if k not in ordered:
            ordered[k] = v
    return json.dumps(ordered, ensure_ascii=False)


def run(args) -> int:
    os.makedirs(args.output_dir, exist_ok=True)
    progress_path = args.progress_file or os.path.join(args.output_dir, "progress.jsonl")

    print(f">> manifest:     {args.manifest}", flush=True)
    print(f">> output_dir:   {args.output_dir}", flush=True)
    print(f">> progress:     {progress_path}", flush=True)

    print(">> loading manifest...", flush=True)
    all_jobs = list(read_manifest(args.manifest))
    print(f">> manifest: {len(all_jobs)} total samples", flush=True)

    processed = load_processed_ids(progress_path)
    if processed:
        print(f">> resume: {len(processed)} samples already processed", flush=True)
    jobs = filter_remaining(all_jobs, processed)
    if not jobs:
        print(">> nothing to do (all samples already processed)", flush=True)
        return 0
    print(f">> remaining: {len(jobs)} samples to process", flush=True)

    # Sanity: dedupe job IDs within this run (resume above already handles
    # cross-run dedupe).
    seen: set[str] = set()
    uniq: List[Job] = []
    dupes = 0
    for j in jobs:
        if j.id in seen:
            dupes += 1
            continue
        seen.add(j.id)
        uniq.append(j)
    if dupes:
        print(f">> WARNING: dropped {dupes} duplicate job ids in manifest", flush=True)
    jobs = uniq

    worker_cfgs = _build_worker_configs(args)
    total_workers = len(worker_cfgs)
    queue_size = args.queue_size if args.queue_size > 0 else 4 * total_workers

    ctx = mp.get_context("spawn")
    job_q = ctx.Queue(maxsize=queue_size)
    progress_q = ctx.Queue()

    print(f">> starting {total_workers} worker(s) across {len(args.gpus)} GPU(s)...", flush=True)
    workers = [_spawn_worker(ctx, c, job_q, progress_q) for c in worker_cfgs]

    # Install a clean shutdown path on SIGINT/SIGTERM.
    stopping = {"flag": False}

    def _signal_handler(signum, _frame):
        if not stopping["flag"]:
            print(f">> received signal {signum}; draining...", flush=True)
            stopping["flag"] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Open progress file for append.
    progress_f = open(progress_path, "a", encoding="utf-8", buffering=1)

    ready_workers = 0
    exited_workers = 0
    done_count = 0
    skip_count = 0
    fail_count = 0

    t_start = time.perf_counter()
    feed_idx = 0
    sentinels_pushed = 0  # how many shutdown sentinels have entered the queue

    def _all_workers_dead() -> bool:
        return not any(p.is_alive() for p in workers)

    try:
        while True:
            # 1. Feed jobs.
            while (
                not stopping["flag"]
                and feed_idx < len(jobs)
            ):
                try:
                    job_q.put(jobs[feed_idx], timeout=0.05)
                    feed_idx += 1
                except queue_mod.Full:
                    break

            # 2. Push shutdown sentinels once feeding is done (or we're stopping).
            #    Push exactly one per worker, across loop iterations if needed.
            if (stopping["flag"] or feed_idx >= len(jobs)) and sentinels_pushed < total_workers:
                try:
                    job_q.put(SHUTDOWN, timeout=0.05)
                    sentinels_pushed += 1
                except queue_mod.Full:
                    pass

            # 3. Drain one progress event.
            try:
                event = progress_q.get(timeout=0.25)
            except queue_mod.Empty:
                # 4a. Exit if every worker has signalled exit OR all are dead.
                if exited_workers >= total_workers or _all_workers_dead():
                    break
                continue

            jid = event.get("id", "")
            status = event.get("status", "")

            if jid == "__worker_ready__":
                ready_workers += 1
                print(f">> worker ready: gpu={event.get('gpu')} slot={event.get('slot')} "
                      f"({ready_workers}/{total_workers})", flush=True)
                continue
            if jid == "__worker_exit__":
                exited_workers += 1
                if exited_workers >= total_workers:
                    break
                continue
            if jid == "__worker_init__":
                print(f">> WORKER INIT FAILED: {event.get('error')}", flush=True)
                exited_workers += 1
                fail_count += 1
                progress_f.write(_format_event_for_log(event) + "\n")
                if exited_workers >= total_workers:
                    break
                continue

            # Real job event.
            progress_f.write(_format_event_for_log(event) + "\n")
            if status == "done":
                done_count += 1
            elif status == "skipped":
                skip_count += 1
            elif status == "failed":
                fail_count += 1

            total = done_count + skip_count + fail_count
            if total and total % 50 == 0:
                elapsed = time.perf_counter() - t_start
                rate = total / elapsed if elapsed > 0 else 0.0
                remaining = len(jobs) - total
                eta_s = remaining / rate if rate > 0 else float("inf")
                print(
                    f">> progress: {total}/{len(jobs)}  "
                    f"done={done_count} skip={skip_count} fail={fail_count}  "
                    f"rate={rate:.2f}/s  eta={eta_s/60:.1f} min",
                    flush=True,
                )
    finally:
        progress_f.close()

        # Make sure children actually exit.
        for p in workers:
            p.join(timeout=30)
        for p in workers:
            if p.is_alive():
                print(f">> force-terminating {p.name}", flush=True)
                p.terminate()
                p.join(timeout=10)

    elapsed = time.perf_counter() - t_start
    print(
        f">> finished in {elapsed/60:.2f} min  "
        f"done={done_count} skip={skip_count} fail={fail_count}  "
        f"of {len(jobs)} jobs",
        flush=True,
    )
    return 0 if fail_count == 0 else 2


def main(argv=None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
