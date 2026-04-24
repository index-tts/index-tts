# Modified Guidance

---

## Manual file reconstruction

If the patch won't apply cleanly, create each file manually. Everything below is copy-paste ready.

### File 1: `indextts/batch/__init__.py` (new)

```python
"""Multi-GPU offline batch inference for IndexTTS2.

Entry point: ``python -m indextts.batch --help``.
"""
```

### File 2: `indextts/batch/__main__.py` (new)

```python
from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
```

### File 3: `indextts/batch/manifest.py` (new)

```python
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

@dataclasses.dataclass
class Job:
    id: str
    text: str
    ref_audio: str
    out_path: Optional[str] = None  # resolved by the runner if None

    def resolved_out_path(self, output_dir: str) -> str:
        if self.out_path:
            return self.out_path
        return os.path.join(output_dir, f"{self.id}.wav")

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
```

### File 4: `indextts/batch/worker.py` (new)

```python
"""Per-GPU worker process for batch inference.

The worker pins itself to one GPU (via ``CUDA_VISIBLE_DEVICES`` set by the
runner *before* the process imports torch), constructs a single ``IndexTTS2``
replica, and then pulls ``Job`` records off a shared queue until it receives a
sentinel.

Each processed job produces exactly one progress event pushed onto
``progress_q``; the runner serialises those into the progress JSONL.
"""

from __future__ import annotations

import dataclasses
import os
import queue as queue_mod
import time
import traceback
import warnings
from typing import Any

# Marker message used to detect the mel-token truncation warning emitted by
# ``IndexTTS2.infer_generator`` (see indextts/infer_v2.py). Matched as a
# substring because the full message includes dynamic lengths.
_TRUNCATION_MARKER = "generation stopped due to exceeding"

# Sentinel pushed onto the job queue to ask a worker to exit.
SHUTDOWN = None

@dataclasses.dataclass
class WorkerConfig:
    gpu_id: int
    slot: int                 # which replica slot on this GPU (0-indexed)
    model_dir: str
    cfg_path: str
    output_dir: str
    use_fp16: bool
    use_cuda_kernel: bool
    use_torch_compile: bool
    max_text_tokens_per_segment: int
    max_mel_tokens: int
    retries: int
    overwrite: bool

def _progress(progress_q, **fields) -> None:
    try:
        progress_q.put(fields, timeout=60)
    except Exception:
        pass

def _process_one(tts, job, cfg: WorkerConfig) -> tuple[str, dict]:
    """Run a single inference. Returns (status, extra_fields)."""
    out_path = job.resolved_out_path(cfg.output_dir)

    if not cfg.overwrite and os.path.exists(out_path):
        return "skipped", {"reason": "already_exists", "out_path": out_path}

    if not job.text or not job.text.strip():
        return "skipped", {"reason": "empty_text"}

    if not os.path.isfile(job.ref_audio):
        return "skipped", {"reason": "ref_audio_missing"}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        tts.infer(
            spk_audio_prompt=job.ref_audio,
            text=job.text,
            output_path=tmp_path,
            max_text_tokens_per_segment=cfg.max_text_tokens_per_segment,
            max_mel_tokens=cfg.max_mel_tokens,
            verbose=False,
        )
    elapsed = time.perf_counter() - t0

    truncated = any(
        _TRUNCATION_MARKER in str(w.message) for w in caught
    )
    if truncated:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return "skipped", {"reason": "truncated_output", "seconds": round(elapsed, 3)}

    if not os.path.exists(tmp_path):
        # infer() returned without writing (e.g. it hit an internal IndexError
        # branch). Treat as skip so the job isn't retried forever.
        return "skipped", {"reason": "no_output_file", "seconds": round(elapsed, 3)}

    os.replace(tmp_path, out_path)
    return "done", {"out_path": out_path, "seconds": round(elapsed, 3)}

def _run_with_retries(tts, job, cfg: WorkerConfig) -> dict:
    """Wrap a single job in retry/OOM handling. Returns a progress event dict."""
    import torch  # local import: worker always has torch by now

    last_error: str = ""
    for attempt in range(cfg.retries + 1):
        try:
            status, extra = _process_one(tts, job, cfg)
            return {"id": job.id, "status": status, **extra}
        except torch.cuda.OutOfMemoryError as e:
            last_error = f"cuda_oom: {e}"
            torch.cuda.empty_cache()
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                last_error = f"cuda_oom: {msg}"
                torch.cuda.empty_cache()
            else:
                last_error = f"runtime_error: {msg}"
        except Exception as e:  # noqa: BLE001 - we want to catch anything
            last_error = f"{type(e).__name__}: {e}"
            # Permanent-looking errors shouldn't be retried.
            if isinstance(e, (ValueError, FileNotFoundError, KeyError)):
                return {"id": job.id, "status": "skipped", "reason": last_error}
            traceback.print_exc()
        # wait briefly before retry; avoids thrashing on transient issues
        time.sleep(1.0 * (attempt + 1))
    return {"id": job.id, "status": "failed", "error": last_error}

def worker_main(cfg: WorkerConfig, job_q, progress_q) -> None:
    """Entry point for each worker process.

    IMPORTANT: the runner sets ``CUDA_VISIBLE_DEVICES`` in the child's
    environment before spawn, so at this point the process sees exactly one
    visible GPU as device index 0.
    """
    # Keep this block minimal so import errors get reported via progress_q.
    try:
        import torch  # noqa: F401
        from indextts.infer_v2 import IndexTTS2
    except Exception as e:  # noqa: BLE001
        _progress(
            progress_q,
            id="__worker_init__",
            status="failed",
            error=f"import_error on gpu {cfg.gpu_id} slot {cfg.slot}: {type(e).__name__}: {e}",
        )
        return

    try:
        tts = IndexTTS2(
            cfg_path=cfg.cfg_path,
            model_dir=cfg.model_dir,
            use_fp16=cfg.use_fp16,
            device="cuda:0",
            use_cuda_kernel=cfg.use_cuda_kernel,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=cfg.use_torch_compile,
            skip_qwen_emo=True,
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _progress(
            progress_q,
            id="__worker_init__",
            status="failed",
            error=f"model_load_error on gpu {cfg.gpu_id} slot {cfg.slot}: {type(e).__name__}: {e}",
        )
        return

    # Signal readiness so the runner can track startup progress.
    _progress(progress_q, id="__worker_ready__", status="info",
              gpu=cfg.gpu_id, slot=cfg.slot)

    while True:
        try:
            job = job_q.get(timeout=5)
        except queue_mod.Empty:
            continue
        if job is SHUTDOWN:
            break
        event = _run_with_retries(tts, job, cfg)
        event["gpu"] = cfg.gpu_id
        event["slot"] = cfg.slot
        _progress(progress_q, **event)

    _progress(progress_q, id="__worker_exit__", status="info",
              gpu=cfg.gpu_id, slot=cfg.slot)
```

### File 5: `indextts/batch/runner.py` (new)

```python
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
```

### File 6: `indextts/infer_v2.py` (edit, 3 hunks)

This is the existing ~800-line file — you only need to apply **three small changes**:

**Hunk 1** — line 41 (end of the `__init__` signature): add `skip_qwen_emo` parameter.

```python
# BEFORE
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, device=None,
            use_cuda_kernel=None,use_deepspeed=False, use_accel=False, use_torch_compile=False
    ):

# AFTER
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, device=None,
            use_cuda_kernel=None,use_deepspeed=False, use_accel=False, use_torch_compile=False,
            skip_qwen_emo=False
    ):
```

**Hunk 2** — in the docstring around line 52, add the new arg's description (optional but recommended):

```python
            use_torch_compile (bool): whether to use torch.compile for optimization or not.
            skip_qwen_emo (bool): if True, skip loading the QwenEmotion text-to-emotion model.
                Saves VRAM and startup time for pure voice-cloning workloads that never call
                ``infer(..., use_emo_text=True)``. Attempting to use emotion-text guidance when
                this is enabled will raise a RuntimeError.
```

**Hunk 3** — line 83 (the `self.qwen_emo = QwenEmotion(...)` call): wrap it in a conditional.

```python
# BEFORE
        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))

# AFTER
        if skip_qwen_emo:
            self.qwen_emo = None
            print(">> QwenEmotion skipped (skip_qwen_emo=True)")
        else:
            self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))
```

**Hunk 4** — inside `infer_generator` around line 401 (the `if use_emo_text:` branch): add a guard for the skip case.

```python
# BEFORE
        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)

# AFTER
        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if self.qwen_emo is None:
                raise RuntimeError(
                    "use_emo_text=True requires QwenEmotion, but it was skipped at init "
                    "(skip_qwen_emo=True). Re-construct IndexTTS2 with skip_qwen_emo=False."
                )
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)
```

---

After placing all five new files and applying the four edits to `infer_v2.py`, from the repo root:

```bash
# verify structure
python -m py_compile indextts/batch/*.py indextts/infer_v2.py
python -m indextts.batch --help

# commit & push
git checkout -b claude/batch-inference-multi-gpu-lgUTm
git add indextts/batch/ indextts/infer_v2.py
git commit -m "Add multi-GPU offline batch inference for IndexTTS2"
git push -u origin claude/batch-inference-multi-gpu-lgUTm
```

Let me know if the patch applies cleanly or if you hit any issues.