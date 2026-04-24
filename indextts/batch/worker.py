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
    base, ext = os.path.splitext(out_path)
    tmp_path = f"{base}.tmp{ext or '.wav'}"
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
