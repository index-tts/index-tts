from __future__ import annotations

import argparse
import atexit
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import threading
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr
import pandas as pd
from omegaconf import OmegaConf

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from tools.i18n.i18n import I18nAuto

parser = argparse.ArgumentParser(description="IndexTTS Parallel WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
parser.add_argument("--port", type=int, default=7862, help="Port for the web UI")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the web UI")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
parser.add_argument("--is_fp16", action="store_true", default=False, help="Enable fp16 inference")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

required_files = [
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
]
for file_name in required_files:
    file_path = os.path.join(cmd_args.model_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

try:
    BASE_CFG = OmegaConf.load(os.path.join(cmd_args.model_dir, "config.yaml"))
except Exception as exc:  # pragma: no cover - config must load
    print(f"Failed to load config.yaml: {exc}")
    sys.exit(1)

# Configure cache locations and disable DeepSpeed before importing heavy modules
hf_cache_dir = os.path.join(cmd_args.model_dir, "hf_cache")
torch_cache_dir = os.path.join(cmd_args.model_dir, "torch_cache")
os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")
os.environ.setdefault("HF_HOME", hf_cache_dir)
os.environ.setdefault("HF_HUB_CACHE", hf_cache_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
os.environ.setdefault("TORCH_HOME", torch_cache_dir)
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(torch_cache_dir, exist_ok=True)

from indextts.infer_v2_modded import IndexTTS2

i18n = I18nAuto(language="Auto")
logger = logging.getLogger("webui_parallel")

os.makedirs(os.path.join(current_dir, "outputs", "tasks"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "prompts"), exist_ok=True)

# Avoid DeepSpeed initialization on platforms where it stalls by default
os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")

example_cases: List[List[Any]] = []
examples_path = Path(current_dir) / "examples" / "cases.jsonl"
if examples_path.exists():
    with examples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            emo_audio = example.get("emo_audio")
            emo_audio_path = os.path.join("examples", emo_audio) if emo_audio else None
            example_cases.append([
                os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                example.get("emo_mode", 0),
                example.get("text"),
                emo_audio_path,
                example.get("emo_weight", 1.0),
                example.get("emo_text", ""),
                example.get("emo_vec_1", 0),
                example.get("emo_vec_2", 0),
                example.get("emo_vec_3", 0),
                example.get("emo_vec_4", 0),
                example.get("emo_vec_5", 0),
                example.get("emo_vec_6", 0),
                example.get("emo_vec_7", 0),
                example.get("emo_vec_8", 0),
            ])

EMO_CHOICES = [
    "Match prompt audio",
    "Use emotion reference audio",
    "Use emotion vector",
    "Use emotion text description",
]

parallel_worker_config = {
    "model_dir": cmd_args.model_dir,
    "is_fp16": cmd_args.is_fp16,
    "verbose": cmd_args.verbose,
    "hf_cache": hf_cache_dir,
    "torch_cache": torch_cache_dir,
    "gpt_path": None,
    "bpe_path": None,
}


class WorkerPool:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ctx = mp.get_context("spawn")
        self.job_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.processes: List[mp.Process] = []
        self.worker_count = 0
        self.lock = threading.Lock()
        self.batch_counter = itertools.count()

    def _all_alive(self) -> bool:
        return all(p.is_alive() for p in self.processes)

    def ensure(self, count: int):
        count = max(1, int(count))
        with self.lock:
            if self.worker_count == count and self.processes and self._all_alive():
                return
            self.stop_locked()
            self.start_locked(count)

    def start_locked(self, count: int):
        self.job_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.processes = []
        self.worker_count = count
        for _ in range(count):
            p = self.ctx.Process(
                target=_worker_loop,
                args=(self.job_queue, self.result_queue, self.config),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

    def stop_locked(self):
        if not self.processes:
            return
        if self.job_queue is not None:
            for _ in self.processes:
                self.job_queue.put({"type": "stop"})
        for p in self.processes:
            p.join(timeout=5)
        self.processes = []
        if self.job_queue is not None:
            self.job_queue.close()
            self.job_queue = None
        if self.result_queue is not None:
            self.result_queue.close()
            self.result_queue = None
        self.worker_count = 0

    def stop(self):
        with self.lock:
            self.stop_locked()

    def run_jobs(self, jobs: List[GenerationJob], progress: Optional[gr.Progress]):
        if not jobs:
            return {}
        with self.lock:
            if not self.processes or self.job_queue is None or self.result_queue is None:
                raise RuntimeError("Worker pool not initialized")
            batch_id = next(self.batch_counter)
            total = len(jobs)
            for job in jobs:
                payload = job.__dict__.copy()
                payload["batch_id"] = batch_id
                self.job_queue.put(payload)

        row_results: Dict[int, Dict[str, Any]] = {}
        processed = 0
        total = len(jobs)
        while processed < total:
            message = self.result_queue.get()  # type: ignore[arg-type]
            if message.get("type") == "init_error":
                raise RuntimeError(f"Worker failed to start: {message['error']}")
            if message.get("batch_id") != batch_id:
                continue
            row_results[message["row_id"]] = message
            processed += 1
            _update_progress(progress, min(processed / total, 0.999), desc=f"Processed {processed}/{total}")

        _update_progress(progress, 1.0, desc="Parallel generation complete")
        return row_results


worker_pool = WorkerPool(parallel_worker_config)


def _shutdown_worker_pool():
    worker_pool.stop()


atexit.register(_shutdown_worker_pool)

_PRIMARY_TTS: Optional[IndexTTS2] = None
_MODEL_SELECTION: Dict[str, Optional[str]] = {"gpt": None, "bpe": None}


def _candidate_paths(base_dirs: List[Path], suffixes: List[str]) -> List[str]:
    results: List[str] = []
    seen: set[str] = set()
    for base in base_dirs:
        if not base or not base.exists():
            continue
        for suffix in suffixes:
            for path in base.glob(f"*{suffix}"):
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    results.append(resolved)
    results.sort()
    return results


def _is_gpt_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".pth"):
        return False
    excluded = ("s2mel", "campplus", "bigvgan", "wav2vec", "emo", "spk", "cfm")
    return not any(token in name for token in excluded)


def _discover_gpt_checkpoints() -> List[str]:
    bases = [
        Path(cmd_args.model_dir),
        Path(current_dir) / "models",
    ]
    candidates = _candidate_paths(bases, [".pth"])
    return [path for path in candidates if _is_gpt_checkpoint(Path(path))]


def _discover_bpe_models() -> List[str]:
    bases = [
        Path(cmd_args.model_dir),
        Path(current_dir) / "tokenizers",
    ]
    return _candidate_paths(bases, [".model"])


def dispose_primary_tts():
    global _PRIMARY_TTS
    if _PRIMARY_TTS is not None:
        try:
            if hasattr(_PRIMARY_TTS, "gr_progress"):
                _PRIMARY_TTS.gr_progress = None
        finally:
            _PRIMARY_TTS = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def build_primary_tts() -> IndexTTS2:
    if _MODEL_SELECTION["gpt"] is None or _MODEL_SELECTION["bpe"] is None:
        raise RuntimeError("Model selection is not set. Provide GPT and BPE paths before loading.")
    return IndexTTS2(
        model_dir=cmd_args.model_dir,
        cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
        is_fp16=cmd_args.is_fp16,
        use_cuda_kernel=False,
        use_accel=True,
        gpt_checkpoint_path=_MODEL_SELECTION["gpt"],
        bpe_model_path=_MODEL_SELECTION["bpe"],
    )


def load_primary_tts(gpt_path: str, bpe_path: str) -> IndexTTS2:
    dispose_primary_tts()
    resolved_gpt = os.path.abspath(gpt_path)
    resolved_bpe = os.path.abspath(bpe_path)
    previous_selection = _MODEL_SELECTION.copy()
    _MODEL_SELECTION["gpt"] = resolved_gpt
    _MODEL_SELECTION["bpe"] = resolved_bpe
    try:
        tts = build_primary_tts()
    except Exception:
        _MODEL_SELECTION.update(previous_selection)
        dispose_primary_tts()
        raise
    global _PRIMARY_TTS
    _PRIMARY_TTS = tts
    parallel_worker_config["gpt_path"] = resolved_gpt
    parallel_worker_config["bpe_path"] = resolved_bpe
    worker_pool.stop()
    return tts


def ensure_primary_tts() -> IndexTTS2:
    if _PRIMARY_TTS is None:
        raise RuntimeError("No GPT checkpoint loaded. Use the Load button in the UI.")
    return _PRIMARY_TTS


def _model_status_text() -> str:
    if _PRIMARY_TTS is None:
        return "⚠️ No model loaded. Select a GPT checkpoint and BPE tokenizer, then click Load."
    gpt_path = _MODEL_SELECTION.get("gpt")
    bpe_path = _MODEL_SELECTION.get("bpe")
    gpt_name = Path(gpt_path).name if gpt_path else "?"
    bpe_name = Path(bpe_path).name if bpe_path else "?"
    return f"✅ Loaded GPT: **{gpt_name}** | BPE: **{bpe_name}**"


def _format_label(path: str) -> str:
    path_obj = Path(path)
    candidates: List[str] = []

    try:
        rel_model = os.path.relpath(path, cmd_args.model_dir)
        if not rel_model.startswith(".."):
            prefix = Path(cmd_args.model_dir).name or "checkpoints"
            candidates.append(f"{prefix}/{rel_model}".replace("\\", "/"))
    except ValueError:
        pass

    try:
        rel_repo = os.path.relpath(path, current_dir)
        if not rel_repo.startswith(".."):
            candidates.append(rel_repo.replace("\\", "/"))
    except ValueError:
        pass

    candidates.append(path_obj.name)
    for label in candidates:
        if label:
            return label
    return str(path_obj)


def _format_dropdown_choices(
    paths: List[str],
    current_selection: Optional[str],
) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    labels: List[str] = []
    mapping: Dict[str, str] = {}
    selected_label: Optional[str] = None
    for path in paths:
        label = _format_label(path)
        base_label = label
        suffix = 1
        while label in mapping:
            label = f"{base_label} ({suffix})"
            suffix += 1
        mapping[label] = path
        labels.append(label)
        if current_selection and os.path.abspath(path) == os.path.abspath(current_selection):
            selected_label = label
    if labels and selected_label is None:
        selected_label = labels[0]
    return labels, mapping, selected_label


@dataclass
class GenerationJob:
    row_id: int
    prompt_path: str
    text: str
    output_path: str
    emo_mode: int
    emo_weight: float
    emo_vector: Optional[List[float]]
    emo_text: str
    emo_random: bool
    emo_ref_path: Optional[str]
    max_tokens: int
    generation_kwargs: Dict[str, Any]
    verbose: bool
    duration_seconds: Optional[float] = None


def _normalize_seed(seed_value: Any) -> Optional[int]:
    if seed_value is None:
        return None
    if isinstance(seed_value, str):
        value = seed_value.strip()
        if not value:
            return None
        try:
            seed = int(value)
        except ValueError:
            try:
                seed = int(float(value))
            except ValueError:
                return None
    elif isinstance(seed_value, bool):
        seed = int(seed_value)
    elif isinstance(seed_value, float):
        if math.isnan(seed_value):
            return None
        seed = int(seed_value)
    else:
        try:
            seed = int(seed_value)
        except (TypeError, ValueError):
            return None
    if seed < 0:
        seed = abs(seed)
    return seed


def _normalize_duration_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return seconds


def _apply_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    py_seed = int(seed % (2**32))
    random.seed(py_seed)
    np.random.seed(py_seed)
    torch_seed = int(seed % (2**63 - 1))
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def _prepare_generation_kwargs(raw_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(raw_kwargs or {})
    seed = _normalize_seed(kwargs.pop("seed", None))
    _apply_seed(seed)
    return kwargs


def _worker_loop(job_queue: mp.Queue, result_queue: mp.Queue, config: Dict[str, Any]):
    hf_cache = config.get("hf_cache")
    torch_cache = config.get("torch_cache")
    if hf_cache:
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("HF_HUB_CACHE", hf_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)
        os.makedirs(hf_cache, exist_ok=True)
    if torch_cache:
        os.environ.setdefault("TORCH_HOME", torch_cache)
        os.makedirs(torch_cache, exist_ok=True)
    os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")
    gpt_override = config.get("gpt_path")
    bpe_override = config.get("bpe_path")
    if not gpt_override or not bpe_override:
        result_queue.put({"type": "init_error", "error": "No GPT/BPE model loaded. Use the Load button."})
        return
    try:
        worker_tts = IndexTTS2(
            model_dir=config["model_dir"],
            cfg_path=os.path.join(config["model_dir"], "config.yaml"),
            is_fp16=config.get("is_fp16", False),
            use_cuda_kernel=False,
            use_accel=True,
            gpt_checkpoint_path=gpt_override,
            bpe_model_path=bpe_override,
        )
    except Exception as exc:  # pragma: no cover - worker init path
        logger.exception("Worker failed to initialize")
        result_queue.put({"type": "init_error", "error": str(exc)})
        return

    while True:
        job = job_queue.get()
        if isinstance(job, dict) and job.get("type") == "stop":
            break

        try:
            emo_mode = job["emo_mode"]
            emo_audio_prompt = job["emo_ref_path"] if emo_mode == 1 else None
            emo_alpha = job["emo_weight"] if emo_mode == 1 else 1.0
            emo_vector = job["emo_vector"] if emo_mode == 2 else None
            use_emo_text = emo_mode == 3
            generation_kwargs = _prepare_generation_kwargs(job.get("generation_kwargs", {}))

            worker_tts.infer(
                spk_audio_prompt=job["prompt_path"],
                text=job["text"],
                output_path=job["output_path"],
                emo_audio_prompt=emo_audio_prompt,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                emo_text=job["emo_text"],
                use_random=job["emo_random"],
                verbose=job.get("verbose", False),
                max_text_tokens_per_sentence=job["max_tokens"],
                duration_seconds=job.get("duration_seconds"),
                **generation_kwargs,
            )
            result_queue.put(
                {
                    "type": "result",
                    "row_id": job["row_id"],
                    "status": "Completed",
                    "output_path": job["output_path"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": None,
                    "batch_id": job.get("batch_id"),
                }
            )
        except Exception as exc:  # pragma: no cover - worker runtime path
            logger.exception("Worker generation error")
            result_queue.put(
                {
                    "type": "result",
                    "row_id": job["row_id"],
                    "status": f"Error: {exc}",
                    "output_path": None,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(exc),
                    "batch_id": job.get("batch_id"),
                }
            )

    try:
        worker_tts.unload()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - optional cleanup
        pass


def _update_progress(progress: Optional[gr.Progress], value: float, desc: str = "") -> None:
    if progress is None:
        return
    try:
        progress(value, desc=desc)
    except Exception:
        pass

MAX_LENGTH_TO_USE_SPEED = 70


def create_demo() -> gr.Blocks:
    gpt_choices = _discover_gpt_checkpoints()
    bpe_choices = _discover_bpe_models()
    gpt_labels, gpt_map, initial_gpt_label = _format_dropdown_choices(gpt_choices, _MODEL_SELECTION["gpt"])
    bpe_labels, bpe_map, initial_bpe_label = _format_dropdown_choices(bpe_choices, _MODEL_SELECTION["bpe"])

    gpt_cfg = getattr(BASE_CFG, "gpt", {})
    max_mel_tokens_limit = int(getattr(gpt_cfg, "max_mel_tokens", 2048))
    if max_mel_tokens_limit < 100:
        max_mel_tokens_limit = 100
    default_mel_value = min(1500, max_mel_tokens_limit)
    max_text_tokens_limit = int(getattr(gpt_cfg, "max_text_tokens", 256))
    if max_text_tokens_limit < 40:
        max_text_tokens_limit = 40
    default_text_tokens = min(120, max_text_tokens_limit)
    cfg_version = getattr(BASE_CFG, "version", "1.0")

    with gr.Blocks(title="IndexTTS Parallel Demo") as demo:
        model_status = gr.Markdown(value=_model_status_text())
        gpt_map_state = gr.State(gpt_map)
        bpe_map_state = gr.State(bpe_map)
        with gr.Row():
            gpt_dropdown = gr.Dropdown(
                choices=gpt_labels,
                value=initial_gpt_label,
                label="GPT Checkpoint (.pth)",
                interactive=True,
            )
            bpe_dropdown = gr.Dropdown(
                choices=bpe_labels,
                value=initial_bpe_label,
                label="BPE Tokenizer (.model)",
                interactive=True,
            )
            refresh_models_button = gr.Button("Refresh Models", variant="secondary")
            load_models_button = gr.Button("Load Models", variant="primary")

        def refresh_model_lists():
            gpt_files = _discover_gpt_checkpoints()
            bpe_files = _discover_bpe_models()
            gpt_labels_new, gpt_map_new, gpt_value = _format_dropdown_choices(gpt_files, _MODEL_SELECTION["gpt"])
            bpe_labels_new, bpe_map_new, bpe_value = _format_dropdown_choices(bpe_files, _MODEL_SELECTION["bpe"])
            return (
                gr.update(choices=gpt_labels_new, value=gpt_value),
                gr.update(choices=bpe_labels_new, value=bpe_value),
                gpt_map_new,
                bpe_map_new,
                _model_status_text(),
            )

        def handle_model_load(
            gpt_label: Optional[str],
            bpe_label: Optional[str],
            gpt_map_value: Optional[Dict[str, str]],
            bpe_map_value: Optional[Dict[str, str]],
            progress: gr.Progress = gr.Progress(track_tqdm=False),
        ) -> str:
            gpt_map_local = gpt_map_value or {}
            bpe_map_local = bpe_map_value or {}
            gpt_path = gpt_map_local.get(gpt_label or "", gpt_label)
            bpe_path = bpe_map_local.get(bpe_label or "", bpe_label)
            if not gpt_path or not bpe_path:
                gr.Warning("Select both a GPT checkpoint and a BPE tokenizer before loading.")
                return _model_status_text()
            progress(0.1, "Loading models...")
            try:
                load_primary_tts(gpt_path, bpe_path)
            except Exception as exc:
                logger.exception("Failed to load models")
                gr.Warning(f"Failed to load models: {exc}")
                return f"❌ Failed to load models: {exc}"
            gr.Info("Models loaded successfully.")
            return _model_status_text()

        refresh_models_button.click(
            refresh_model_lists,
            inputs=[],
            outputs=[gpt_dropdown, bpe_dropdown, gpt_map_state, bpe_map_state, model_status],
        )
        load_models_button.click(
            handle_model_load,
            inputs=[gpt_dropdown, bpe_dropdown, gpt_map_state, bpe_map_state],
            outputs=model_status,
        )
        batch_rows_state = gr.State([])
        next_batch_id_state = gr.State(1)

        gr.HTML(
            """
            <h2 style=\"text-align:center;\">IndexTTS2 Parallel Batch Demo</h2>
            """
        )

        with gr.Accordion("Emotion Settings", open=True):
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES,
                    type="index",
                    value=0,
                    label="Emotion Control Mode",
                )

        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label="Emotion Reference Audio", type="filepath")
            with gr.Row():
                emo_weight = gr.Slider(label="Emotion Weight", minimum=0.0, maximum=1.6, value=0.8, step=0.01)

        with gr.Row():
            emo_random = gr.Checkbox(label="Random Emotion Sampling", value=False, visible=False)

        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label="Joy", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec6 = gr.Slider(label="Low Mood", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.4, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            emo_text = gr.Textbox(label="Emotion Description", placeholder="Describe the target emotion", value="")

        with gr.Accordion("Advanced Generation Settings", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**GPT2 Sampling Settings**")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info="Enable sampling")
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(
                        label="max_mel_tokens",
                        value=default_mel_value,
                        minimum=50,
                        maximum=max_mel_tokens_limit,
                        step=10,
                        info="Maximum generated mel tokens",
                    )
                    seed_value = gr.Number(
                        label="Seed",
                        value=None,
                        precision=0,
                        minimum=0,
                        step=1,
                        info="Leave blank for random sampling; set a value for reproducible outputs.",
                    )
                with gr.Column(scale=2):
                    gr.Markdown("**Sentence Settings**")
                    max_text_tokens_per_sentence = gr.Slider(
                        label="Max tokens per sentence",
                        value=default_text_tokens,
                        minimum=20,
                        maximum=max_text_tokens_limit,
                        step=2,
                        key="max_text_tokens_per_sentence",
                    )
                    duration_seconds_input = gr.Number(
                        label="Target duration (seconds)",
                        value=None,
                        precision=2,
                        minimum=0,
                        step=0.1,
                        info="Optional: approximate overall audio length. Leave blank for free duration.",
                    )
                    with gr.Accordion("Preview sentences", open=True):
                        sentences_preview = gr.Dataframe(
                            headers=["Index", "Sentence", "Token Count"],
                            key="sentences_preview",
                            wrap=True,
                        )
        advanced_params = [
            do_sample,
            top_p,
            top_k,
            temperature,
            length_penalty,
            num_beams,
            repetition_penalty,
            max_mel_tokens,
            seed_value,
        ]

        with gr.Tab("Single Generation"):
            with gr.Row():
                prompt_audio = gr.Audio(label="Prompt Audio", key="prompt_audio", sources=["upload", "microphone"], type="filepath")
                with gr.Column():
                    input_text_single = gr.TextArea(
                        label="Text",
                        key="input_text_single",
                        placeholder="Enter text to synthesize",
                        info=f"Model version {cfg_version}",
                    )
                    gen_button = gr.Button("Generate", key="gen_button", interactive=True)
            output_audio = gr.Audio(label="Generated Result", visible=True, key="output_audio")

            if example_cases:
                gr.Examples(
                    examples=example_cases,
                    inputs=[
                        prompt_audio,
                        emo_control_method,
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1,
                        vec2,
                        vec3,
                        vec4,
                        vec5,
                        vec6,
                        vec7,
                        vec8,
                    ],
                )

        with gr.Tab("Batch Generation"):
            gr.Markdown("Manage multiple prompt audios, give each its own text, generate in bulk, and retry specific entries as needed.")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        dataset_path_input = gr.Textbox(
                            label="Dataset train.txt path",
                            value="vivy_va_dataset/train.txt",
                            scale=3,
                            placeholder="Path to train.txt",
                        )
                        load_dataset_button = gr.Button("Load Dataset", scale=1)
                    batch_file_input = gr.Files(
                        label="Add prompt audio files",
                        file_types=["audio"],
                        file_count="multiple",
                        type="filepath",
                    )
                    worker_count = gr.Slider(
                        label="Parallel workers",
                        minimum=1,
                        maximum=8,
                        value=2,
                        step=1,
                        info="Number of parallel TTS workers",
                    )
                    batch_table = gr.Dataframe(
                        headers=["ID", "Prompt", "Text", "Output", "Status", "Last Generated"],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        row_count=(0, "dynamic"),
                        col_count=6,
                        interactive=False,
                        value=[],
                    )
                with gr.Column():
                    selected_entry = gr.Dropdown(label="Select entry", choices=[], value=None, interactive=True)
                    batch_prompt_player = gr.Audio(label="Prompt Audio", type="filepath", interactive=False)
                    batch_output_player = gr.Audio(label="Generated Audio", type="filepath", interactive=False)
                    batch_text_input = gr.TextArea(label="Text", placeholder="Enter text for this entry", interactive=True)
                    apply_text_button = gr.Button("Save Text", variant="secondary")
                    batch_status = gr.Markdown(value="No entry selected.")
                    with gr.Row():
                        generate_all_button = gr.Button("Generate All")
                        regenerate_button = gr.Button("Regenerate Selected")
                    with gr.Row():
                        delete_entry_button = gr.Button("Delete Selected")
                        clear_entries_button = gr.Button("Clear All")

        def gen_single(
            emo_control_method_value,
            prompt,
            text,
            emo_ref_path,
            emo_weight_value,
            vec1_value,
            vec2_value,
            vec3_value,
            vec4_value,
            vec5_value,
            vec6_value,
            vec7_value,
            vec8_value,
            emo_text_value,
            emo_random_value,
            max_text_tokens_per_sentence_value,
            duration_seconds_value,
            *args,
            progress: gr.Progress = gr.Progress(),
        ):
            if not prompt:
                gr.Warning("Upload a prompt audio file first.")
                return gr.update()

            output_path = os.path.join(current_dir, "outputs", f"spk_{int(time.time())}.wav")
            try:
                tts = ensure_primary_tts()
            except RuntimeError as exc:
                gr.Warning(str(exc))
                return gr.update()

            tts.gr_progress = progress

            advanced_values = list(args)
            expected_len = len(advanced_params)
            if len(advanced_values) < expected_len:
                advanced_values.extend([None] * (expected_len - len(advanced_values)))
            raw_generation_kwargs = build_generation_kwargs(*advanced_values[:expected_len])
            generation_kwargs = _prepare_generation_kwargs(raw_generation_kwargs)

            emo_mode = emo_control_method_value if isinstance(emo_control_method_value, int) else getattr(emo_control_method_value, "value", 0)
            if emo_mode == 2:
                vec_sum = vec1_value + vec2_value + vec3_value + vec4_value + vec5_value + vec6_value + vec7_value + vec8_value
                if vec_sum > 1.5:
                    gr.Warning("Emotion vector sum cannot exceed 1.5. Adjust the sliders and retry.")
                    return gr.update()
                emo_vector = [vec1_value, vec2_value, vec3_value, vec4_value, vec5_value, vec6_value, vec7_value, vec8_value]
            else:
                emo_vector = None

            duration_seconds = _normalize_duration_seconds(duration_seconds_value)

            tts.infer(
                spk_audio_prompt=prompt,
                text=text,
                output_path=output_path,
                emo_audio_prompt=emo_ref_path if emo_mode == 1 else None,
                emo_alpha=float(emo_weight_value) if emo_mode == 1 else 1.0,
                emo_vector=emo_vector if emo_mode == 2 else None,
                use_emo_text=(emo_mode == 3),
                emo_text=emo_text_value,
                use_random=emo_random_value,
                verbose=cmd_args.verbose,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence_value),
                duration_seconds=duration_seconds,
                **generation_kwargs,
            )

            return gr.update(value=output_path, visible=True)

        def on_input_text_change(text_value, max_tokens_value):
            if not text_value:
                return {sentences_preview: gr.update(value=[], visible=True, type="array")}

            try:
                tts = ensure_primary_tts()
            except RuntimeError as exc:
                gr.Warning(str(exc))
                return {sentences_preview: gr.update(value=[], visible=True, type="array")}

            tokenized = tts.tokenizer.tokenize(text_value)
            sentences = tts.tokenizer.split_segments(
                tokenized, max_text_tokens_per_segment=int(max_tokens_value)
            )
            data = []
            for idx, sentence_tokens in enumerate(sentences):
                sentence_str = "".join(sentence_tokens)
                data.append([idx, sentence_str, len(sentence_tokens)])
            return {sentences_preview: gr.update(value=data, visible=True, type="array")}

        def on_method_select(emo_control_value):
            if emo_control_value == 1:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            if emo_control_value == 2:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                )
            if emo_control_value == 3:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def build_batch_table_data(rows: List[Dict[str, Any]]):
            table_data = []
            for row in rows:
                text_preview = (row.get("text") or "")[:57]
                if row.get("text") and len(row["text"]) > 60:
                    text_preview += "..."
                table_data.append(
                    [
                        row.get("id"),
                        os.path.basename(row.get("prompt_path", "")) if row.get("prompt_path") else "",
                        text_preview,
                        os.path.basename(row.get("output_path", "")) if row.get("output_path") else "",
                        row.get("status", "Pending"),
                        row.get("last_generated", ""),
                    ]
                )
            return table_data

        def find_batch_row(rows, row_id):
            for row in rows or []:
                if row.get("id") == row_id:
                    return row
            return None

        def resolve_batch_selection(rows, selected_value):
            choices = [str(row.get("id")) for row in rows or []]
            if not choices:
                return gr.update(choices=[], value=None), None
            if selected_value is not None:
                selected_str = str(selected_value)
                if selected_str in choices:
                    return gr.update(choices=choices, value=selected_str), int(selected_str)
            return gr.update(choices=choices, value=choices[-1]), int(choices[-1])

        def prepare_batch_selection(rows, selected_value):
            dropdown_update, resolved_id = resolve_batch_selection(rows, selected_value)
            row = find_batch_row(rows, resolved_id)
            prompt_update = gr.update(value=row.get("prompt_path") if row else None)
            output_update = gr.update(value=row.get("output_path") if row else None)
            text_update = gr.update(value=row.get("text", "") if row else "")
            return dropdown_update, resolved_id, prompt_update, output_update, text_update, row

        def format_batch_status(row, message=None):
            if not row:
                base = "No entry selected."
            else:
                details = [f"Row {row.get('id')}: {row.get('status', 'Pending')}"]
                if row.get("text"):
                    preview = row["text"][:117] + ("..." if len(row["text"]) > 120 else "")
                    details.append(f"Text: {preview}")
                if row.get("output_path"):
                    details.append(f"Output: {row['output_path']}")
                if row.get("last_generated"):
                    details.append(f"Last generated: {row['last_generated']}")
                base = "\n".join(details)
            if message:
                base = f"{base}\n{message}" if base else message
            return gr.update(value=base)

        def add_batch_prompts(files, rows, next_id, selected_value):
            rows = rows or []
            next_id = next_id or 1
            files = files or []
            updated_rows = [dict(row) for row in rows]
            prompts_dir = os.path.join(current_dir, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)

            added = 0
            last_added_id = None
            for file_path in files:
                if not file_path:
                    continue
                safe_name = os.path.basename(file_path)
                timestamp = int(time.time() * 1000)
                target_name = f"batch_prompt_{next_id}_{timestamp}_{safe_name}"
                target_path = os.path.join(prompts_dir, target_name)
                try:
                    shutil.copy(file_path, target_path)
                except Exception as exc:
                    logger.exception("Failed to store prompt %s", file_path)
                    gr.Warning(f"Failed to add {safe_name}: {exc}")
                    continue
                entry = {
                    "id": next_id,
                    "prompt_path": target_path,
                    "output_path": None,
                    "status": "Pending",
                    "last_generated": "",
                    "text": "",
                }
                updated_rows.append(entry)
                added += 1
                last_added_id = entry["id"]
                next_id += 1

            selected_seed = last_added_id if added else selected_value
            dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(
                updated_rows, selected_seed
            )
            table_update = gr.update(value=build_batch_table_data(updated_rows))
            status_message = f"Added {added} prompt{'s' if added != 1 else ''}." if added else "No new prompts were added."
            status_update = format_batch_status(selected_row, status_message)
            return updated_rows, next_id, gr.update(value=None), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        def validate_emotion_settings(emo_control_method_value, vec_values):
            mode = emo_control_method_value if isinstance(emo_control_method_value, int) else getattr(
                emo_control_method_value, "value", 0
            )
            try:
                mode = int(mode)
            except (TypeError, ValueError):
                mode = 0
            vec = None
            if mode == 2:
                if sum(vec_values) > 1.5:
                    gr.Warning("Emotion vector sum cannot exceed 1.5. Adjust the sliders and retry.")
                    return mode, None
                vec = vec_values
            return mode, vec

        def build_generation_kwargs(
            do_sample_value,
            top_p_value,
            top_k_value,
            temperature_value,
            length_penalty_value,
            num_beams_value,
            repetition_penalty_value,
            max_mel_tokens_value,
            seed_value,
        ):
            try:
                top_k_int = int(top_k_value)
            except (TypeError, ValueError):
                top_k_int = 0
            try:
                num_beams_int = int(num_beams_value)
            except (TypeError, ValueError):
                num_beams_int = 1
            kwargs = {
                "do_sample": bool(do_sample_value),
                "top_p": float(top_p_value),
                "top_k": top_k_int if top_k_int > 0 else None,
                "temperature": float(temperature_value),
                "length_penalty": float(length_penalty_value),
                "num_beams": num_beams_int,
                "repetition_penalty": float(repetition_penalty_value),
                "max_mel_tokens": int(max_mel_tokens_value),
            }
            seed_int = _normalize_seed(seed_value)
            if seed_int is not None:
                kwargs["seed"] = seed_int
            return kwargs

        def load_dataset_entries(dataset_path, rows, next_id, selected_value, *, progress: Optional[gr.Progress] = None):
            rows = rows or []
            next_id = next_id or 1
            dataset_path = (dataset_path or "").strip()
            if not dataset_path:
                gr.Warning("Provide a dataset train.txt path before loading.")
                dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(row)
                return rows, next_id, gr.update(value=""), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            dataset_path_abs = dataset_path if os.path.isabs(dataset_path) else os.path.abspath(os.path.join(current_dir, dataset_path))
            if not os.path.exists(dataset_path_abs):
                gr.Warning(f"Dataset file not found: {dataset_path_abs}")
                dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(row)
                return rows, next_id, gr.update(value=dataset_path), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            dataset_dir = os.path.dirname(dataset_path_abs)
            candidate_dirs = [dataset_dir, os.path.join(dataset_dir, "wavs"), os.path.join(dataset_dir, "audio")]

            try:
                lines = Path(dataset_path_abs).read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                gr.Warning(f"Failed to read dataset file: {exc}")
                dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(row)
                return rows, next_id, gr.update(value=dataset_path), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            updated_rows = [dict(row) for row in rows]
            prompts_dir = os.path.join(current_dir, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)

            existing_prompts = {os.path.basename(r.get("prompt_path", "")) for r in updated_rows if r.get("prompt_path")}

            added = 0
            missing_audio = 0
            invalid_lines = 0
            total_lines = len(lines)
            _update_progress(progress, 0.0, desc="Parsing dataset")

            for idx, raw_line in enumerate(lines):
                _update_progress(progress, min((idx + 1) / max(total_lines, 1), 0.95), desc=f"Processing line {idx + 1}/{total_lines}")
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split("|", 1)
                if len(parts) != 2:
                    invalid_lines += 1
                    continue
                audio_name = parts[0].strip()
                text_value = parts[1].strip()
                if not audio_name or not text_value:
                    invalid_lines += 1
                    continue

                source_path = None
                for base_dir in candidate_dirs:
                    candidate = os.path.join(base_dir, audio_name)
                    if os.path.exists(candidate):
                        source_path = candidate
                        break
                if not source_path:
                    missing_audio += 1
                    continue

                unique_prefix = f"dataset_{next_id}_{int(time.time() * 1000)}"
                target_name = f"{unique_prefix}_{os.path.basename(audio_name)}"
                if target_name in existing_prompts:
                    target_name = f"{unique_prefix}_{next_id}_{os.path.basename(audio_name)}"
                target_path = os.path.join(prompts_dir, target_name)
                try:
                    shutil.copy(source_path, target_path)
                except Exception as exc:
                    logger.exception("Failed to copy dataset prompt %s", source_path)
                    gr.Warning(f"Failed to copy {audio_name}: {exc}")
                    missing_audio += 1
                    continue

                entry = {
                    "id": next_id,
                    "prompt_path": target_path,
                    "output_path": None,
                    "status": "Pending",
                    "last_generated": "",
                    "text": text_value,
                }
                updated_rows.append(entry)
                existing_prompts.add(target_name)
                added += 1
                next_id += 1

            selected_seed = updated_rows[-1]["id"] if added else selected_value
            dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(
                updated_rows, selected_seed
            )
            table_update = gr.update(value=build_batch_table_data(updated_rows))

            messages = []
            if added:
                messages.append(f"Loaded {added} entries")
            if missing_audio:
                messages.append(f"{missing_audio} missing audio")
            if invalid_lines:
                messages.append(f"{invalid_lines} invalid lines")
            status_message = ", ".join(messages) if messages else "No new entries loaded."
            status_update = format_batch_status(selected_row, status_message)
            _update_progress(progress, 1.0, desc="Dataset load complete")
            return updated_rows, next_id, gr.update(value=dataset_path), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        def generate_all_batch(rows, selected_value, worker_count_value, emo_control_method_value, emo_ref_path, emo_weight_value, vec1_value, vec2_value, vec3_value, vec4_value, vec5_value, vec6_value, vec7_value, vec8_value, emo_text_value, emo_random_value, max_text_tokens_per_sentence_value, duration_seconds_value, *advanced_param_values, progress: Optional[gr.Progress] = None):
            rows = rows or []
            if not rows:
                gr.Warning("Add prompt audio files before generating.")
                dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(row)
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            if parallel_worker_config.get("gpt_path") is None or parallel_worker_config.get("bpe_path") is None:
                gr.Warning("Load a GPT checkpoint and BPE tokenizer before generating.")
                dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(row, "Model not loaded.")
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            vec_values = [vec1_value, vec2_value, vec3_value, vec4_value, vec5_value, vec6_value, vec7_value, vec8_value]
            emo_mode, emo_vector = validate_emotion_settings(emo_control_method_value, vec_values)
            if emo_mode == 2 and emo_vector is None:
                dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(row, "Emotion vector sum exceeded limit.")
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            try:
                max_tokens = int(max_text_tokens_per_sentence_value)
            except (TypeError, ValueError):
                max_tokens = 120

            duration_seconds = _normalize_duration_seconds(duration_seconds_value)

            duration_seconds = _normalize_duration_seconds(duration_seconds_value)

            adv_values = list(advanced_param_values)
            expected_len = len(advanced_params)
            if len(adv_values) < expected_len:
                adv_values.extend([None] * (expected_len - len(adv_values)))
            base_generation_kwargs = build_generation_kwargs(*adv_values[:expected_len])

            outputs_dir = os.path.join(current_dir, "outputs", "tasks")
            os.makedirs(outputs_dir, exist_ok=True)

            jobs: List[GenerationJob] = []
            row_map: Dict[int, Dict[str, Any]] = {}
            for row in rows:
                new_row = dict(row)
                prompt_path = new_row.get("prompt_path")
                if not prompt_path or not os.path.exists(prompt_path):
                    new_row["status"] = "Error: Prompt missing"
                    row_map[new_row["id"]] = new_row
                    continue
                text_value = (new_row.get("text") or "").strip()
                if not text_value:
                    new_row["status"] = "Error: Text missing"
                    row_map[new_row["id"]] = new_row
                    continue
                output_path = os.path.join(outputs_dir, f"batch_row_{new_row['id']}_{int(time.time() * 1000)}.wav")
                new_row["status"] = "Running"
                new_row["output_path"] = output_path
                row_map[new_row["id"]] = new_row

                jobs.append(
                    GenerationJob(
                        row_id=new_row["id"],
                        prompt_path=prompt_path,
                        text=text_value,
                        output_path=output_path,
                        emo_mode=emo_mode,
                        emo_weight=float(emo_weight_value) if emo_mode == 1 else 1.0,
                        emo_vector=emo_vector if emo_mode == 2 else None,
                        emo_text=emo_text_value,
                        emo_random=bool(emo_random_value),
                        emo_ref_path=emo_ref_path if emo_mode == 1 else None,
                        max_tokens=max_tokens,
                        generation_kwargs=dict(base_generation_kwargs),
                        verbose=cmd_args.verbose,
                        duration_seconds=duration_seconds,
                    )
                )

            running_rows = list(row_map.values())
            table_running = gr.update(value=build_batch_table_data(running_rows))
            dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(
                running_rows, selected_value
            )

            if not jobs:
                status_update = format_batch_status(selected_row, "No rows ready for generation.")
                return running_rows, table_running, dropdown_update, prompt_update, output_update, text_update, status_update

            _update_progress(progress, 0.0, desc="Starting parallel generation")
            worker_pool.ensure(worker_count_value)
            results = worker_pool.run_jobs(jobs, progress)

            for row_id, result in results.items():
                row_entry = row_map.get(row_id)
                if not row_entry:
                    continue
                row_entry["status"] = result["status"]
                row_entry["last_generated"] = result.get("timestamp", "")
                if result["output_path"]:
                    row_entry["output_path"] = result["output_path"]

            final_rows = list(row_map.values())
            table_update = gr.update(value=build_batch_table_data(final_rows))
            dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(
                final_rows, resolved_id
            )
            status_update = format_batch_status(selected_row, "Parallel generation finished.")
            return final_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        def regenerate_batch_entry(rows, selected_value, worker_count_value, emo_control_method_value, emo_ref_path, emo_weight_value, vec1_value, vec2_value, vec3_value, vec4_value, vec5_value, vec6_value, vec7_value, vec8_value, emo_text_value, emo_random_value, max_text_tokens_per_sentence_value, duration_seconds_value, *advanced_param_values, progress: Optional[gr.Progress] = None):
            rows = rows or []
            dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(rows, selected_value)
            if not selected_row:
                gr.Warning("Select an entry to regenerate.")
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(None)
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            if parallel_worker_config.get("gpt_path") is None or parallel_worker_config.get("bpe_path") is None:
                gr.Warning("Load a GPT checkpoint and BPE tokenizer before generating.")
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(selected_row, "Model not loaded.")
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            vec_values = [vec1_value, vec2_value, vec3_value, vec4_value, vec5_value, vec6_value, vec7_value, vec8_value]
            emo_mode, emo_vector = validate_emotion_settings(emo_control_method_value, vec_values)
            if emo_mode == 2 and emo_vector is None:
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(selected_row, "Emotion vector sum exceeded limit.")
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            prompt_path = selected_row.get("prompt_path")
            if not prompt_path or not os.path.exists(prompt_path):
                gr.Warning("Prompt audio file is missing.")
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(selected_row, "Prompt audio file missing.")
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            text_value = (selected_row.get("text") or "").strip()
            if not text_value:
                gr.Warning("Enter text for this entry before regenerating.")
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(selected_row, "Text is missing.")
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            try:
                max_tokens = int(max_text_tokens_per_sentence_value)
            except (TypeError, ValueError):
                max_tokens = 120

            adv_values = list(advanced_param_values)
            expected_len = len(advanced_params)
            if len(adv_values) < expected_len:
                adv_values.extend([None] * (expected_len - len(adv_values)))
            generation_kwargs = build_generation_kwargs(*adv_values[:expected_len])
            outputs_dir = os.path.join(current_dir, "outputs", "tasks")
            os.makedirs(outputs_dir, exist_ok=True)
            output_path = os.path.join(outputs_dir, f"batch_row_{selected_row['id']}_{int(time.time() * 1000)}.wav")

            job = GenerationJob(
                row_id=selected_row["id"],
                prompt_path=prompt_path,
                text=text_value,
                output_path=output_path,
                emo_mode=emo_mode,
                emo_weight=float(emo_weight_value) if emo_mode == 1 else 1.0,
                emo_vector=emo_vector if emo_mode == 2 else None,
                emo_text=emo_text_value,
                emo_random=bool(emo_random_value),
                emo_ref_path=emo_ref_path if emo_mode == 1 else None,
                max_tokens=max_tokens,
                generation_kwargs=dict(generation_kwargs),
                verbose=cmd_args.verbose,
                duration_seconds=duration_seconds,
            )

            _update_progress(progress, 0.0, desc="Regenerating entry")
            worker_pool.ensure(worker_count_value)
            results = worker_pool.run_jobs([job], progress)
            result = results.get(job.row_id)
            updated_rows = []
            for row in rows:
                if row.get("id") != job.row_id:
                    updated_rows.append(dict(row))
                    continue
                new_row = dict(row)
                if result:
                    new_row["status"] = result["status"]
                    new_row["output_path"] = result.get("output_path", new_row.get("output_path"))
                    new_row["last_generated"] = result.get("timestamp", "")
                else:
                    new_row["status"] = "Error: Unknown"
                updated_rows.append(new_row)

            table_update = gr.update(value=build_batch_table_data(updated_rows))
            dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(
                updated_rows, job.row_id
            )
            status_update = format_batch_status(selected_row, "Regeneration finished.")
            return updated_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        def delete_batch_entry(rows, selected_value):
            rows = rows or []
            dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(rows, selected_value)
            if not selected_row:
                gr.Warning("Select an entry to delete.")
                table_update = gr.update(value=build_batch_table_data(rows))
                status_update = format_batch_status(None)
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update
            remaining_rows = [dict(row) for row in rows if row.get("id") != selected_row.get("id")]
            dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(remaining_rows, None)
            table_update = gr.update(value=build_batch_table_data(remaining_rows))
            status_update = format_batch_status(row, "Entry deleted.")
            return remaining_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        def clear_batch_rows(rows, next_id):
            dropdown_update = gr.update(choices=[], value=None)
            prompt_update = gr.update(value=None)
            output_update = gr.update(value=None)
            text_update = gr.update(value="")
            status_update = format_batch_status(None, "Batch list cleared.")
            return [], 1, gr.update(value=[]), dropdown_update, prompt_update, output_update, text_update, status_update

        def on_select_batch_entry(selected_value, rows):
            dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            status_update = format_batch_status(row)
            return dropdown_update, prompt_update, output_update, text_update, status_update

        def update_batch_text(new_text, rows, selected_value):
            rows = rows or []
            try:
                selected_id = int(selected_value) if selected_value is not None else None
            except (TypeError, ValueError):
                selected_id = None

            if selected_id is None:
                gr.Warning("Select an entry before editing text.")
                table_update = gr.update(value=build_batch_table_data(rows))
                dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
                status_update = format_batch_status(row)
                return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

            updated_rows = []
            target_row = None
            for row in rows:
                new_row = dict(row)
                if row.get("id") == selected_id:
                    new_row["text"] = new_text
                    if new_row.get("output_path"):
                        new_row["status"] = "Pending"
                    target_row = new_row
                updated_rows.append(new_row)

            dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(updated_rows, selected_id)
            table_update = gr.update(value=build_batch_table_data(updated_rows))
            status_update = format_batch_status(row, "Text updated. Regenerate to apply." if target_row else None)
            return updated_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        def update_prompt_audio():
            return gr.update(interactive=True)

        emo_control_method.select(
            on_method_select,
            inputs=[emo_control_method],
            outputs=[emotion_reference_group, emo_random, emotion_vector_group, emo_text_group],
        )

        input_text_single.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_sentence],
            outputs=[sentences_preview],
        )
        max_text_tokens_per_sentence.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_sentence],
            outputs=[sentences_preview],
        )

        prompt_audio.upload(update_prompt_audio, inputs=[], outputs=[gen_button])

        gen_button.click(
            gen_single,
            inputs=[
                emo_control_method,
                prompt_audio,
                input_text_single,
                emo_upload,
                emo_weight,
                vec1,
                vec2,
                vec3,
                vec4,
                vec5,
                vec6,
                vec7,
                vec8,
                emo_text,
                emo_random,
                max_text_tokens_per_sentence,
                duration_seconds_input,
                *advanced_params,
            ],
            outputs=[output_audio],
            show_progress=True,
        )

        batch_file_input.upload(
            add_batch_prompts,
            inputs=[batch_file_input, batch_rows_state, next_batch_id_state, selected_entry],
            outputs=[batch_rows_state, next_batch_id_state, batch_file_input, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status],
        )

        load_dataset_button.click(
            load_dataset_entries,
            inputs=[dataset_path_input, batch_rows_state, next_batch_id_state, selected_entry],
            outputs=[batch_rows_state, next_batch_id_state, dataset_path_input, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status],
        )

        selected_entry.change(
            on_select_batch_entry,
            inputs=[selected_entry, batch_rows_state],
            outputs=[selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status],
        )

        apply_text_button.click(
            update_batch_text,
            inputs=[batch_text_input, batch_rows_state, selected_entry],
            outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status],
        )

        generate_all_button.click(
            generate_all_batch,
            inputs=[
                batch_rows_state,
                selected_entry,
                worker_count,
                emo_control_method,
                emo_upload,
                emo_weight,
                vec1,
                vec2,
                vec3,
                vec4,
                vec5,
                vec6,
                vec7,
                vec8,
                emo_text,
                emo_random,
                max_text_tokens_per_sentence,
                duration_seconds_input,
                *advanced_params,
            ],
            outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status],
        )

        regenerate_button.click(
            regenerate_batch_entry,
            inputs=[
                batch_rows_state,
                selected_entry,
                worker_count,
                emo_control_method,
                emo_upload,
                emo_weight,
                vec1,
                vec2,
                vec3,
                vec4,
                vec5,
                vec6,
                vec7,
                vec8,
                emo_text,
                emo_random,
                max_text_tokens_per_sentence,
                duration_seconds_input,
                *advanced_params,
            ],
            outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status],
        )

        delete_entry_button.click(
            delete_batch_entry,
            inputs=[batch_rows_state, selected_entry],
            outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status],
        )

        clear_entries_button.click(
            clear_batch_rows,
            inputs=[batch_rows_state, next_batch_id_state],
            outputs=[batch_rows_state, next_batch_id_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status],
        )

    return demo


def main():
    demo = create_demo()
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
