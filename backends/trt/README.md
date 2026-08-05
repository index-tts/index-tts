# Faster IndexTTS-2: GPU-Accelerated Inference and Serving for IndexTTS-2

[![arXiv](https://img.shields.io/badge/arXiv-2607.21042-b31b1b.svg)](https://arxiv.org/abs/2607.21042)

> **Attribution.** This backend is taken from
> [MuyangDu/index-tts](https://github.com/MuyangDu/index-tts/tree/main/deploy) —
> *Faster IndexTTS-2*, by Muyang Du, Shuang Yu and Junjie Lai
> ([arXiv:2607.21042](https://arxiv.org/abs/2607.21042)). The code below is
> theirs; it was copied here with only path and module renames (`deploy/` →
> `backends/trt/`), and with the untested Docker and native-Triton serving paths
> removed. "We"/"our" in this document refers to those authors, not the
> IndexTTS team.

This folder contains the GPU-accelerated inference and serving solution for IndexTTS-2, built with the NVIDIA
TensorRT, TensorRT-LLM, and Triton Inference Server. For more technical details, please refer to our paper
[Faster IndexTTS-2](https://arxiv.org/abs/2607.21042).

**Key features:**

- **Fully Accelerated**: All the neural network components are accelerated with NVIDIA TensorRT and TensorRT-LLM.
- **Optimized Serving**: Production serving via Triton Inference Server with dynamic batching of concurrent requests.
- **Real-time Streaming**: Chunked audio generation with low time-to-first-audio (TTFA) for latency-sensitive applications.

---

## Prerequisites

- NVIDIA GPUs (tested on NVIDIA A100 80GB, RTX A6000 48GB and RTX 4090 24GB)
- IndexTTS-2 checkpoints in the `checkpoints` folder.
- Example audio files in the `examples` folder.
- **OpenMPI 4.x on the host.** `tensorrt_llm` links `libmpi.so.40` and needs the
  `orted` binary for singleton init, so `import tensorrt_llm` fails with
  `RuntimeError: cannot load MPI library` without it. On Debian/Ubuntu:
  `apt-get install libopenmpi3 openmpi-bin`. Intel MPI (`impi-rt` from PyPI) is
  not a substitute — it lacks `OMPI_COMM_TYPE_HOST` and aborts in `MPI_Init_thread`.
  Upstream ran this inside `nvcr.io/nvidia/tritonserver`, which bundles HPC-X
  OpenMPI, so the dependency is invisible there.

Please follow the README of [index-tts](https://github.com/index-tts/index-tts) to download the checkpoints and example audios.

---

## Verified environment

This backend runs in its own venv (`uv sync --directory backends/trt`), separate
from the project's, because TensorRT-LLM's pins conflict with the root lockfile.
What has actually been run:

| | Verified |
|---|---|
| GPU | 1x RTX 4090 24GB, driver CUDA 12.4 |
| Python | 3.12.11 (backend venv) |
| tensorrt / tensorrt-llm | 10.11.0.33 / 0.21.0 |
| torch | 2.7.1+cu128 |
| OpenMPI | 4.1.2 |
| Settings | `PRECISION=fp16`, `MAX_BATCH_SIZE=1` |
| Pipeline | 9 ONNX exports, 10 engines |
| Serving | PyTriton non-streaming and streaming |

Not verified: `MAX_BATCH_SIZE > 1`, `int8`/`int4` precision, and multi-GPU
serving. Upstream additionally reports A100 80GB and RTX A6000 48GB.

### Measured RTF

Median of 3 reps per text on the box above, first iteration discarded (it runs
~55% slow), all three pinned to GPU 0. RTF is wall time over generated audio
duration, so lower is faster.

| Text | 2.0 PyTorch fp16 | 2.5 PyTorch bf16 | 2.0 + TensorRT fp16 |
|---|---|---|---|
| 7 chars | 0.3817 | 0.2593 | **0.1515** |
| 16 chars | 0.3301 | 0.2091 | **0.1397** |
| 28 chars | 0.3184 | 0.1990 | **0.1327** |
| 80 chars | 0.3212 | 0.1909 | **0.1333** |
| overall | 0.3263 | 0.2035 | **0.1365** |

TensorRT is 2.39x the 2.0 PyTorch path on the same weights. The 2.5 column is a
different model (half the token rate, 50.1 → 25.1 tokens per audio second) and
has no TensorRT engines yet, so it is context rather than a like-for-like
comparison.

Caveats worth knowing before quoting these: each path generates a different
audio length for the same text (80 chars: 18.25s / 14.01s / 15.26s), since
sampling differs; RTF normalizes for that but absolute latency does not. The
half-precision dtypes differ because each version exposes only one flag (2.0
`use_fp16`, 2.5 `use_bf16`). Both PyTorch columns need `kv_cache=True`, which is
2.0's default and became 2.5's in this commit.

### Single-venv installation

Installing this backend into the project's main venv instead of an isolated one
resolves cleanly (228 packages, Python 3.10.14) but downgrades the main
environment, so it is **not** how this backend is set up:

| | Main project | Merged |
|---|---|---|
| Python | 3.11.13 | 3.10.14 |
| torch | 2.8.* | 2.7.1 |
| numpy | 2.2.6 | 1.26.4 |
| transformers | 4.52.1 | 4.51.3 |
| protobuf | 3.19.6 | 5.29.6 |

Python can only be 3.10: TensorRT-LLM 0.21.0 publishes wheels for cp310 and
cp312 only, and the project caps Python at `<3.12` for llvmlite. Note that this
would revert the numpy 2.2.6 and Python 3.11.13 upgrades from #720 and #721.

Only dependency resolution was tested here — the merged environment was never
installed or run, and `flash-attn==2.8.3.post1` (the `accel` extra, built for
torch 2.8) was not part of that test.

---

## Quick Start

`run.sh` is the only entry point you need. It activates the venv, sets
`PYTHONPATH`/`LD_LIBRARY_PATH`, locates OpenMPI and checks the environment before
doing anything, so these work from a bare shell with nothing sourced:

```bash
# Install dependencies
uv sync --directory backends/trt

# Check the host can run this; prints a fix hint for anything missing
bash backends/trt/run.sh check

# Export ONNX -> convert GPT checkpoint -> build engines (slow, one time)
bash backends/trt/run.sh build

# Synthesize
bash backends/trt/run.sh infer \
    --text "Translate for me, what is a surprise!" \
    --speaker examples/voice_01.wav \
    --output output.wav
```

`bash backends/trt/run.sh --help` lists the commands; `infer --help` and
`serve --help` forward to the underlying scripts.

| Variable | Purpose |
|---|---|
| `PRECISION` | `fp32`/`fp16`/`int8`/`int4`, default `fp16` |
| `MAX_BATCH_SIZE` | Engine build batch size, default `1` |
| `OPENMPI_PREFIX` | OpenMPI prefix, if it isn't on the default library path |
| `SKIP_CHECK=1` | Skip the environment check |

OpenMPI is searched for under `$OPENMPI_PREFIX`, `~/local-mpi/root/usr`,
`/usr/lib/x86_64-linux-gnu/openmpi`, `/usr`, `/usr/local` and `/opt/hpcx/ompi`.
That last one means it works unchanged inside the NVIDIA Triton images.

The individual steps under `scripts/` (`export_models.sh`,
`convert_checkpoint.sh`, `build_engines.sh`) can still be run directly if you
need to redo just one of them.

---

## Serving (PyTriton)

`triton_server.py` starts an in-process Triton server via PyTriton — no container
required, since `nvidia-pytriton` bundles the server binary.

```bash
# Start the server. --max_batch_size must not exceed the MAX_BATCH_SIZE the
# engines were built with.
python backends/trt/serving/triton_server.py \
    --mode non-streaming --precision fp16 --max_batch_size 1

# Or streaming mode (decoupled, chunked audio)
# python backends/trt/serving/triton_server.py \
#     --mode streaming --precision fp16 --max_batch_size 1

# Send a request from another shell
python backends/trt/serving/triton_client.py --mode non-streaming \
    --url localhost:8001 \
    --text "Translate for me, what is a surprise!" \
    --speaker_audio examples/voice_01.wav \
    --output output_ns.wav
```

> **Warning:** the server binds `0.0.0.0` on ports 8000/8001/8002 with
> `restricted_endpoints=[]`, i.e. no authentication. Do not expose it on an
> untrusted network without putting access control in front of it.

---

## Python API

```python
from backends.trt.pipeline import FasterIndexTTS2
from backends.trt.utils import resolve_engine_paths
import os

paths = resolve_engine_paths("fp16")
pipeline = FasterIndexTTS2(
    config_path=os.path.join(paths["model_dir"], "config.yaml"),
    model_dir=paths["model_dir"],
    gpt_engine_dir=paths["gpt_engine_dir"],
    speed_emb_path=paths["speed_emb_path"],
    speech_semantic_encoder_engine=paths["speech_semantic_encoder_engine"],
    semantic_codec_engine=paths["semantic_codec_engine"],
    speaker_perceiver_conditioner_engine=paths["speaker_perceiver_conditioner_engine"],
    emotion_perceiver_conditioner_engine=paths["emotion_perceiver_conditioner_engine"],
    latent_projector_engine=paths["latent_projector_engine"],
    length_regulator_engine=paths["length_regulator_engine"],
    campplus_engine=paths["campplus_engine"],
    dit_engine=paths["dit_engine"],
    bigvgan_engine=paths["bigvgan_engine"],
)

# Non-streaming
sr, audio = pipeline.generate(text="Hello world", speaker=pipeline.preload_speaker("voice.wav"))

# Streaming
spk = pipeline.preload_speaker("voice.wav")
for chunk in pipeline.generate(text="Hello world", speaker=spk, stream=True):
    play(chunk.audio)  # chunk.is_last indicates final chunk
```

---

## Throughput tuning

`triton_server.py` batches concurrent requests dynamically. The relevant flags:

| Flag | Default | Notes |
|---|---|---|
| `--max_batch_size` | 4 | Must not exceed the engines' build-time `MAX_BATCH_SIZE` |
| `--max_queue_delay_ms` | 100 | How long to wait while filling a batch |
| `--num_beams` | 3 | Must not exceed the engine's `max_beam_width` |
| `--speaker_cache_size` | 64 | Cached speaker conditionings |

To serve on a specific GPU, set `CUDA_VISIBLE_DEVICES` before starting the
server; run one server process per GPU to use several.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{du2026faster,
  title={Faster IndexTTS-2: Accelerating and Streaming Autoregressive Zero-Shot Text-to-Speech Synthesis on GPUs},
  author={Du, Muyang and Yu, Shuang and Lai, Junjie},
  journal={arXiv preprint arXiv:2607.21042},
  year={2026}
}
```

---

## License

The acceleration and serving code in this folder is provided as-is for research and development purposes. Usage of the IndexTTS-2 model weights and checkpoints is subject to the [index-tts license](https://github.com/index-tts/index-tts/blob/main/LICENSE). Please ensure you comply with the original license terms when using Faster IndexTTS-2.