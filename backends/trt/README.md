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

## Quick Start

```bash
# Install dependencies
uv sync --directory backends/trt
source backends/trt/.venv/bin/activate
source backends/trt/scripts/setup_env.sh

# Export ONNX models
bash backends/trt/scripts/export_models.sh

# Convert TRT-LLM checkpoint
PRECISION=fp16 bash backends/trt/scripts/convert_checkpoint.sh

# Build engines (default MAX_BATCH_SIZE=4)
PRECISION=fp16 bash backends/trt/scripts/build_engines.sh

# Or with custom batch size
# PRECISION=fp16 MAX_BATCH_SIZE=2 bash backends/trt/scripts/build_engines.sh

# Run inference
python backends/trt/infer.py \
    --text "Translate for me, what is a surprise!" \
    --speaker examples/voice_01.wav \
    --output output.wav
```

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