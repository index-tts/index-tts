# Faster IndexTTS-2: GPU-Accelerated Inference and Serving for IndexTTS-2

[![arXiv](https://img.shields.io/badge/arXiv-2607.21042-b31b1b.svg)](https://arxiv.org/abs/2607.21042)

This folder contains the GPU-accelerated inference and serving solution for IndexTTS-2, built with the NVIDIA
TensorRT, TensorRT-LLM, and Triton Inference Server. For more technical details, please refer to our paper
[Faster IndexTTS-2](https://arxiv.org/abs/2607.21042).

**Key features:**

- **Fully Accelerated**: All the neural network components are accelerated with NVIDIA TensorRT and TensorRT-LLM.
- **Optimized Serving**: Production serving via Triton Inference Server with dynamic batching of concurrent requests.
- **Real-time Streaming**: Chunked audio generation with low time-to-first-audio (TTFA) for latency-sensitive applications.

---

## Prerequisites

- NVIDIA GPUs (tested on NVIDIA A100 80GB and RTX A6000 48GB)
- Docker with NVIDIA Container Toolkit
- IndexTTS-2 checkpoints in the `checkpoints` folder.
- Example audio files in the `examples` folder.

Please follow the README of [index-tts](https://github.com/index-tts/index-tts) to download the checkpoints and example audios.

---

## Quick Start (Docker)

```bash
# Build from a pre-built image with all dependencies and pre-exported ONNX models (--fast flag).
# Pulls the image -> builds TRT engines
bash deploy/scripts/build_image.sh --triton --fast fp16

# Custom max batch size (default: 4, increase or decrease based on your GPU)
# MAX_BATCH_SIZE=2 bash deploy/scripts/build_image.sh --triton --fast fp16

# Or build from scratch
# Pull the plain Triton image -> install required packages -> export ONNX models -> build engines
# bash deploy/scripts/build_image.sh --triton fp16

# Or build in two steps (export once, then rebuild engines as needed):
# bash deploy/scripts/build_image.sh --triton --export-only
# bash deploy/scripts/build_image.sh --triton --engines-only fp16

# Or if you want to use PyTriton instead of native Triton Inference Server
# bash deploy/scripts/build_image.sh --pytriton --fast fp16

# Run the server (streaming mode)
# If you build using PyTriton, replace the image name with faster-indextts-2-pytriton:fp16
docker run --rm --gpus all --network=host faster-indextts-2-triton:fp16 \
    tritonserver \
      --model-repository=/workspace/indextts/deploy/serving/model_repository \
      --model-control-mode=explicit \
      --load-model=indextts2_stream \
      --log-verbose=1

# Send a streaming request (pip install tritonclient[grpc] soundfile numpy)
python deploy/serving/triton_client.py --mode streaming \
    --text "Translate for me, what is a surprise!" \
    --speaker_audio examples/voice_01.wav \
    --output output_s.wav

# Run the server (non-streaming mode)
docker run --rm --gpus all --network=host faster-indextts-2-triton:fp16 \
    tritonserver \
      --model-repository=/workspace/indextts/deploy/serving/model_repository \
      --model-control-mode=explicit \
      --load-model=indextts2 \
      --log-verbose=1

# Send a non-streaming request (pip install tritonclient[grpc] soundfile numpy)
python deploy/serving/triton_client.py --mode non-streaming \
    --text "Translate for me, what is a surprise!" \
    --speaker_audio examples/voice_01.wav \
    --output output_ns.wav
```

---

## Quick Start (Manual)

```bash
# Install dependencies
uv sync --directory deploy
source deploy/.venv/bin/activate
source deploy/scripts/setup_env.sh

# Export ONNX models
bash deploy/scripts/export_models.sh

# Convert TRT-LLM checkpoint
PRECISION=fp16 bash deploy/scripts/convert_checkpoint.sh

# Build engines (default MAX_BATCH_SIZE=4)
PRECISION=fp16 bash deploy/scripts/build_engines.sh

# Or with custom batch size
# PRECISION=fp16 MAX_BATCH_SIZE=2 bash deploy/scripts/build_engines.sh

# Run inference
python deploy/infer.py \
    --text "Translate for me, what is a surprise!" \
    --speaker examples/voice_01.wav \
    --output output.wav
```

---

## Python API

```python
from deploy.pipeline import FasterIndexTTS2
from deploy.utils import resolve_engine_paths
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

## Scaling with multiple Triton instances

If you have sufficient GPU memory or multiple GPUs, consider adding more instances per GPU or spreading across multiple GPUs by editing the `config.pbtxt`.

```
# 1 instance per GPU on a 2-GPU system
instance_group [
  { count: 1, kind: KIND_GPU, gpus: [0] },
  { count: 1, kind: KIND_GPU, gpus: [1] }
]

# 2 instances per GPU on a 2-GPU system
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [0] },
  { count: 2, kind: KIND_GPU, gpus: [1] }
]
```

> **Important:** `max_batch_size` in `config.pbtxt` must match the engine build-time `MAX_BATCH_SIZE` (default: 4). Do not increase it beyond what the engines were built with. You may safely change `instance_group` (count, gpus) and `dynamic_batching` settings.

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