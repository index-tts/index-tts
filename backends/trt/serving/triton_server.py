"""PyTriton server for FasterIndexTTS2 pipeline.

Supports two modes (one per server instance):
  --mode non-streaming   Registers model "indextts2" (standard request-response)
  --mode streaming       Registers model "indextts2_stream" (decoupled, chunked audio)

Dynamic batching groups concurrent client requests into GPU batches.

Usage:
    # Non-streaming server
    python triton_server.py --mode non-streaming

    # Streaming server
    python triton_server.py --mode streaming

    # With custom params
    python triton_server.py --mode streaming --max_batch_size 4 --num_beams 1
"""

import argparse
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig, TritonSecurityConfig

logger = logging.getLogger("triton_server")

# Maximum allowed size for a speaker audio payload (50 MB).
MAX_SPEAKER_AUDIO_BYTES = 50 * 1024 * 1024

# ---------------------------------------------------------------------------
# Speaker cache (LRU)
# ---------------------------------------------------------------------------

class SpeakerCache:
    """Thread-safe LRU cache for pre-computed speaker conditions."""

    def __init__(self, pipeline, max_size=64):
        self._pipeline = pipeline
        self._max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def _audio_key(self, audio_bytes: bytes) -> str:
        return hashlib.sha256(audio_bytes).hexdigest()[:16]

    def get(self, audio_bytes: bytes):
        key = self._audio_key(audio_bytes)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def get_or_compute(self, audio_bytes: bytes):
        cached = self.get(audio_bytes)
        if cached is not None:
            return cached

        if len(audio_bytes) > MAX_SPEAKER_AUDIO_BYTES:
            raise ValueError(
                f"speaker_audio payload ({len(audio_bytes)} bytes) exceeds the "
                f"{MAX_SPEAKER_AUDIO_BYTES // (1024 * 1024)} MB limit."
            )

        import torch, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            condition = self._pipeline.preload_speaker(tmp_path)
        finally:
            os.unlink(tmp_path)

        key = self._audio_key(audio_bytes)
        with self._lock:
            self._cache[key] = condition
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return condition


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FasterIndexTTS2Server:
    def __init__(self, args):
        from backends.trt.pipeline import FasterIndexTTS2

        self.args = args
        logger.info("Loading FasterIndexTTS2 pipeline...")
        self.pipeline = FasterIndexTTS2(
            config_path=os.path.join(args.model_dir, "config.yaml"),
            model_dir=args.model_dir,
            gpt_engine_dir=args.gpt_engine_dir,
            speed_emb_path=args.speed_emb,
            speech_semantic_encoder_engine=args.speech_semantic_encoder,
            semantic_codec_engine=args.semantic_codec,
            speaker_perceiver_conditioner_engine=args.speaker_perceiver_conditioner,
            emotion_perceiver_conditioner_engine=args.emotion_perceiver_conditioner,
            latent_projector_engine=args.latent_projector,
            length_regulator_engine=args.length_regulator,
            campplus_engine=args.campplus,
            dit_engine=args.dit,
            bigvgan_engine=args.bigvgan,
            streaming_chunk_size=args.streaming_chunk_size,
            streaming_overlap_size=args.streaming_overlap_size,
        )
        self.speaker_cache = SpeakerCache(self.pipeline, max_size=args.speaker_cache_size)
        logger.info("Pipeline ready")

    def _decode_inputs(self, text, speaker_audio, emo_speaker_audio=None,
                       emo_alpha=None, emo_vector=None):
        """Decode a batch of inputs from Triton numpy arrays.

        Returns per-sample: (texts, spk_conditions, emo_spk_conditions, emo_alphas, emo_vectors)
        """
        batch_size = text.shape[0]
        texts = []
        spk_conditions = []
        emo_spk_conditions = []
        emo_alphas = []
        emo_vectors = []

        for b in range(batch_size):
            t = text[b, 0]
            if isinstance(t, bytes):
                t = t.decode("utf-8")
            texts.append(t)

            spk_audio_bytes = speaker_audio[b, 0]
            if isinstance(spk_audio_bytes, np.void):
                spk_audio_bytes = spk_audio_bytes.tobytes()
            spk_conditions.append(self.speaker_cache.get_or_compute(spk_audio_bytes))

            if emo_speaker_audio is not None and emo_speaker_audio.size > 0:
                emo_bytes = emo_speaker_audio[b, 0]
                if isinstance(emo_bytes, np.void):
                    emo_bytes = emo_bytes.tobytes()
                if len(emo_bytes) > 0:
                    emo_spk_conditions.append(self.speaker_cache.get_or_compute(emo_bytes))
                else:
                    emo_spk_conditions.append(None)
            else:
                emo_spk_conditions.append(None)

            if emo_alpha is not None and emo_alpha.size > 0:
                emo_alphas.append(float(emo_alpha[b, 0]))
            else:
                emo_alphas.append(1.0)

            if emo_vector is not None and emo_vector.size > 0:
                vec = emo_vector[b]
                if np.any(vec != 0):
                    emo_vectors.append(vec.tolist())
                else:
                    emo_vectors.append(None)
            else:
                emo_vectors.append(None)

        return texts, spk_conditions, emo_spk_conditions, emo_alphas, emo_vectors

    @batch
    def infer_non_streaming(self, text, speaker_audio, emo_speaker_audio=None,
                            emo_alpha=None, emo_vector=None):
        """Non-streaming inference: returns full audio per request."""
        texts, spk_conds, emo_spk_conds, emo_alphas, emo_vecs = \
            self._decode_inputs(text, speaker_audio, emo_speaker_audio, emo_alpha, emo_vector)

        batch_size = len(texts)
        args = self.args

        results = self.pipeline.generate(
            text=texts if batch_size > 1 else texts[0],
            speaker=spk_conds if batch_size > 1 else spk_conds[0],
            emo_speaker=emo_spk_conds if batch_size > 1 else emo_spk_conds[0],
            emo_alpha=emo_alphas if batch_size > 1 else emo_alphas[0],
            emo_vector=emo_vecs if batch_size > 1 else emo_vecs[0],
            max_mel_tokens=args.max_mel_tokens,
            num_beams=args.num_beams,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )

        if batch_size == 1:
            results = [results]

        max_len = max(r[1].shape[0] for r in results)
        audio_batch = np.zeros((batch_size, max_len), dtype=np.int16)
        sr_batch = np.full((batch_size, 1), 22050, dtype=np.int32)
        length_batch = np.zeros((batch_size, 1), dtype=np.int32)

        for b, (sr, audio) in enumerate(results):
            audio_flat = audio.flatten().astype(np.int16)
            audio_batch[b, :len(audio_flat)] = audio_flat
            length_batch[b, 0] = len(audio_flat)

        return {
            "audio": audio_batch,
            "sample_rate": sr_batch,
            "audio_length": length_batch,
        }

    @batch
    def infer_streaming(self, text, speaker_audio, emo_speaker_audio=None,
                        emo_alpha=None, emo_vector=None):
        """Streaming inference: yields audio chunks per request."""
        texts, spk_conds, emo_spk_conds, emo_alphas, emo_vecs = \
            self._decode_inputs(text, speaker_audio, emo_speaker_audio, emo_alpha, emo_vector)

        batch_size = len(texts)
        args = self.args

        from backends.trt.pipeline import BatchAudioChunk, AudioChunk

        gen = self.pipeline.generate(
            text=texts if batch_size > 1 else texts[0],
            speaker=spk_conds if batch_size > 1 else spk_conds[0],
            emo_speaker=emo_spk_conds if batch_size > 1 else emo_spk_conds[0],
            emo_alpha=emo_alphas if batch_size > 1 else emo_alphas[0],
            emo_vector=emo_vecs if batch_size > 1 else emo_vecs[0],
            stream=True,
            max_mel_tokens=args.max_mel_tokens,
            num_beams=args.num_beams,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )

        sample_finished = [False] * batch_size

        for chunk in gen:
            if batch_size == 1:
                audio = chunk.audio
                is_last = chunk.is_last
                yield {
                    "audio_chunk": audio.reshape(1, -1),
                    "is_last": np.array([[is_last]], dtype=np.bool_),
                    "sample_rate": np.array([[22050]], dtype=np.int32),
                }
                if is_last:
                    break
            else:
                max_chunk_len = 1
                for b in range(batch_size):
                    a = chunk.audio_list[b]
                    if a is not None and len(a) > 0:
                        max_chunk_len = max(max_chunk_len, len(a))

                audio_batch = np.zeros((batch_size, max_chunk_len), dtype=np.int16)
                is_last_batch = np.zeros((batch_size, 1), dtype=np.bool_)
                sr_batch = np.full((batch_size, 1), 22050, dtype=np.int32)

                for b in range(batch_size):
                    a = chunk.audio_list[b]
                    if a is not None and len(a) > 0:
                        audio_batch[b, :len(a)] = a
                    is_last_batch[b, 0] = chunk.done_list[b]
                    if chunk.done_list[b]:
                        sample_finished[b] = True

                yield {
                    "audio_chunk": audio_batch,
                    "is_last": is_last_batch,
                    "sample_rate": sr_batch,
                }

                if all(sample_finished):
                    break


INPUT_TENSORS = [
    Tensor(name="text", dtype=bytes, shape=(1,)),
    Tensor(name="speaker_audio", dtype=bytes, shape=(1,)),
    Tensor(name="emo_speaker_audio", dtype=bytes, shape=(1,), optional=True),
    Tensor(name="emo_alpha", dtype=np.float32, shape=(1,), optional=True),
    Tensor(name="emo_vector", dtype=np.float32, shape=(8,), optional=True),
]

OUTPUT_TENSORS_NON_STREAMING = [
    Tensor(name="audio", dtype=np.int16, shape=(-1,)),
    Tensor(name="sample_rate", dtype=np.int32, shape=(1,)),
    Tensor(name="audio_length", dtype=np.int32, shape=(1,)),
]

OUTPUT_TENSORS_STREAMING = [
    Tensor(name="audio_chunk", dtype=np.int16, shape=(-1,)),
    Tensor(name="is_last", dtype=np.bool_, shape=(1,)),
    Tensor(name="sample_rate", dtype=np.int32, shape=(1,)),
]


def main():
    parser = argparse.ArgumentParser(description="FasterIndexTTS2 PyTriton Server")
    parser.add_argument("--mode", choices=["streaming", "non-streaming"], default="non-streaming")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8", "int4"])

    # Override engine paths (optional, auto-resolved from --precision if not specified)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--backend_dir", default=None)
    parser.add_argument("--gpt_engine_dir", default=None)
    parser.add_argument("--speed_emb", default=None)

    # Generation params (server-side, not client-specified)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=10.0)
    parser.add_argument("--max_mel_tokens", type=int, default=1500)
    parser.add_argument("--streaming_chunk_size", type=int, default=100)
    parser.add_argument("--streaming_overlap_size", type=int, default=5)

    # Triton config
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--max_queue_delay_ms", type=int, default=100)
    parser.add_argument("--grpc_port", type=int, default=8001)
    parser.add_argument("--http_port", type=int, default=8000)
    parser.add_argument("--speaker_cache_size", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Auto-resolve engine paths from precision
    from backends.trt.utils import resolve_engine_paths
    _FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_FILE_DIR)))
    backend_dir = args.backend_dir or os.path.join(_PROJECT_ROOT, "backends", "trt")
    paths = resolve_engine_paths(args.precision, backend_dir, _PROJECT_ROOT)
    args.model_dir = args.model_dir or paths["model_dir"]
    args.gpt_engine_dir = args.gpt_engine_dir or paths["gpt_engine_dir"]
    args.speed_emb = args.speed_emb or paths["speed_emb_path"]
    args.speech_semantic_encoder = paths["speech_semantic_encoder_engine"]
    args.semantic_codec = paths["semantic_codec_engine"]
    args.speaker_perceiver_conditioner = paths["speaker_perceiver_conditioner_engine"]
    args.emotion_perceiver_conditioner = paths["emotion_perceiver_conditioner_engine"]
    args.latent_projector = paths["latent_projector_engine"]
    args.length_regulator = paths["length_regulator_engine"]
    args.campplus = paths["campplus_engine"]
    args.dit = paths["dit_engine"]
    args.bigvgan = paths["bigvgan_engine"]

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    server = FasterIndexTTS2Server(args)

    triton_config = TritonConfig(
        grpc_port=args.grpc_port,
        http_port=args.http_port,
        log_verbose=3 if args.verbose else 0,
    )

    security_config = TritonSecurityConfig(restricted_endpoints=[])

    batcher = DynamicBatcher(
        max_queue_delay_microseconds=args.max_queue_delay_ms * 1000,
    )

    with Triton(config=triton_config, security_config=security_config) as triton:
        if args.mode == "non-streaming":
            logger.info("Registering model: indextts2 (non-streaming)")
            triton.bind(
                model_name="indextts2",
                infer_func=server.infer_non_streaming,
                inputs=INPUT_TENSORS,
                outputs=OUTPUT_TENSORS_NON_STREAMING,
                config=ModelConfig(
                    max_batch_size=args.max_batch_size,
                    batcher=batcher,
                ),
                strict=False,
            )
        else:
            logger.info("Registering model: indextts2_stream (streaming/decoupled)")
            triton.bind(
                model_name="indextts2_stream",
                infer_func=server.infer_streaming,
                inputs=INPUT_TENSORS,
                outputs=OUTPUT_TENSORS_STREAMING,
                config=ModelConfig(
                    max_batch_size=args.max_batch_size,
                    batcher=batcher,
                    decoupled=True,
                ),
                strict=False,
            )

        logger.info(f"Server starting (mode={args.mode}, grpc={args.grpc_port}, http={args.http_port})")
        triton.serve()


if __name__ == "__main__":
    main()
