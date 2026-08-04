"""Standalone inference script for FasterIndexTTS2.

Runs inference directly without Triton. Supports single/batched and
streaming/non-streaming modes.

Usage:
    # Basic inference
    python backends/trt/infer.py --text "Hello world" --speaker voice.wav --output result.wav

    # Streaming (prints time-to-first-chunk)
    python backends/trt/infer.py --text "Hello world" --speaker voice.wav --output result.wav --stream

    # Batched inference (multiple texts, single speaker)
    python backends/trt/infer.py --text "First." "Second." --speaker voice.wav --output out1.wav out2.wav

    # Specify precision (auto-resolves engine dirs)
    python backends/trt/infer.py --text "Hello" --speaker voice.wav --output out.wav --precision fp16

    # With emotion control
    python backends/trt/infer.py --text "Hello" --speaker voice.wav --output out.wav \\
        --emo_speaker emo.wav --emo_alpha 0.8
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser(description="FasterIndexTTS2 Standalone Inference")
    parser.add_argument("--text", nargs="+", required=True, help="Text(s) to synthesize")
    parser.add_argument("--speaker", required=True, help="Path to reference speaker audio")
    parser.add_argument("--output", nargs="+", required=True, help="Output WAV path(s)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")

    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8", "int4"],
                        help="Engine precision (determines engine directory paths)")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--backend_dir", default=None,
                        help="Path to backends/trt/ directory")

    # Emotion options
    parser.add_argument("--emo_speaker", default=None, help="Emotion reference audio path")
    parser.add_argument("--emo_alpha", type=float, default=1.0, help="Emotion blend weight")
    parser.add_argument("--emo_vector", type=float, nargs=8, default=None,
                        help="8-D emotion vector")

    # Generation params
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=10.0)
    parser.add_argument("--max_mel_tokens", type=int, default=1500)
    parser.add_argument("--streaming_chunk_size", type=int, default=100)
    parser.add_argument("--streaming_overlap_size", type=int, default=5)

    # Override individual engine paths (optional, overrides --precision)
    parser.add_argument("--gpt_engine_dir", default=None)
    parser.add_argument("--trt_engine_dir", default=None)
    parser.add_argument("--speed_emb", default=None)

    args = parser.parse_args()

    # Auto-detect project root from this file's location
    _FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_FILE_DIR))

    # Resolve defaults relative to project root
    backend_dir = args.backend_dir or os.path.join(_PROJECT_ROOT, "backends", "trt")
    model_dir = args.model_dir or os.path.join(_PROJECT_ROOT, "checkpoints")

    # Resolve engine paths from precision
    from backends.trt.utils import resolve_engine_paths
    paths = resolve_engine_paths(args.precision, backend_dir, _PROJECT_ROOT)
    trt_engine_dir = args.trt_engine_dir or paths["trt_engine_dir"]
    gpt_engine_dir = args.gpt_engine_dir or paths["gpt_engine_dir"]
    speed_emb_path = args.speed_emb or paths["speed_emb_path"]

    # Validate paths
    if not os.path.exists(speed_emb_path):
        print(f"ERROR: speed_emb.pt not found at {speed_emb_path}")
        print("Run backends/trt/scripts/export_models.sh first.")
        sys.exit(1)
    if not os.path.isdir(gpt_engine_dir):
        print(f"ERROR: GPT engine dir not found at {gpt_engine_dir}")
        print(f"Run PRECISION={args.precision} bash backends/trt/scripts/build_engines.sh first.")
        sys.exit(1)

    from backends.trt.pipeline import FasterIndexTTS2

    print("=" * 60)
    print("Initializing FasterIndexTTS2...")
    print(f"  precision: {args.precision}")
    print(f"  trt_engine_dir: {trt_engine_dir}")
    print(f"  gpt_engine_dir: {gpt_engine_dir}")
    print("=" * 60)

    t0 = time.perf_counter()
    pipeline = FasterIndexTTS2(
        config_path=os.path.join(model_dir, "config.yaml"),
        model_dir=model_dir,
        gpt_engine_dir=gpt_engine_dir,
        speed_emb_path=speed_emb_path,
        speech_semantic_encoder_engine=os.path.join(trt_engine_dir, "speech_semantic_encoder.engine"),
        semantic_codec_engine=os.path.join(trt_engine_dir, "semantic_codec.engine"),
        speaker_perceiver_conditioner_engine=os.path.join(trt_engine_dir, "speaker_perceiver_conditioner.engine"),
        emotion_perceiver_conditioner_engine=os.path.join(trt_engine_dir, "emotion_perceiver_conditioner.engine"),
        latent_projector_engine=os.path.join(trt_engine_dir, "latent_projector.engine"),
        length_regulator_engine=os.path.join(trt_engine_dir, "length_regulator.engine"),
        campplus_engine=os.path.join(trt_engine_dir, "campplus.engine"),
        dit_engine=os.path.join(trt_engine_dir, "dit.engine"),
        bigvgan_engine=os.path.join(trt_engine_dir, "bigvgan.engine"),
        streaming_chunk_size=args.streaming_chunk_size if args.stream else 0,
        streaming_overlap_size=args.streaming_overlap_size,
    )
    print(f"Pipeline initialized in {time.perf_counter() - t0:.2f}s\n")

    # Preload speaker
    print("Preloading speaker...")
    t0 = time.perf_counter()
    spk = pipeline.preload_speaker(args.speaker)
    print(f"  Speaker loaded in {time.perf_counter() - t0:.3f}s\n")

    # Preload emotion speaker if provided
    emo_spk = None
    if args.emo_speaker:
        print("Preloading emotion speaker...")
        emo_spk = pipeline.preload_speaker(args.emo_speaker)

    texts = args.text
    outputs = args.output
    is_batch = len(texts) > 1

    if len(outputs) == 1 and is_batch:
        base, ext = os.path.splitext(outputs[0])
        outputs = [f"{base}_{i}{ext}" for i in range(len(texts))]

    gen_kwargs = dict(
        max_mel_tokens=args.max_mel_tokens,
        num_beams=args.num_beams,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

    if args.stream and not is_batch:
        # --- Streaming single inference ---
        print("=" * 60)
        print(f"Streaming inference: {texts[0]!r}")
        print("=" * 60)
        t0 = time.perf_counter()
        first_chunk_time = None
        chunks = []

        for chunk in pipeline.generate(
            text=texts[0], speaker=spk,
            emo_speaker=emo_spk, emo_alpha=args.emo_alpha,
            emo_vector=args.emo_vector,
            stream=True, **gen_kwargs,
        ):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - t0
            chunks.append(chunk.audio)
            print(f"  Chunk: {len(chunk.audio)} samples, is_last={chunk.is_last}")
            if chunk.is_last:
                break

        elapsed = time.perf_counter() - t0
        full_audio = np.concatenate(chunks)
        sr = 22050
        duration = len(full_audio) / sr

        print(f"\n  Duration:          {duration:.2f}s")
        print(f"  Time-to-first:     {first_chunk_time:.3f}s")
        print(f"  Total time:        {elapsed:.2f}s")
        print(f"  RTF:               {elapsed / duration:.4f}")

        wav = torch.tensor(full_audio, dtype=torch.float32).unsqueeze(0).to(torch.int16)
        torchaudio.save(outputs[0], wav, sr)
        print(f"  Saved: {outputs[0]}")

    elif is_batch and args.stream:
        # --- Batched streaming inference ---
        print("=" * 60)
        print(f"Batched streaming inference: {len(texts)} texts")
        print("=" * 60)
        t0 = time.perf_counter()
        first_chunk_time = None
        per_sample_chunks = [[] for _ in range(len(texts))]

        for batch_chunk in pipeline.generate(
            text=texts, speaker=spk,
            emo_speaker=emo_spk, emo_alpha=args.emo_alpha,
            emo_vector=args.emo_vector,
            stream=True, **gen_kwargs,
        ):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - t0
            for b in range(len(texts)):
                a = batch_chunk.audio_list[b]
                if a is not None and len(a) > 0:
                    per_sample_chunks[b].append(a)

            done_str = ", ".join(
                f"b{b}:{'done' if batch_chunk.done_list[b] else f'{len(batch_chunk.audio_list[b])}samp'}"
                if batch_chunk.audio_list[b] is not None
                else f"b{b}:None"
                for b in range(len(texts))
            )
            print(f"  Chunk: {done_str}")

            if all(batch_chunk.done_list):
                break

        elapsed = time.perf_counter() - t0
        total_duration = 0
        sr = 22050

        for b in range(len(texts)):
            if per_sample_chunks[b]:
                full_audio = np.concatenate(per_sample_chunks[b])
                duration = len(full_audio) / sr
                total_duration += duration
                out_path = outputs[b] if b < len(outputs) else outputs[0].replace(".wav", f"_{b}.wav")
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                wav = torch.tensor(full_audio, dtype=torch.float32).unsqueeze(0).to(torch.int16)
                torchaudio.save(out_path, wav, sr)
                print(f"  [{b}] {duration:.2f}s -> {out_path}")

        print(f"\n  Total audio:       {total_duration:.2f}s")
        print(f"  Time-to-first:     {first_chunk_time:.3f}s")
        print(f"  Total time:        {elapsed:.2f}s")
        print(f"  RTF:               {elapsed / total_duration:.4f}")

    elif is_batch:
        # --- Batched inference ---
        print("=" * 60)
        print(f"Batched inference: {len(texts)} texts")
        print("=" * 60)
        t0 = time.perf_counter()

        results = pipeline.generate(
            text=texts, speaker=spk,
            emo_speaker=emo_spk, emo_alpha=args.emo_alpha,
            emo_vector=args.emo_vector,
            **gen_kwargs,
        )

        elapsed = time.perf_counter() - t0
        total_duration = 0

        for i, (sr, audio) in enumerate(results):
            duration = audio.shape[0] / sr
            total_duration += duration
            out_path = outputs[i] if i < len(outputs) else outputs[0].replace(".wav", f"_{i}.wav")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            wav = torch.tensor(audio.T, dtype=torch.int16)
            torchaudio.save(out_path, wav, sr)
            print(f"  [{i}] {duration:.2f}s -> {out_path}")

        print(f"\n  Total audio:  {total_duration:.2f}s")
        print(f"  Total time:   {elapsed:.2f}s")
        print(f"  RTF:          {elapsed / total_duration:.4f}")

    else:
        # --- Single non-streaming inference ---
        print("=" * 60)
        print(f"Inference: {texts[0]!r}")
        print("=" * 60)
        t0 = time.perf_counter()

        sr, audio = pipeline.generate(
            text=texts[0], speaker=spk,
            emo_speaker=emo_spk, emo_alpha=args.emo_alpha,
            emo_vector=args.emo_vector,
            output_path=outputs[0],
            **gen_kwargs,
        )

        elapsed = time.perf_counter() - t0
        duration = audio.shape[0] / sr

        print(f"\n  Duration:  {duration:.2f}s")
        print(f"  Time:      {elapsed:.2f}s")
        print(f"  RTF:       {elapsed / duration:.4f}")
        print(f"  Saved:     {outputs[0]}")


if __name__ == "__main__":
    main()
