"""Triton client for FasterIndexTTS2.

Supports both non-streaming and streaming modes.
Only requires: tritonclient[grpc], numpy, soundfile

Install:
    pip install tritonclient[grpc] soundfile numpy

Usage:
    # Non-streaming
    python triton_client.py --mode non-streaming \
        --text "Hello world." --speaker_audio voice.wav --output out.wav

    # Streaming
    python triton_client.py --mode streaming \
        --text "Hello world." --speaker_audio voice.wav --output out.wav
"""

import argparse
import time
import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient


def load_audio_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def run_non_streaming(args):
    client = grpcclient.InferenceServerClient(url=args.url)

    speaker_bytes = load_audio_bytes(args.speaker_audio)

    inputs = [
        grpcclient.InferInput("text", [1, 1], "BYTES"),
        grpcclient.InferInput("speaker_audio", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(np.array([[args.text.encode("utf-8")]], dtype=object))
    inputs[1].set_data_from_numpy(np.array([[speaker_bytes]], dtype=object))

    outputs = [
        grpcclient.InferRequestedOutput("audio"),
        grpcclient.InferRequestedOutput("sample_rate"),
        grpcclient.InferRequestedOutput("audio_length"),
    ]

    print(f"Sending non-streaming request: text={args.text!r}")
    t0 = time.perf_counter()

    result = client.infer(model_name="indextts2", inputs=inputs, outputs=outputs)

    elapsed = time.perf_counter() - t0
    audio = result.as_numpy("audio")[0]
    sr = result.as_numpy("sample_rate")[0][0]
    audio_length = result.as_numpy("audio_length")[0][0]
    audio = audio[:audio_length]

    print(f"Received: {len(audio)} samples, sr={sr}, dur={len(audio)/sr:.2f}s, elapsed={elapsed:.2f}s")
    sf.write(args.output, audio, sr, subtype="PCM_16")
    print(f"Saved to {args.output}")


def run_streaming(args):
    import queue

    result_queue = queue.Queue()

    def callback(result, error):
        result_queue.put((result, error))

    client = grpcclient.InferenceServerClient(url=args.url)

    speaker_bytes = load_audio_bytes(args.speaker_audio)

    inputs = [
        grpcclient.InferInput("text", [1, 1], "BYTES"),
        grpcclient.InferInput("speaker_audio", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(np.array([[args.text.encode("utf-8")]], dtype=object))
    inputs[1].set_data_from_numpy(np.array([[speaker_bytes]], dtype=object))

    outputs = [
        grpcclient.InferRequestedOutput("audio_chunk"),
        grpcclient.InferRequestedOutput("is_last"),
        grpcclient.InferRequestedOutput("sample_rate"),
    ]

    print(f"Sending streaming request: text={args.text!r}")
    t0 = time.perf_counter()

    client.start_stream(callback=callback)
    client.async_stream_infer(model_name="indextts2_stream", inputs=inputs, outputs=outputs)

    all_chunks = []
    chunk_count = 0
    first_chunk_time = None

    while True:
        result, error = result_queue.get()
        if error:
            print(f"Error: {error}")
            break

        chunk_count += 1
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - t0

        audio_chunk = result.as_numpy("audio_chunk")[0]
        is_last = result.as_numpy("is_last")[0][0]

        nonzero = np.nonzero(audio_chunk)[0]
        if len(nonzero) > 0:
            audio_chunk = audio_chunk[:nonzero[-1] + 1]
            all_chunks.append(audio_chunk)

        print(f"  Chunk {chunk_count}: {len(audio_chunk)} samples, is_last={is_last}")

        if is_last:
            break

    client.stop_stream()
    elapsed = time.perf_counter() - t0

    if all_chunks:
        full_audio = np.concatenate(all_chunks)
        sr = 22050
        print(f"Total: {chunk_count} chunks, {len(full_audio)/sr:.2f}s audio, "
              f"first_chunk={first_chunk_time:.3f}s, total={elapsed:.2f}s")
        sf.write(args.output, full_audio, sr, subtype="PCM_16")
        print(f"Saved to {args.output}")
    else:
        print("No audio received")


def main():
    parser = argparse.ArgumentParser(description="FasterIndexTTS2 Triton Client")
    parser.add_argument("--mode", choices=["streaming", "non-streaming"], default="non-streaming")
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC endpoint (host:port)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--speaker_audio", required=True, help="Path to reference speaker audio")
    parser.add_argument("--output", default="triton_output.wav", help="Output WAV file path")

    args = parser.parse_args()

    if args.mode == "non-streaming":
        run_non_streaming(args)
    else:
        run_streaming(args)


if __name__ == "__main__":
    main()
