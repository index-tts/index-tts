"""Streaming chunked inference."""

import time
from typing import Callable, List, Optional

import numpy as np

MEL_CODE_TO_FRAME_RATIO = 1.72
HOP_SIZE = 256
SAMPLE_RATE = 22050


def overlap_samples(n_codes: int) -> int:
    return int(n_codes * MEL_CODE_TO_FRAME_RATIO) * HOP_SIZE


def _get_crossfade_coeffs(n: int):
    hann_win = np.hanning(n * 2)
    return hann_win[:n], hann_win[n:]


def _crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    n = min(len(tail), len(head))
    if n == 0:
        return np.array([], dtype=np.float32)
    fade_in, fade_out = _get_crossfade_coeffs(n)
    return tail[:n] * fade_out + head[:n] * fade_in


def _to_int16(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio * 32767, -32767, 32767).astype(np.int16)


def _fade_out_tail(audio: np.ndarray, fade_samples: int = 512) -> np.ndarray:
    if len(audio) <= fade_samples:
        return audio
    fade = np.linspace(1.0, 0.0, fade_samples)
    audio = audio.copy()
    audio[-fade_samples:] *= fade
    return audio


class StreamingDecoder:
    """Streaming generation with per-sample cross-fading.

    Parameters
    ----------
    gpt_engine : GPTTRTEngine
        The TRT-LLM GPT engine that exposes ``generate_chunks``.
    codes_to_audio_fn : callable
        ``(codes, latent, code_lens, prompt_condition, ref_mel, style) -> list[np.ndarray]``
        Converts a batch of mel-code chunks into per-sample float32 audio arrays.
    chunk_size, overlap_size : int
        Code-level chunking parameters.
    """

    def __init__(
        self,
        gpt_engine,
        codes_to_audio_fn: Callable,
        chunk_size: int = 100,
        overlap_size: int = 20,
    ):
        self.gpt_engine = gpt_engine
        self.codes_to_audio_fn = codes_to_audio_fn
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.stride = chunk_size - overlap_size

    def generate(
        self,
        conds_latent,
        trt_text_ids,
        stop_token_id: int,
        prompt_condition,
        ref_mel,
        style,
        max_new_tokens: int = 1500,
        num_beams: int = 3,
        top_k: int = 30,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 10.0,
        audio_ws=None,
        text_lengths=None,
        prompt_lens=None,
    ):
        """Yield ``(sample_rate, audio_list, done_list)`` per chunk.

        ``audio_list[b]`` is an int16 numpy array (or None if sample *b*
        finished earlier).  ``done_list[b]`` is True when sample *b* just
        completed.
        """
        batch_size = conds_latent.shape[0]
        ovlp = overlap_samples(self.overlap_size)

        prev_tails: List[Optional[np.ndarray]] = [None] * batch_size
        sample_finished = [False] * batch_size
        chunk_idx = 0
        stream_start_time = time.perf_counter()
        first_chunk_latency = None
        total_audio_samples = 0

        for chunk_codes, chunk_latent, is_last, batch_done, chunk_code_lens in (
            self.gpt_engine.generate_chunks(
                conds_latents=conds_latent,
                text_input_ids=trt_text_ids,
                stop_token_id=stop_token_id,
                chunk_size=self.chunk_size,
                overlap_size=self.overlap_size,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                text_lengths=text_lengths,
            )
        ):
            _t = time.perf_counter()
            audios = self.codes_to_audio_fn(
                chunk_codes, chunk_latent, chunk_code_lens,
                prompt_condition, ref_mel, style,
                prompt_lens=prompt_lens,
            )
            chunk_time = time.perf_counter() - _t

            if first_chunk_latency is None:
                first_chunk_latency = time.perf_counter() - stream_start_time
                print(f">> [streaming] First chunk latency: {first_chunk_latency:.3f}s")

            print(
                f">> [streaming] chunk {chunk_idx}: {chunk_codes.shape[1]} codes -> "
                f"{len(audios[0])} samples ({chunk_time:.3f}s)"
            )

            audio_outputs: List[Optional[np.ndarray]] = [None] * batch_size
            done_outputs = [False] * batch_size

            for b in range(batch_size):
                if sample_finished[b]:
                    continue

                audio = audios[b]
                is_first_b = prev_tails[b] is None
                is_last_b = batch_done[b] or is_last

                if is_first_b:
                    if is_last_b:
                        audio_outputs[b] = _to_int16(_fade_out_tail(audio))
                        sample_finished[b] = True
                        done_outputs[b] = True
                    else:
                        split = len(audio) - ovlp
                        audio_outputs[b] = _to_int16(audio[:split])
                        prev_tails[b] = audio[split:]
                else:
                    head = audio[:ovlp]
                    rest = audio[ovlp:]
                    blended = _crossfade(prev_tails[b], head)

                    if is_last_b:
                        out = np.concatenate([blended, rest])
                        audio_outputs[b] = _to_int16(_fade_out_tail(out))
                        prev_tails[b] = None
                        sample_finished[b] = True
                        done_outputs[b] = True
                    else:
                        split = len(rest) - ovlp
                        out = np.concatenate([blended, rest[:split]])
                        audio_outputs[b] = _to_int16(out)
                        prev_tails[b] = rest[split:]

            if audio_outputs[0] is not None:
                total_audio_samples += len(audio_outputs[0])

            if audio_ws is not None and audio_outputs[0] is not None:
                audio_ws.push_chunk(audio_outputs[0])

            chunk_idx += 1
            yield (SAMPLE_RATE, audio_outputs, done_outputs)

        # Flush remaining tails
        has_remaining = any(
            prev_tails[b] is not None and not sample_finished[b]
            for b in range(batch_size)
        )
        if has_remaining:
            audio_outputs = [None] * batch_size
            done_outputs = [False] * batch_size
            for b in range(batch_size):
                if prev_tails[b] is not None and not sample_finished[b]:
                    audio_outputs[b] = _to_int16(_fade_out_tail(prev_tails[b]))
                    done_outputs[b] = True
                    sample_finished[b] = True
            if audio_ws is not None and audio_outputs[0] is not None:
                audio_ws.push_chunk(audio_outputs[0])
            yield (SAMPLE_RATE, audio_outputs, done_outputs)

        total_time = time.perf_counter() - stream_start_time
        audio_duration = total_audio_samples / SAMPLE_RATE
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        print(
            f">> [streaming] Total: {chunk_idx} chunks, {audio_duration:.2f}s audio, "
            f"{total_time:.2f}s wall time, RTF={rtf:.4f}"
        )
