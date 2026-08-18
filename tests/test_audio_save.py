"""Regression tests for the WAV save path (index-tts/index-tts#724).

These are CPU-only and need no checkpoints, so they run in the `not gpu` CI job.

Background: torchaudio >= 2.9 delegates ``torchaudio.save()`` to TorchCodec, whose
compatibility shim converts non-float32 input with a bare ``src.float()`` -- no
rescaling -- and then treats the result as ``[-1, 1]`` audio. Handing it the
PCM-scale int16 tensor IndexTTS used to build clips nearly every frame to full
scale, silently, with no exception or warning.

Run with:
    uv run --extra test pytest tests/test_audio_save.py -v
"""
import struct
import warnings

import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from indextts.utils.common import PCM16_MAX, save_pcm_wav  # noqa: E402

SAMPLE_RATE = 24000


def _reference_waveform():
    """A normalized [-1, 1] stand-in for vocoder output, peaking at 0.8."""
    t = torch.arange(SAMPLE_RATE, dtype=torch.float32) / SAMPLE_RATE
    wave = (
        0.72 * torch.sin(2 * torch.pi * 440.0 * t)
        + 0.08 * torch.sin(2 * torch.pi * 997.0 * t)
    ).clamp(-0.8, 0.8)
    return wave.unsqueeze(0).contiguous()


def _to_pcm_scale(normalized):
    """Exactly the scaling the inference code applies before saving."""
    return torch.clamp(PCM16_MAX * normalized, -PCM16_MAX, PCM16_MAX)


def _read_wav(path):
    """Parse a RIFF/WAVE file with no audio library, so the test can't be fooled
    by the same normalization bug on the read side."""
    blob = path.read_bytes()
    assert blob[:4] == b"RIFF" and blob[8:12] == b"WAVE", "not a RIFF/WAVE file"

    fmt = data = None
    pos = 12
    while pos + 8 <= len(blob):
        chunk_id = blob[pos : pos + 4]
        size = struct.unpack("<I", blob[pos + 4 : pos + 8])[0]
        body = blob[pos + 8 : pos + 8 + size]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            data = body
        pos += 8 + size + (size & 1)  # chunks are word-aligned
    assert fmt is not None and data is not None, "missing fmt / data chunk"

    audio_format, channels, rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    if audio_format == 0xFFFE and len(fmt) >= 26:  # WAVE_FORMAT_EXTENSIBLE
        audio_format = struct.unpack("<H", fmt[24:26])[0]

    return {
        "format": audio_format,  # 1 = integer PCM, 3 = IEEE float
        "bits": bits,
        "channels": channels,
        "sample_rate": rate,
        "samples": torch.frombuffer(bytearray(data), dtype=torch.int16),
    }


# -- the actual regression: no saturation on any torchaudio version -------------


def test_save_pcm_wav_round_trips_without_saturation(tmp_path):
    normalized = _reference_waveform()
    out = tmp_path / "out.wav"

    save_pcm_wav(str(out), _to_pcm_scale(normalized), SAMPLE_RATE)

    wav = _read_wav(out)
    assert wav["sample_rate"] == SAMPLE_RATE
    assert wav["channels"] == 1

    samples = wav["samples"]
    assert samples.numel() == normalized.shape[-1]

    # The reference peaks at 0.8, so nothing may land on a rail. Before the fix
    # this was ~99.99% of frames on torchaudio >= 2.9.
    at_full_scale = int(((samples <= -32768) | (samples >= 32767)).sum())
    assert at_full_scale == 0, f"{at_full_scale}/{samples.numel()} samples clipped to full scale"

    # Saturation also collapses the waveform onto a handful of levels.
    assert len(set(samples.tolist())) > 1000, "amplitude information was destroyed"

    # And the peak must still reflect the 0.8 input, not full scale.
    peak = int(samples.abs().max())
    assert 0.75 * 32767 < peak < 0.85 * 32767, f"unexpected peak {peak}"


def test_save_pcm_wav_preserves_the_waveform_shape(tmp_path):
    normalized = _reference_waveform()
    out = tmp_path / "out.wav"

    save_pcm_wav(str(out), _to_pcm_scale(normalized), SAMPLE_RATE)

    decoded = _read_wav(out)["samples"].to(torch.float32) / 32768.0
    expected = normalized[0]

    # 16-bit quantization error is bounded by one LSB.
    assert torch.allclose(decoded, expected, atol=2.0 / 32768.0)


# -- keep the historical container format --------------------------------------


def test_save_pcm_wav_writes_16_bit_integer_pcm(tmp_path):
    """torchaudio < 2.9 picks the WAV subtype from the input dtype, so handing it
    float32 without pinning the encoding would silently emit a 32-bit float WAV."""
    out = tmp_path / "out.wav"
    save_pcm_wav(str(out), _to_pcm_scale(_reference_waveform()), SAMPLE_RATE)

    wav = _read_wav(out)
    assert wav["format"] == 1, "expected integer PCM, got IEEE float"
    assert wav["bits"] == 16, f"expected 16-bit samples, got {wav['bits']}"


def test_save_pcm_wav_does_not_warn_about_ignored_encoding_args(tmp_path):
    """torchaudio >= 2.9 warns when encoding/bits_per_sample are supplied, so they
    must only be passed on versions that honour them.

    Matched narrowly on TorchCodec's "The '<arg>' parameter is not ... supported"
    wording; torchaudio 2.8 raises an unrelated blanket deprecation notice that
    also happens to name these arguments, and that one is not ours to silence.
    """
    out = tmp_path / "out.wav"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_pcm_wav(str(out), _to_pcm_scale(_reference_waveform()), SAMPLE_RATE)

    offending = [
        str(w.message)
        for w in caught
        if str(w.message).startswith("The '") and "parameter is not" in str(w.message)
    ]
    assert not offending, f"torchaudio ignored our encoding arguments: {offending}"


# -- what is handed to torchaudio.save(): version-independent negative control --


def test_save_pcm_wav_hands_normalized_float32_to_torchaudio(tmp_path, monkeypatch):
    """The core contract, asserted without depending on the installed torchaudio.

    TorchCodec requires float32 in [-1, 1]; anything else is silently clipped.
    """
    captured = {}

    def spy(uri, src, sample_rate, **kwargs):
        captured["src"] = src.clone()
        captured["kwargs"] = kwargs

    monkeypatch.setattr(torchaudio, "save", spy)
    save_pcm_wav(str(tmp_path / "out.wav"), _to_pcm_scale(_reference_waveform()), SAMPLE_RATE)

    src = captured["src"]
    assert src.dtype == torch.float32, f"torchaudio.save() got {src.dtype}, not float32"
    assert float(src.abs().max()) <= 1.0, f"samples out of [-1, 1]: peak {float(src.abs().max())}"
    assert float(src.abs().max()) == pytest.approx(0.8, abs=1e-3), "amplitude was not preserved"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int16, torch.int32])
def test_save_pcm_wav_normalizes_every_pcm_scale_dtype(tmp_path, monkeypatch, dtype):
    """Call sites pass PCM-scale data as float32 *and* as int16; both must normalize."""
    captured = {}
    monkeypatch.setattr(torchaudio, "save", lambda uri, src, sr, **kw: captured.update(src=src.clone()))

    save_pcm_wav(str(tmp_path / "out.wav"), _to_pcm_scale(_reference_waveform()).to(dtype), SAMPLE_RATE)

    src = captured["src"]
    assert src.dtype == torch.float32
    assert float(src.abs().max()) == pytest.approx(0.8, abs=1e-3)


def test_save_pcm_wav_does_not_mutate_the_caller_tensor(tmp_path):
    """The inference code reuses `wav` after saving, so normalization must copy."""
    pcm = _to_pcm_scale(_reference_waveform())
    before = pcm.clone()

    save_pcm_wav(str(tmp_path / "out.wav"), pcm, SAMPLE_RATE)

    assert torch.equal(pcm, before), "save_pcm_wav modified its input in place"
