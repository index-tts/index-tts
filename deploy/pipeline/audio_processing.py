"""Audio preprocessing utilities for the standalone TRT pipeline."""

import librosa
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn

_mel_basis = {}
_hann_window = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 80,
    sampling_rate: int = 22050,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: int = 0,
    fmax=None,
    center: bool = False,
) -> torch.Tensor:
    """Compute log-mel spectrogram. Matches indextts/s2mel/modules/audio.py exactly."""
    global _mel_basis, _hann_window

    key = f"{sampling_rate}_{fmax}_{y.device}"
    if key not in _mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        _hann_window[f"{sampling_rate}_{y.device}"] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y, n_fft,
            hop_length=hop_size, win_length=win_size,
            window=_hann_window[f"{sampling_rate}_{y.device}"],
            center=center, pad_mode="reflect",
            normalized=False, onesided=True, return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_mel_basis[key], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec


def load_and_resample(path: str, max_seconds: float = 15.0):
    """Load audio, truncate, return (waveform_tensor, native_sr).

    Returns:
        audio: (1, num_samples) float32 tensor on CPU
        sr: native sample rate
    """
    audio, sr = librosa.load(path, sr=None)
    max_samples = int(max_seconds * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    return audio, sr


def resample(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a (1, T) waveform tensor."""
    if orig_sr == target_sr:
        return audio
    return torchaudio.transforms.Resample(orig_sr, target_sr)(audio)


def compute_kaldi_fbank(audio_16k: torch.Tensor, device=None) -> torch.Tensor:
    """Compute 80-bin Kaldi fbank features with per-utterance mean removal.

    Args:
        audio_16k: (1, T) waveform at 16 kHz

    Returns:
        feat: (1, T_frames, 80)
    """
    if device is not None:
        audio_16k = audio_16k.to(device)
    feat = torchaudio.compliance.kaldi.fbank(
        audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000,
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    return feat.unsqueeze(0)


def compute_semantic_features(audio_16k: torch.Tensor, feature_extractor, device=None):
    """Run HuggingFace SeamlessM4TFeatureExtractor on 16 kHz audio.

    Returns:
        input_features: (B, T, F) on device
        attention_mask: (B, T) on device
    """
    inputs = feature_extractor(
        audio_16k.squeeze(0).numpy() if audio_16k.dim() > 1 else audio_16k.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs["input_features"]
    attention_mask = inputs["attention_mask"]
    if device is not None:
        input_features = input_features.to(device)
        attention_mask = attention_mask.to(device)
    return input_features, attention_mask
