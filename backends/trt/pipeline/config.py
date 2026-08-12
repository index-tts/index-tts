"""Configuration loading for the standalone TRT pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf


@dataclass
class TTSConfig:
    """Flat configuration extracted from checkpoints/config.yaml."""

    # GPT token constants
    number_mel_codes: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    start_text_token: int = 0
    stop_text_token: int = 1
    model_dim: int = 1280

    # s2mel / mel spectrogram
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: int = 0
    fmax: Optional[int] = None

    # DiT
    in_channels: int = 80
    zero_prompt_speech_token: bool = False

    # Emotion
    emo_num: List[int] = field(default_factory=lambda: [3, 17, 2, 8, 4, 5, 10, 24])

    # File paths (relative to model_dir)
    bpe_model: str = "bpe.model"
    w2v_stat: str = "wav2vec2bert_stats.pt"
    emo_matrix: str = "feat2.pt"
    spk_matrix: str = "feat1.pt"


def load_config(config_path: str) -> TTSConfig:
    """Load TTSConfig from a checkpoints/config.yaml file."""
    raw = OmegaConf.load(config_path)

    fmax_raw = raw.s2mel["preprocess_params"]["spect_params"].get("fmax", "None")
    fmax = None if fmax_raw == "None" else int(fmax_raw)

    return TTSConfig(
        number_mel_codes=raw.gpt.number_mel_codes,
        start_mel_token=raw.gpt.start_mel_token,
        stop_mel_token=raw.gpt.stop_mel_token,
        start_text_token=raw.gpt.start_text_token,
        stop_text_token=raw.gpt.stop_text_token,
        model_dim=raw.gpt.model_dim,
        sample_rate=raw.s2mel["preprocess_params"]["sr"],
        n_fft=raw.s2mel["preprocess_params"]["spect_params"]["n_fft"],
        hop_length=raw.s2mel["preprocess_params"]["spect_params"]["hop_length"],
        win_length=raw.s2mel["preprocess_params"]["spect_params"]["win_length"],
        n_mels=raw.s2mel["preprocess_params"]["spect_params"]["n_mels"],
        fmin=raw.s2mel["preprocess_params"]["spect_params"].get("fmin", 0),
        fmax=fmax,
        in_channels=raw.s2mel.DiT.in_channels,
        zero_prompt_speech_token=raw.s2mel.DiT.get("zero_prompt_speech_token", False),
        emo_num=list(raw.emo_num),
        bpe_model=raw.dataset.bpe_model,
        w2v_stat=raw.w2v_stat,
        emo_matrix=raw.emo_matrix,
        spk_matrix=raw.spk_matrix,
    )
