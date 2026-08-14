import os
import random
import re

import torch
import torchaudio

MATPLOTLIB_FLAG = False

# Full-scale value used everywhere we convert between normalized waveforms and
# 16-bit PCM. 32767 rather than 32768 so that the positive peak cannot overflow.
PCM16_MAX = 32767.0


def _torchaudio_honors_wav_encoding_args():
    """Whether ``torchaudio.save()`` still respects ``encoding``/``bits_per_sample``.

    torchaudio < 2.9 derives the WAV subtype from the input dtype, so float32 input
    silently produces a 32-bit float WAV. We pass the arguments there to keep the
    16-bit PCM output IndexTTS has always written.

    torchaudio >= 2.9 delegates encoding to TorchCodec, which ignores both arguments
    and warns when they are supplied. Its WAV default is already 16-bit signed PCM,
    so we omit them instead of spamming a warning on every save.

    An unparseable version is treated as the newer path: omitting the arguments is
    harmless there, whereas passing them to TorchCodec always warns.
    """
    raw = getattr(torchaudio, "__version__", "") or ""
    parts = []
    for chunk in raw.split("+")[0].split(".")[:2]:
        if not chunk.isdigit():
            return False
        parts.append(int(chunk))
    return len(parts) == 2 and tuple(parts) < (2, 9)


def save_pcm_wav(path, wav, sampling_rate):
    """Write a **PCM-scale** waveform to ``path`` as a 16-bit PCM WAV file.

    ``wav`` holds values in ``[-32767, 32767]`` and may be either an integer tensor
    or the float tensor produced by ``torch.clamp(PCM16_MAX * wav, ...)``. It is
    normalized to ``[-1, 1]`` float32 here, because that is the only input range
    ``torchaudio.save()`` interprets identically across versions:

    * torchaudio <= 2.8 rescales integer input by dtype range when encoding.
    * torchaudio >= 2.9 routes through TorchCodec's ``AudioEncoder``, whose
      compatibility shim does a bare ``src.float()`` with no rescaling and then
      treats the result as ``[-1, 1]`` audio. Feeding it PCM-scale samples clips
      almost every frame to full scale -- no exception, no warning, just a
      saturated file (see index-tts/index-tts#724).

    Do not pass an already-normalized waveform; it would be attenuated by 32767.
    """
    wav = wav.detach().to(device="cpu", dtype=torch.float32) / PCM16_MAX
    wav = wav.clamp_(-1.0, 1.0)
    encoding_args = {"encoding": "PCM_S", "bits_per_sample": 16} if _torchaudio_honors_wav_encoding_args() else {}
    torchaudio.save(path, wav, sampling_rate, **encoding_args)


def load_audio(audiopath, sampling_rate):
    audio, sr = torchaudio.load(audiopath)
    # print(f"wave shape: {audio.shape}, sample_rate: {sr}")

    if audio.size(0) > 1:  # mix to mono
        audio = audio[0].unsqueeze(0)

    if sr != sampling_rate:
        try:
            audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        except Exception as e:
            print(f"Warning: {audiopath}, wave shape: {audio.shape}, sample_rate: {sr}")
            return None
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


def tokenize_by_CJK_char(line: str, do_upper_case=True) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return charaters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    CJK_RANGE_PATTERN = (
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = re.split(CJK_RANGE_PATTERN, line.strip())
    return " ".join([w.strip().upper() if do_upper_case else w.strip() for w in chars if w.strip()])


def de_tokenized_by_CJK_char(line: str, do_lower_case=False) -> str:
    """
    Example:
      input = "你 好 世 界 是 HELLO WORLD 的 中 文"
      output = "你好世界是 hello world 的中文"

    do_lower_case:
      input = "SEE YOU!"
      output = "see you!"
    """
    # replace english words in the line with placeholders
    english_word_pattern = re.compile(r"([A-Z]+(?:[\s'-][A-Z-]+)*)", re.IGNORECASE)
    english_sents = english_word_pattern.findall(line)
    for i, sent in enumerate(english_sents):
        line = line.replace(sent, f"<sent_{i}>")

    words = line.split()
    # restore english sentences
    sent_placeholder_pattern = re.compile(r"(<sent_(\d+)>)")
    for i in range(len(words)):
        all_matches = sent_placeholder_pattern.findall(words[i])
        if len(all_matches) > 1:
            # restore the english word
            for h,j in all_matches:
                placeholder_index = int(j)
                words[i] = words[i].replace(h, english_sents[placeholder_index])
                if do_lower_case:
                    words[i] = words[i].lower()
    return "".join(words)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))
