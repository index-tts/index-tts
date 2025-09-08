import os
import re
import random
import torch
import torchaudio

MATPLOTLIB_FLAG = False


def load_audio(audiopath, sampling_rate):
    audio, sr = torchaudio.load(audiopath)
    #print(f"wave shape: {audio.shape}, sample_rate: {sr}")

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


def tokenize_by_CJK_char(line: str) -> str: 
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
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )    
    chars = pattern.split(line.strip().upper())
    return " ".join([w.strip() for w in chars if w.strip()])


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
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
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
