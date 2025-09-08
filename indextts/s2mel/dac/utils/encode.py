import math
import warnings
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm

from dac.utils import load_model

warnings.filterwarnings("ignore", category=UserWarning)


@argbind.bind(group="encode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    n_quantizers: int = None,
    device: str = "cuda",
    model_type: str = "44khz",
    win_duration: float = 5.0,
    verbose: bool = False,
):
    """Encode audio files in input path to .dac format.

    Parameters
    ----------
    input : str
        Path to input audio file or directory
    output : str, optional
        Path to output directory, by default "". If `input` is a directory, the directory sub-tree relative to `input` is re-created in `output`.
    weights_path : str, optional
        Path to weights file, by default "". If not specified, the weights file will be downloaded from the internet using the
        model_tag and model_type.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if `weights_path` is specified.
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
    n_quantizers : int, optional
        Number of quantizers to use, by default None. If not specified, all the quantizers will be used and the model will compress at maximum bitrate.
    device : str, optional
        Device to use, by default "cuda"
    model_type : str, optional
        The type of model to use. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz". Ignored if `weights_path` is specified.
    """
    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()
    kwargs = {"n_quantizers": n_quantizers}

    # Find all audio files in input path
    input = Path(input)
    audio_files = util.find_audio(input)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(audio_files)), desc="Encoding files"):
        # Load file
        signal = AudioSignal(audio_files[i])

        # Encode audio to .dac format
        artifact = generator.compress(signal, win_duration, verbose=verbose, **kwargs)

        # Compute output path
        relative_path = audio_files[i].relative_to(input)
        output_dir = output / relative_path.parent
        if not relative_path.name:
            output_dir = output
            relative_path = audio_files[i]
        output_name = relative_path.with_suffix(".dac").name
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        artifact.save(output_path)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()
