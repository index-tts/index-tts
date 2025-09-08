import warnings
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from tqdm import tqdm

from dac import DACFile
from dac.utils import load_model

warnings.filterwarnings("ignore", category=UserWarning)


@argbind.bind(group="decode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def decode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    device: str = "cuda",
    model_type: str = "44khz",
    verbose: bool = False,
):
    """Decode audio from codes.

    Parameters
    ----------
    input : str
        Path to input directory or file
    output : str, optional
        Path to output directory, by default "".
        If `input` is a directory, the directory sub-tree relative to `input` is re-created in `output`.
    weights_path : str, optional
        Path to weights file, by default "". If not specified, the weights file will be downloaded from the internet using the
        model_tag and model_type.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if `weights_path` is specified.
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
    device : str, optional
        Device to use, by default "cuda". If "cpu", the model will be loaded on the CPU.
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

    # Find all .dac files in input directory
    _input = Path(input)
    input_files = list(_input.glob("**/*.dac"))

    # If input is a .dac file, add it to the list
    if _input.suffix == ".dac":
        input_files.append(_input)

    # Create output directory
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(input_files)), desc=f"Decoding files"):
        # Load file
        artifact = DACFile.load(input_files[i])

        # Reconstruct audio from codes
        recons = generator.decompress(artifact, verbose=verbose)

        # Compute output path
        relative_path = input_files[i].relative_to(input)
        output_dir = output / relative_path.parent
        if not relative_path.name:
            output_dir = output
            relative_path = input_files[i]
        output_name = relative_path.with_suffix(".wav").name
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        recons.write(output_path)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        decode()
