from pathlib import Path

import argbind
from audiotools import ml

import indextts.s2mel.dac as dac

DAC = dac.model.DAC
Accelerator = ml.Accelerator

__MODEL_LATEST_TAGS__ = {
    ("44khz", "8kbps"): "0.0.1",
    ("24khz", "8kbps"): "0.0.4",
    ("16khz", "8kbps"): "0.0.5",
    ("44khz", "16kbps"): "1.0.0",
}

__MODEL_URLS__ = {
    (
        "44khz",
        "0.0.1",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
    (
        "24khz",
        "0.0.4",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth",
    (
        "16khz",
        "0.0.5",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth",
    (
        "44khz",
        "1.0.0",
        "16kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps.pth",
}


@argbind.bind(group="download", positional=True, without_prefix=True)
def download(
    model_type: str = "44khz", model_bitrate: str = "8kbps", tag: str = "latest"
):
    """
    Function that downloads the weights file from URL if a local cache is not found.

    Parameters
    ----------
    model_type : str
        The type of model to download. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz".
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
        Only 44khz model supports 16kbps.
    tag : str
        The tag of the model to download. Defaults to "latest".

    Returns
    -------
    Path
        Directory path required to load model via audiotools.
    """
    model_type = model_type.lower()
    tag = tag.lower()

    assert model_type in [
        "44khz",
        "24khz",
        "16khz",
    ], "model_type must be one of '44khz', '24khz', or '16khz'"

    assert model_bitrate in [
        "8kbps",
        "16kbps",
    ], "model_bitrate must be one of '8kbps', or '16kbps'"

    if tag == "latest":
        tag = __MODEL_LATEST_TAGS__[(model_type, model_bitrate)]

    download_link = __MODEL_URLS__.get((model_type, tag, model_bitrate), None)

    if download_link is None:
        raise ValueError(
            f"Could not find model with tag {tag} and model type {model_type}"
        )

    local_path = (
        Path.home()
        / ".cache"
        / "descript"
        / "dac"
        / f"weights_{model_type}_{model_bitrate}_{tag}.pth"
    )
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        import requests

        response = requests.get(download_link)

        if response.status_code != 200:
            raise ValueError(
                f"Could not download model. Received response code {response.status_code}"
            )
        local_path.write_bytes(response.content)

    return local_path


def load_model(
    model_type: str = "44khz",
    model_bitrate: str = "8kbps",
    tag: str = "latest",
    load_path: str = None,
):
    if not load_path:
        load_path = download(
            model_type=model_type, model_bitrate=model_bitrate, tag=tag
        )
    generator = DAC.load(load_path)
    return generator
