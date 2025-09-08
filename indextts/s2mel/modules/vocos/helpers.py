import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback

matplotlib.use("Agg")


def save_figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    Save a matplotlib figure to a numpy array.

    Args:
        fig (Figure): Matplotlib figure object.

    Returns:
        ndarray: Numpy array representing the figure.
    """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    """
    Plot a spectrogram and convert it to a numpy array.

    Args:
        spectrogram (ndarray): Spectrogram data.

    Returns:
        ndarray: Numpy array representing the plotted spectrogram.
    """
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


class GradNormCallback(Callback):
    """
    Callback to log the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        model.log("grad_norm", gradient_norm(model))


def gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the gradient norm.

    Args:
        model (Module): PyTorch model.
        norm_type (float, optional): Type of the norm. Defaults to 2.0.

    Returns:
        Tensor: Gradient norm.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
    return total_norm
