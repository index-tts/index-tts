import torch


def show_cuda_gpu_list() -> None:
    """
    Displays a list of all detected GPUs that support the CUDA Torch APIs.
    """

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs found: {num_gpus}")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f'GPU {i}: "{gpu_name}"')


def check_torch_gpus() -> None:
    """
    Checks for the availability of various PyTorch GPU acceleration platforms
    and prints information about the discovered GPUs.
    """

    # Check for AMD ROCm/HIP first, since it modifies the CUDA APIs.
    # NOTE: The unofficial ROCm/HIP backend exposes the AMD features through
    # the CUDA Torch API calls.
    if hasattr(torch.backends, "hip") and torch.backends.hip.is_available():
        print("PyTorch: AMD ROCm/HIP is available!")
        show_cuda_gpu_list()

    # Check for NVIDIA CUDA.
    elif torch.cuda.is_available():
        print("PyTorch: NVIDIA CUDA is available!")
        show_cuda_gpu_list()

    # Check for Apple Metal Performance Shaders (MPS).
    elif torch.backends.mps.is_available():
        print("PyTorch: Apple MPS is available!")
        # PyTorch with MPS doesn't have a direct equivalent of `device_count()`
        # or `get_device_name()` for now, so we just confirm its presence.
        print("Using Apple Silicon GPU.")

    else:
        print("PyTorch: No GPU acceleration detected. Running in CPU mode.")


if __name__ == "__main__":
    check_torch_gpus()
