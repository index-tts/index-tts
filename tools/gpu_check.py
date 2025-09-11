import torch


def show_device_list(backend: str) -> int:
    """
    Displays a list of all detected devices for a given PyTorch backend.

    Args:
        backend: The name of the device backend module (e.g., "cuda", "xpu").

    Returns:
        The number of devices found if the backend is usable, otherwise 0.
    """

    backend_upper = backend.upper()

    try:
        # Get the backend module from PyTorch, e.g., `torch.cuda`.
        # NOTE: Backends always exist even if the user has no devices.
        backend_module = getattr(torch, backend)

        # Determine which vendor brand name to display.
        brand_name = backend_upper
        if backend == "cuda":
            # NOTE: This also checks for PyTorch's official AMD ROCm support,
            # since that's implemented inside the PyTorch CUDA APIs.
            # SEE: https://docs.pytorch.org/docs/stable/cuda.html
            brand_name = "NVIDIA CUDA / AMD ROCm"
        elif backend == "xpu":
            brand_name = "Intel XPU"
        elif backend == "mps":
            brand_name = "Apple MPS"

        if not backend_module.is_available():
            print(f"PyTorch: No devices found for {brand_name} backend.")
            return 0

        print(f"PyTorch: {brand_name} is available!")

        # Show all available hardware acceleration devices.
        device_count = backend_module.device_count()
        print(f"  * Number of {backend_upper} devices found: {device_count}")

        # NOTE: Apple Silicon devices don't have `get_device_name()` at the
        # moment, so we'll skip those since we can't get their device names.
        # SEE: https://docs.pytorch.org/docs/stable/mps.html
        if backend != "mps":
            for i in range(device_count):
                device_name = backend_module.get_device_name(i)
                print(f'  * Device {i}: "{device_name}"')

        return device_count

    except AttributeError:
        print(
            f'Error: The PyTorch backend "{backend}" does not exist, or is missing the necessary APIs (is_available, device_count, get_device_name).'
        )
    except Exception as e:
        print(f"Error: {e}")

    return 0


def check_torch_devices() -> None:
    """
    Checks for the availability of various PyTorch hardware acceleration
    platforms and prints information about the discovered devices.
    """

    print("Scanning for PyTorch hardware acceleration devices...\n")

    device_count = 0

    device_count += show_device_list("cuda")  # NVIDIA CUDA / AMD ROCm.
    device_count += show_device_list("xpu")  # Intel XPU.
    device_count += show_device_list("mps")  # Apple Metal Performance Shaders (MPS).

    if device_count > 0:
        print("\nHardware acceleration detected. Your system is ready!")
    else:
        print("\nNo hardware acceleration detected. Running in CPU mode.")


if __name__ == "__main__":
    check_torch_devices()
