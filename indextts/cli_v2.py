import argparse
import importlib
import sys
from pathlib import Path


EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_MISSING_RESOURCE = 2
EXIT_RUNTIME_UNAVAILABLE = 3

REQUIRED_MODEL_FILES = (
    "config.yaml",
    "bpe.model",
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
)
REQUIRED_PACKAGES = ("torch", "torchaudio", "indextts")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _run_check(args)

    parser.print_help(sys.stderr)
    return EXIT_INPUT_ERROR


def _build_parser():
    parser = argparse.ArgumentParser(prog="indextts2", description="IndexTTS2 command line")
    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser(
        "check",
        help="Check local IndexTTS2 prerequisites without loading model weights",
    )
    check.add_argument(
        "--model-dir",
        default="checkpoints",
        help="Path to the IndexTTS2 model directory",
    )
    check.add_argument(
        "--device",
        default=None,
        help="Required runtime device, e.g. cpu, cuda, cuda:0, mps or xpu",
    )
    return parser


def _run_check(args):
    model_dir = Path(args.model_dir)
    missing_files = _missing_model_files(model_dir)
    if missing_files is None:
        print(f"ERROR: model directory does not exist: {model_dir}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    if missing_files:
        missing = ", ".join(missing_files)
        print(f"ERROR: missing required model files: {missing}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE

    imports = _import_required_packages()
    if imports.missing:
        missing = ", ".join(imports.missing)
        print(f"ERROR: missing required Python packages: {missing}", file=sys.stderr)
        return EXIT_RUNTIME_UNAVAILABLE

    devices = _detect_devices(imports.torch)
    if args.device and not _is_requested_device_available(imports.torch, devices, args.device):
        print(f"ERROR: requested device is not available: {args.device}", file=sys.stderr)
        return EXIT_RUNTIME_UNAVAILABLE

    print(f"OK: model directory {model_dir}")
    print("OK: required model files")
    print("OK: python packages")
    for device in ("cuda", "xpu", "mps", "cpu"):
        status = "available" if devices[device] else "unavailable"
        print(f"{device}: {status}")
    return EXIT_SUCCESS


def _missing_model_files(model_dir):
    if not model_dir.is_dir():
        return None
    return [filename for filename in REQUIRED_MODEL_FILES if not (model_dir / filename).is_file()]


def _import_required_packages():
    missing = []
    imported = {}
    for package in REQUIRED_PACKAGES:
        try:
            imported[package] = importlib.import_module(package)
        except (ImportError, OSError):
            missing.append(package)
    return argparse.Namespace(missing=missing, torch=imported.get("torch"))


def _detect_devices(torch_module):
    return {
        "cuda": _is_available(torch_module, "cuda"),
        "xpu": _is_available(torch_module, "xpu"),
        "mps": _is_mps_available(torch_module),
        "cpu": True,
    }


def _is_available(torch_module, name):
    device_backend = getattr(torch_module, name, None)
    is_available = getattr(device_backend, "is_available", None)
    return bool(is_available and is_available())


def _is_mps_available(torch_module):
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return bool(is_available and is_available())


def _is_requested_device_available(torch_module, devices, device):
    family = _device_family(device)
    if not devices.get(family, False):
        return False
    if family in {"cuda", "xpu"}:
        index = _device_index(device, family)
        if index is None:
            return True
        return _indexed_device_available(torch_module, family, index)
    return device == family


def _device_index(device, family):
    if device == family:
        return None
    prefix = f"{family}:"
    if not device.startswith(prefix):
        return -1
    try:
        return int(device[len(prefix) :])
    except ValueError:
        return -1


def _indexed_device_available(torch_module, family, index):
    if index < 0:
        return False
    device_backend = getattr(torch_module, family, None)
    device_count = getattr(device_backend, "device_count", None)
    if device_count is None:
        return False
    return index < device_count()


def _device_family(device):
    if device == "cuda" or device.startswith("cuda:"):
        return "cuda"
    if device == "xpu" or device.startswith("xpu:"):
        return "xpu"
    if device == "mps":
        return "mps"
    if device == "cpu":
        return "cpu"
    return device


if __name__ == "__main__":
    raise SystemExit(main())
