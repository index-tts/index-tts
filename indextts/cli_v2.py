import argparse
import contextlib
import importlib
import io
import json
import math
import sys
from pathlib import Path


EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_MISSING_RESOURCE = 2
EXIT_RUNTIME_UNAVAILABLE = 3
EXIT_INFERENCE_ERROR = 4

REQUIRED_MODEL_FILES = (
    "config.yaml",
    "bpe.model",
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
)
REQUIRED_PACKAGES = ("torch", "torchaudio", "indextts")


class InputValidationError(ValueError):
    pass


class BatchFileError(ValueError):
    def __init__(self, message, exit_code):
        super().__init__(message)
        self.exit_code = exit_code


def main(argv=None, tts_factory=None, stdin=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _run_check(args)
    if args.command == "synth":
        return _run_synth(args, tts_factory=tts_factory, stdin=stdin)
    if args.command == "batch":
        return _run_batch(args, tts_factory=tts_factory)

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
    batch = subparsers.add_parser(
        "batch",
        help="Validate a batch file and run batch synthesis",
    )
    batch.add_argument(
        "--batch-file",
        required=True,
        help="Path to the JSON Lines batch file",
    )
    batch.add_argument(
        "--model-dir",
        default="checkpoints",
        help="Path to the IndexTTS2 model directory",
    )
    batch.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the batch file without loading model weights",
    )
    batch.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    synth = subparsers.add_parser(
        "synth",
        help="Synthesize one text input with IndexTTS2",
    )
    synth.add_argument("--text", help="Text to synthesize")
    synth.add_argument("--text-file", help="UTF-8 text file to synthesize")
    synth.add_argument("--stdin", action="store_true", help="Read text from standard input")
    synth.add_argument("--voice", help="Path to the speaker reference audio")
    synth.add_argument("--emotion-audio", help="Path to the emotion reference audio")
    synth.add_argument("--emotion-text", help="Emotion description text")
    synth.add_argument("--emotion-vector", help="Comma-separated 8-dimensional emotion vector")
    synth.add_argument(
        "--emotion-weight",
        default="1.0",
        help="Emotion weight mapped to IndexTTS2 emo_alpha",
    )
    synth.add_argument("--output", help="Path to write generated audio")
    synth.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    synth.add_argument(
        "--model-dir",
        default="checkpoints",
        help="Path to the IndexTTS2 model directory",
    )
    synth.add_argument("--device", default=None, help="Runtime device")
    synth.add_argument("--fp16", action="store_true", help="Use FP16 inference")
    synth.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed")
    synth.add_argument("--cuda-kernel", action="store_true", help="Use CUDA kernel")
    synth.add_argument("--verbose", action="store_true", help="Show verbose inference output")
    return parser


def _run_synth(args, tts_factory=None, stdin=None):
    if _text_source_count(args) != 1:
        print("ERROR: provide exactly one text source: --text, --text-file or --stdin", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if args.text_file and not Path(args.text_file).is_file():
        print(f"ERROR: text file does not exist: {args.text_file}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    text = _read_synth_text(args, stdin)
    if not text:
        print("ERROR: text is empty", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not args.voice:
        print("ERROR: --voice is required", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    voice_path = Path(args.voice)
    if not voice_path.is_file():
        print(f"ERROR: voice reference audio does not exist: {voice_path}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    emotion_conflict_error = _emotion_conflict_error(args)
    if emotion_conflict_error is not None:
        print(emotion_conflict_error, file=sys.stderr)
        return EXIT_INPUT_ERROR
    emotion_vector = None
    if args.emotion_vector is not None:
        try:
            emotion_vector = _parse_emotion_vector(args.emotion_vector)
        except InputValidationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
    if args.emotion_text is not None and not args.emotion_text.strip():
        print("ERROR: --emotion-text must not be empty", file=sys.stderr)
        return EXIT_INPUT_ERROR
    emotion_path = Path(args.emotion_audio) if args.emotion_audio is not None else None
    if emotion_path is not None and not emotion_path.is_file():
        print(f"ERROR: emotion reference audio does not exist: {emotion_path}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    try:
        emotion_weight = float(args.emotion_weight)
    except ValueError:
        print(f"ERROR: --emotion-weight must be a float: {args.emotion_weight}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not args.output:
        print("ERROR: --output is required", file=sys.stderr)
        return EXIT_INPUT_ERROR
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"ERROR: output file already exists: {output_path}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    model_dir = Path(args.model_dir)
    missing_files = _missing_model_files(model_dir)
    if missing_files is None:
        print(f"ERROR: model directory does not exist: {model_dir}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    if missing_files:
        missing = ", ".join(missing_files)
        print(f"ERROR: missing required model files: {missing}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tts_factory is None:
        try:
            tts_factory = _load_indextts2()
        except (ImportError, OSError) as exc:
            print(f"ERROR: runtime unavailable: {exc}", file=sys.stderr)
            return EXIT_RUNTIME_UNAVAILABLE
    try:
        with _synth_stdout_context(args.verbose):
            tts = tts_factory(
                cfg_path=str(model_dir / "config.yaml"),
                model_dir=args.model_dir,
                use_fp16=args.fp16,
                device=args.device,
                use_cuda_kernel=args.cuda_kernel,
                use_deepspeed=args.deepspeed,
            )
            infer_kwargs = {
                "spk_audio_prompt": str(voice_path),
                "text": text,
                "output_path": str(output_path),
                "verbose": args.verbose,
            }
            if emotion_path is not None:
                infer_kwargs["emo_audio_prompt"] = str(emotion_path)
                infer_kwargs["emo_alpha"] = emotion_weight
            if args.emotion_text is not None:
                infer_kwargs["use_emo_text"] = True
                infer_kwargs["emo_text"] = args.emotion_text
                infer_kwargs["emo_alpha"] = emotion_weight
            if emotion_vector is not None:
                infer_kwargs["emo_vector"] = emotion_vector
                infer_kwargs["emo_alpha"] = emotion_weight
            tts.infer(
                **infer_kwargs,
            )
    except Exception as exc:
        print(f"ERROR: inference failed: {exc}", file=sys.stderr)
        return EXIT_INFERENCE_ERROR
    print(f"Generated: {output_path}")
    return EXIT_SUCCESS


def _run_batch(args, tts_factory=None):
    try:
        tasks = _load_batch_tasks(Path(args.batch_file))
    except BatchFileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    model_dir = Path(args.model_dir)
    missing_files = _missing_model_files(model_dir)
    if missing_files is None:
        print(f"ERROR: model directory does not exist: {model_dir}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    if missing_files:
        missing = ", ".join(missing_files)
        print(f"ERROR: missing required model files: {missing}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    if args.dry_run:
        print(f"Batch file OK: {len(tasks)} tasks")
        return EXIT_SUCCESS
    print("ERROR: batch execution is not implemented yet; use --dry-run", file=sys.stderr)
    return EXIT_INPUT_ERROR


def _text_source_count(args):
    return sum((args.text is not None, args.text_file is not None, args.stdin))


def _emotion_source_count(args):
    return sum(
        (
            args.emotion_audio is not None,
            args.emotion_text is not None,
            args.emotion_vector is not None,
        )
    )


def _emotion_conflict_error(args):
    if _emotion_source_count(args) <= 1:
        return None
    if args.emotion_vector is None and args.emotion_audio is not None and args.emotion_text is not None:
        return "ERROR: --emotion-audio and --emotion-text are mutually exclusive"
    return "ERROR: --emotion-vector, --emotion-audio and --emotion-text are mutually exclusive"


def _read_synth_text(args, stdin):
    if args.stdin:
        source = sys.stdin if stdin is None else stdin
        return source.read().strip()
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return args.text.strip()


def _load_batch_tasks(batch_file):
    if not batch_file.is_file():
        raise BatchFileError(f"batch file does not exist: {batch_file}", EXIT_MISSING_RESOURCE)

    batch_dir = batch_file.parent
    tasks = []
    outputs = {}
    allowed_fields = {"output", "text", "text_file", "voice"}
    for line_number, raw_line in enumerate(batch_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            task = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise BatchFileError(f"batch file line {line_number} is not valid JSON: {exc.msg}", EXIT_INPUT_ERROR) from exc
        if not isinstance(task, dict):
            raise BatchFileError(
                f"batch file line {line_number} must be a JSON object",
                EXIT_INPUT_ERROR,
            )
        unknown_fields = sorted(set(task) - allowed_fields)
        if unknown_fields:
            unknown = ", ".join(unknown_fields)
            raise BatchFileError(
                f"batch file line {line_number} has unknown fields: {unknown}",
                EXIT_INPUT_ERROR,
            )

        text_source_count = sum(key in task for key in ("text", "text_file"))
        if text_source_count != 1:
            raise BatchFileError(
                f"batch file line {line_number} must provide exactly one text source: text or text_file",
                EXIT_INPUT_ERROR,
            )
        if "text" in task:
            if not isinstance(task["text"], str):
                raise BatchFileError(
                    f"batch file line {line_number} field 'text' must be a string",
                    EXIT_INPUT_ERROR,
                )
            text = task["text"].strip()
            if not text:
                raise BatchFileError(f"batch file line {line_number} text is empty", EXIT_INPUT_ERROR)
        else:
            text_file = _require_batch_string(task, "text_file", line_number)
            text_path = _resolve_batch_path(batch_dir, text_file)
            if not text_path.is_file():
                raise BatchFileError(
                    f"batch file line {line_number} text file does not exist: {text_path}",
                    EXIT_MISSING_RESOURCE,
                )
            text = text_path.read_text(encoding="utf-8").strip()
            if not text:
                raise BatchFileError(f"batch file line {line_number} text is empty", EXIT_INPUT_ERROR)

        voice_value = task.get("voice")
        if voice_value is None:
            raise BatchFileError(f"batch file line {line_number} missing required field: voice", EXIT_INPUT_ERROR)
        voice_path = _resolve_batch_path(batch_dir, _require_batch_string(task, "voice", line_number))
        if not voice_path.is_file():
            raise BatchFileError(
                f"batch file line {line_number} voice reference audio does not exist: {voice_path}",
                EXIT_MISSING_RESOURCE,
            )

        output_value = task.get("output")
        if output_value is None:
            raise BatchFileError(f"batch file line {line_number} missing required field: output", EXIT_INPUT_ERROR)
        output_path = _resolve_batch_path(batch_dir, _require_batch_string(task, "output", line_number))
        output_key = str(output_path.resolve(strict=False))
        if output_key in outputs:
            raise BatchFileError(
                f"batch file line {line_number} has duplicate output path: {output_path}",
                EXIT_INPUT_ERROR,
            )
        outputs[output_key] = line_number
        tasks.append(
            {
                "line_number": line_number,
                "text": text,
                "voice_path": voice_path,
                "output_path": output_path,
            }
        )
    return tasks


def _require_batch_string(task, field_name, line_number):
    value = task[field_name]
    if not isinstance(value, str):
        raise BatchFileError(
            f"batch file line {line_number} field '{field_name}' must be a string",
            EXIT_INPUT_ERROR,
        )
    if not value.strip():
        raise BatchFileError(
            f"batch file line {line_number} field '{field_name}' must not be empty",
            EXIT_INPUT_ERROR,
        )
    return value


def _resolve_batch_path(batch_dir, path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = batch_dir / path
    return path


def _parse_emotion_vector(value):
    value = value.strip()
    if not value:
        raise InputValidationError("--emotion-vector must not be empty")
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value.strip():
        raise InputValidationError("--emotion-vector must not be empty")
    parts = [part.strip() for part in value.split(",")]
    try:
        emotion_vector = [float(part) for part in parts]
    except ValueError as exc:
        raise InputValidationError("--emotion-vector entries must be numeric") from exc
    if len(emotion_vector) != 8:
        raise InputValidationError(f"--emotion-vector must contain exactly 8 values; got {len(emotion_vector)}")
    out_of_range = [item for item in emotion_vector if not math.isfinite(item) or item < 0.0 or item > 1.0]
    if out_of_range:
        raise InputValidationError("--emotion-vector values must be between 0.0 and 1.0")
    vector_sum = sum(emotion_vector)
    if vector_sum > 0.8:
        raise InputValidationError(f"--emotion-vector sum must be <= 0.8; got {vector_sum:g}")
    return emotion_vector


def _load_indextts2():
    from indextts.infer_v2 import IndexTTS2

    return IndexTTS2


def _synth_stdout_context(verbose):
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


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
