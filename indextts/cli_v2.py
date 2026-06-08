import argparse
import contextlib
import importlib
import io
import json
import math
import os
import sys
import tempfile
import wave
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


class ConcatFileError(ValueError):
    def __init__(self, message, exit_code):
        super().__init__(message)
        self.exit_code = exit_code


class ConcatExecutionError(RuntimeError):
    def __init__(self, message, cleanup_error=None):
        super().__init__(message)
        self.cleanup_error = cleanup_error


def main(argv=None, tts_factory=None, stdin=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _run_check(args)
    if args.command == "synth":
        return _run_synth(args, tts_factory=tts_factory, stdin=stdin)
    if args.command == "batch":
        return _run_batch(args, tts_factory=tts_factory)
    if args.command == "concat":
        return _run_concat(args)

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
    batch.add_argument(
        "--output-dir",
        help="Directory for automatically named independent WAV outputs",
    )
    batch.add_argument(
        "--output-prefix",
        help="Filename prefix for automatically named independent WAV outputs",
    )
    batch.add_argument("--concat", action="store_true", help="Generate one concatenated batch output")
    batch.add_argument("--output", help="Path to write concatenated batch WAV audio")
    batch.add_argument("--keep-temp", action="store_true", help="Keep internal batch concat temporary files")
    batch.add_argument("--device", default=None, help="Runtime device")
    batch.add_argument("--fp16", action="store_true", help="Use FP16 inference")
    batch.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed")
    batch.add_argument("--cuda-kernel", action="store_true", help="Use CUDA kernel")
    batch.add_argument("--verbose", action="store_true", help="Show verbose inference output")
    batch.add_argument("--voice", help="Default speaker reference audio for every batch task")
    batch.add_argument("--emotion-audio", help="Default emotion reference audio for every batch task")
    batch.add_argument("--emotion-text", help="Default emotion description text for every batch task")
    batch.add_argument("--emotion-vector", help="Default comma-separated 8-dimensional emotion vector")
    batch.add_argument(
        "--emotion-weight",
        default="1.0",
        help="Default emotion weight mapped to IndexTTS2 emo_alpha",
    )
    concat = subparsers.add_parser(
        "concat",
        help="Validate an audio concat file",
    )
    concat.add_argument(
        "--concat-file",
        required=True,
        help="Path to the JSON Lines concat file",
    )
    concat.add_argument("--output", required=True, help="Path to write concatenated WAV audio")
    concat.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    concat.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the concat file without creating output audio",
    )
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
        defaults = _validate_batch_defaults(args)
        output_config = _validate_batch_output_config(args)
        tasks = _load_batch_tasks(
            Path(args.batch_file),
            force=args.force,
            defaults=defaults,
            output_config=output_config,
        )
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
    if tts_factory is None:
        try:
            tts_factory = _load_indextts2()
        except (ImportError, OSError) as exc:
            print(f"ERROR: runtime unavailable: {exc}", file=sys.stderr)
            return EXIT_RUNTIME_UNAVAILABLE
    verbose = getattr(args, "verbose", False)
    try:
        with _synth_stdout_context(verbose):
            tts = tts_factory(
                cfg_path=str(model_dir / "config.yaml"),
                model_dir=args.model_dir,
                use_fp16=getattr(args, "fp16", False),
                device=getattr(args, "device", None),
                use_cuda_kernel=getattr(args, "cuda_kernel", False),
                use_deepspeed=getattr(args, "deepspeed", False),
            )
    except Exception as exc:
        print(f"ERROR: inference failed: {exc}", file=sys.stderr)
        return EXIT_INFERENCE_ERROR

    for task in tasks:
        output_path = task["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _synth_stdout_context(verbose):
                infer_kwargs = {
                    "spk_audio_prompt": str(task["voice_path"]),
                    "text": task["text"],
                    "output_path": str(output_path),
                    "verbose": verbose,
                }
                infer_kwargs.update(task["emotion_kwargs"])
                tts.infer(**infer_kwargs)
        except Exception as exc:
            print(f"ERROR: batch file line {task['line_number']} inference failed: {exc}", file=sys.stderr)
            return EXIT_INFERENCE_ERROR
        print(f"Generated: {output_path}")
    print(f"Batch complete: {len(tasks)} tasks generated")
    return EXIT_SUCCESS


def _run_concat(args):
    try:
        output_path = _resolve_command_path(args.output)
        segments = _load_concat_segments(
            _resolve_command_path(args.concat_file),
            output_path,
            force=args.force,
        )
    except ConcatFileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    if not args.dry_run:
        try:
            _concatenate_wav_segments(segments, output_path)
        except ConcatExecutionError as exc:
            print(f"ERROR: concat failed: {exc}", file=sys.stderr)
            if exc.cleanup_error is not None:
                print(f"WARNING: cleanup failed: {exc.cleanup_error}", file=sys.stderr)
            return EXIT_INFERENCE_ERROR
        print(f"Generated: {output_path}")
        return EXIT_SUCCESS
    print(f"Concat file OK: {len(segments)} segments")
    return EXIT_SUCCESS


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


def _validate_batch_defaults(args):
    emotion_conflict_error = _emotion_conflict_error(args)
    if emotion_conflict_error is not None:
        raise BatchFileError(_strip_error_prefix(emotion_conflict_error), EXIT_INPUT_ERROR)

    try:
        emotion_weight = _parse_emotion_weight(args.emotion_weight, "--emotion-weight")
    except InputValidationError as exc:
        raise BatchFileError(str(exc), EXIT_INPUT_ERROR) from exc

    voice_path = None
    if args.voice is not None:
        voice_path = Path(args.voice)
        if not voice_path.is_file():
            raise BatchFileError(f"voice reference audio does not exist: {voice_path}", EXIT_MISSING_RESOURCE)

    emotion_source = None
    if args.emotion_audio is not None:
        emotion_path = Path(args.emotion_audio)
        if not emotion_path.is_file():
            raise BatchFileError(
                f"emotion reference audio does not exist: {emotion_path}",
                EXIT_MISSING_RESOURCE,
            )
        emotion_source = ("emotion_audio", emotion_path)
    elif args.emotion_text is not None:
        if not args.emotion_text.strip():
            raise BatchFileError("--emotion-text must not be empty", EXIT_INPUT_ERROR)
        emotion_source = ("emotion_text", args.emotion_text)
    elif args.emotion_vector is not None:
        try:
            emotion_source = ("emotion_vector", _parse_emotion_vector(args.emotion_vector))
        except InputValidationError as exc:
            raise BatchFileError(str(exc), EXIT_INPUT_ERROR) from exc

    return {
        "voice_path": voice_path,
        "emotion_source": emotion_source,
        "emotion_weight": emotion_weight,
    }


def _validate_batch_output_config(args):
    if args.concat:
        if args.output_dir is not None:
            raise BatchFileError("--concat cannot be used with --output-dir", EXIT_INPUT_ERROR)
        if args.output_prefix is not None:
            raise BatchFileError("--concat cannot be used with --output-prefix", EXIT_INPUT_ERROR)
        raise BatchFileError("batch --concat is not implemented yet", EXIT_INPUT_ERROR)
    if args.output is not None:
        raise BatchFileError("--output is only valid with --concat", EXIT_INPUT_ERROR)
    if args.keep_temp:
        raise BatchFileError("--keep-temp requires --concat", EXIT_INPUT_ERROR)
    if args.output_prefix is not None and args.output_dir is None:
        raise BatchFileError("--output-prefix requires --output-dir", EXIT_INPUT_ERROR)
    if args.output_prefix is not None:
        _validate_batch_output_prefix(args.output_prefix)
    if args.output_dir is None:
        return {"mode": "row"}
    return {
        "mode": "auto",
        "output_dir": _resolve_command_path(args.output_dir),
        "output_prefix": args.output_prefix,
    }


def _validate_batch_output_prefix(output_prefix):
    if "/" in output_prefix or "\\" in output_prefix:
        raise BatchFileError("--output-prefix must not contain path separators", EXIT_INPUT_ERROR)
    prefix_path = Path(output_prefix)
    if prefix_path.suffix:
        raise BatchFileError("--output-prefix must not include a file extension", EXIT_INPUT_ERROR)
    if not output_prefix.strip():
        raise BatchFileError("--output-prefix must not be empty", EXIT_INPUT_ERROR)


def _strip_error_prefix(message):
    prefix = "ERROR: "
    if message.startswith(prefix):
        return message[len(prefix) :]
    return message


def _load_batch_tasks(batch_file, force=False, defaults=None, output_config=None):
    if not batch_file.is_file():
        raise BatchFileError(f"batch file does not exist: {batch_file}", EXIT_MISSING_RESOURCE)

    if defaults is None:
        defaults = {"voice_path": None, "emotion_source": None, "emotion_weight": 1.0}
    if output_config is None:
        output_config = {"mode": "row"}
    batch_dir = batch_file.parent
    tasks = []
    outputs = {}
    allowed_fields = {
        "output",
        "text",
        "text_file",
        "voice",
        "emotion_audio",
        "emotion_text",
        "emotion_vector",
        "emotion_weight",
    }
    for line_number, raw_line in enumerate(batch_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        text_path = None
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

        if "voice" in task:
            voice_path = _resolve_batch_path(batch_dir, _require_batch_string(task, "voice", line_number))
        else:
            voice_path = defaults["voice_path"]
        if voice_path is None:
            raise BatchFileError(f"batch file line {line_number} missing required field: voice", EXIT_INPUT_ERROR)
        if not voice_path.is_file():
            raise BatchFileError(
                f"batch file line {line_number} voice reference audio does not exist: {voice_path}",
                EXIT_MISSING_RESOURCE,
            )

        emotion_kwargs = _batch_emotion_kwargs(task, batch_dir, line_number, defaults)
        output_path = _batch_task_output_path(
            task,
            batch_dir,
            line_number,
            len(tasks) + 1,
            output_config,
        )
        if output_config["mode"] == "auto":
            _reject_batch_auto_output_input_conflicts(
                output_path,
                line_number,
                _batch_task_protected_input_paths(batch_file, text_path, voice_path, emotion_kwargs),
            )
            _reject_batch_auto_output_parent_conflicts(output_path)
        output_key = str(output_path.resolve(strict=False))
        if output_key in outputs:
            raise BatchFileError(
                f"batch file line {line_number} has duplicate output path: {output_path}",
                EXIT_INPUT_ERROR,
            )
        outputs[output_key] = line_number
        if output_path.exists() and not force:
            raise BatchFileError(
                f"batch file line {line_number} output file already exists: {output_path}",
                EXIT_INPUT_ERROR,
            )
        tasks.append(
            {
                "line_number": line_number,
                "text": text,
                "voice_path": voice_path,
                "output_path": output_path,
                "emotion_kwargs": emotion_kwargs,
            }
        )
    return tasks


def _batch_task_protected_input_paths(batch_file, text_path, voice_path, emotion_kwargs):
    protected_paths = [batch_file, voice_path]
    if text_path is not None:
        protected_paths.append(text_path)
    emotion_path = emotion_kwargs.get("emo_audio_prompt")
    if emotion_path is not None:
        protected_paths.append(Path(emotion_path))
    return protected_paths


def _reject_batch_auto_output_input_conflicts(output_path, line_number, protected_paths):
    output_key = _normalized_path_key(output_path)
    for protected_path in protected_paths:
        if output_key == _normalized_path_key(protected_path):
            raise BatchFileError(
                f"batch file line {line_number} generated output conflicts with protected input path: {protected_path}",
                EXIT_INPUT_ERROR,
            )


def _reject_batch_auto_output_parent_conflicts(output_path):
    parent = output_path.parent
    existing_parent = parent
    while not existing_parent.exists():
        if existing_parent.parent == existing_parent:
            break
        existing_parent = existing_parent.parent
    if existing_parent.exists() and not existing_parent.is_dir():
        raise BatchFileError(
            f"output parent path cannot be created because a file exists: {existing_parent}",
            EXIT_INPUT_ERROR,
        )


def _batch_task_output_path(task, batch_dir, line_number, task_number, output_config):
    output_value = task.get("output")
    if output_config["mode"] == "row":
        if output_value is None:
            raise BatchFileError(f"batch file line {line_number} missing required field: output", EXIT_INPUT_ERROR)
        return _resolve_batch_path(batch_dir, _require_batch_string(task, "output", line_number))

    if output_value is not None:
        raise BatchFileError(
            f"batch file line {line_number} field 'output' is not allowed with --output-dir",
            EXIT_INPUT_ERROR,
        )
    return output_config["output_dir"] / _auto_batch_output_name(task_number, output_config["output_prefix"])


def _auto_batch_output_name(task_number, output_prefix):
    stem = f"{task_number:04d}"
    if output_prefix:
        stem = f"{output_prefix}-{stem}"
    return f"{stem}.wav"


def _batch_emotion_kwargs(task, batch_dir, line_number, defaults):
    row_source_fields = [
        field_name for field_name in ("emotion_audio", "emotion_text", "emotion_vector") if field_name in task
    ]
    if len(row_source_fields) > 1:
        raise BatchFileError(
            f"batch file line {line_number} emotion_audio, emotion_text and emotion_vector are mutually exclusive",
            EXIT_INPUT_ERROR,
        )

    if "emotion_weight" in task:
        try:
            emotion_weight = _parse_emotion_weight(
                task["emotion_weight"],
                f"batch file line {line_number} field 'emotion_weight'",
            )
        except InputValidationError as exc:
            raise BatchFileError(str(exc), EXIT_INPUT_ERROR) from exc
    else:
        emotion_weight = defaults["emotion_weight"]

    if row_source_fields:
        source = _parse_batch_emotion_source(task, row_source_fields[0], batch_dir, line_number)
    else:
        source = defaults["emotion_source"]

    if source is None:
        if "emotion_weight" in task:
            raise BatchFileError(
                f"batch file line {line_number} field 'emotion_weight' requires an emotion source",
                EXIT_INPUT_ERROR,
            )
        return {}

    source_name, source_value = source
    if source_name == "emotion_audio":
        return {"emo_audio_prompt": str(source_value), "emo_alpha": emotion_weight}
    if source_name == "emotion_text":
        return {"use_emo_text": True, "emo_text": source_value, "emo_alpha": emotion_weight}
    return {"emo_vector": source_value, "emo_alpha": emotion_weight}


def _parse_batch_emotion_source(task, field_name, batch_dir, line_number):
    if field_name == "emotion_audio":
        emotion_path = _resolve_batch_path(batch_dir, _require_batch_string(task, field_name, line_number))
        if not emotion_path.is_file():
            raise BatchFileError(
                f"batch file line {line_number} emotion reference audio does not exist: {emotion_path}",
                EXIT_MISSING_RESOURCE,
            )
        return ("emotion_audio", emotion_path)
    if field_name == "emotion_text":
        emotion_text = _require_batch_string(task, field_name, line_number)
        return ("emotion_text", emotion_text)
    try:
        emotion_vector = _parse_emotion_vector(
            task[field_name],
            f"batch file line {line_number} field 'emotion_vector'",
        )
    except InputValidationError as exc:
        raise BatchFileError(str(exc), EXIT_INPUT_ERROR) from exc
    return ("emotion_vector", emotion_vector)


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


def _load_concat_segments(concat_file, output_path, force=False):
    if not concat_file.is_file():
        raise ConcatFileError(f"concat file does not exist: {concat_file}", EXIT_MISSING_RESOURCE)
    if _normalized_path_key(output_path) == _normalized_path_key(concat_file):
        raise ConcatFileError("--output must not be the same path as --concat-file", EXIT_INPUT_ERROR)
    if not _has_wav_extension(output_path):
        raise ConcatFileError(f"--output must be a .wav file: {output_path}", EXIT_INPUT_ERROR)
    _reject_concat_output_parent_conflicts(output_path)

    concat_dir = concat_file.parent
    segments = []
    expected_format = None
    expected_format_line = None
    allowed_fields = {"audio", "silence_after_ms"}
    for line_number, raw_line in enumerate(concat_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            segment = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ConcatFileError(
                f"concat file line {line_number} is not valid JSON: {exc.msg}",
                EXIT_INPUT_ERROR,
            ) from exc
        if not isinstance(segment, dict):
            raise ConcatFileError(
                f"concat file line {line_number} must be a JSON object",
                EXIT_INPUT_ERROR,
            )
        unknown_fields = sorted(set(segment) - allowed_fields)
        if unknown_fields:
            unknown = ", ".join(unknown_fields)
            raise ConcatFileError(
                f"concat file line {line_number} has unknown fields: {unknown}",
                EXIT_INPUT_ERROR,
            )
        audio_path = _resolve_concat_audio_path(concat_dir, _require_concat_string(segment, "audio", line_number))
        if not _has_wav_extension(audio_path):
            raise ConcatFileError(
                f"concat file line {line_number} field 'audio' must be a .wav file: {audio_path}",
                EXIT_INPUT_ERROR,
            )
        silence_after_ms = _parse_concat_silence_after_ms(segment, line_number)
        audio_format = _read_concat_wav_format(audio_path, line_number)
        if expected_format is None:
            expected_format = audio_format
            expected_format_line = line_number
        elif audio_format != expected_format:
            raise ConcatFileError(
                f"concat file line {line_number} WAV format does not match baseline line {expected_format_line}",
                EXIT_INPUT_ERROR,
            )
        segments.append(
            {
                "line_number": line_number,
                "audio_path": audio_path,
                "silence_after_ms": silence_after_ms,
                "format": audio_format,
            }
        )
    if not segments:
        raise ConcatFileError("concat file must contain at least one segment", EXIT_INPUT_ERROR)
    _reject_concat_input_conflicts(output_path, segments)
    _reject_concat_output_file_conflicts(output_path, force=force)
    return segments


def _concatenate_wav_segments(segments, output_path):
    temp_path = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = _create_concat_temp_path(output_path)
        _write_concat_wav(temp_path, segments)
        os.replace(temp_path, output_path)
        temp_path = None
    except Exception as exc:
        cleanup_error = None
        if temp_path is not None:
            cleanup_error = _cleanup_concat_temp_file(temp_path)
        raise ConcatExecutionError(str(exc), cleanup_error=cleanup_error) from exc


def _create_concat_temp_path(output_path):
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".wav",
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        return Path(temp_file.name)


def _write_concat_wav(temp_path, segments):
    frame_rate, channels, sample_width = segments[0]["format"]
    with wave.open(str(temp_path), "wb") as output_wav:
        output_wav.setnchannels(channels)
        output_wav.setsampwidth(sample_width)
        output_wav.setframerate(frame_rate)
        for segment in segments:
            with wave.open(str(segment["audio_path"]), "rb") as input_wav:
                output_wav.writeframes(input_wav.readframes(input_wav.getnframes()))
            silence_frames = frame_rate * segment["silence_after_ms"] // 1000
            if silence_frames:
                output_wav.writeframes(b"\0" * channels * sample_width * silence_frames)


def _cleanup_concat_temp_file(temp_path):
    try:
        temp_path.unlink(missing_ok=True)
    except OSError as exc:
        return exc
    return None


def _resolve_command_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _resolve_concat_audio_path(concat_dir, path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = concat_dir / path
    return path


def _has_wav_extension(path):
    return path.suffix.lower() == ".wav"


def _normalized_path_key(path):
    return str(path.resolve(strict=False)).casefold()


def _reject_concat_output_parent_conflicts(output_path):
    parent = output_path.parent
    existing_parent = parent
    while not existing_parent.exists():
        if existing_parent.parent == existing_parent:
            break
        existing_parent = existing_parent.parent
    if existing_parent.exists() and not existing_parent.is_dir():
        raise ConcatFileError(
            f"output parent path cannot be created because a file exists: {existing_parent}",
            EXIT_INPUT_ERROR,
        )


def _reject_concat_output_file_conflicts(output_path, force=False):
    if output_path.exists() and not force:
        raise ConcatFileError(f"output file already exists: {output_path}", EXIT_INPUT_ERROR)


def _reject_concat_input_conflicts(output_path, segments):
    output_key = _normalized_path_key(output_path)
    for segment in segments:
        if output_key == _normalized_path_key(segment["audio_path"]):
            raise ConcatFileError(
                f"concat file line {segment['line_number']} audio conflicts with --output: {segment['audio_path']}",
                EXIT_INPUT_ERROR,
            )


def _require_concat_string(segment, field_name, line_number):
    if field_name not in segment:
        raise ConcatFileError(f"concat file line {line_number} missing required field: {field_name}", EXIT_INPUT_ERROR)
    value = segment[field_name]
    if not isinstance(value, str):
        raise ConcatFileError(
            f"concat file line {line_number} field '{field_name}' must be a string",
            EXIT_INPUT_ERROR,
        )
    if not value.strip():
        raise ConcatFileError(
            f"concat file line {line_number} field '{field_name}' must not be empty",
            EXIT_INPUT_ERROR,
        )
    return value


def _parse_concat_silence_after_ms(segment, line_number):
    if "silence_after_ms" not in segment:
        return 0
    value = segment["silence_after_ms"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConcatFileError(
            f"concat file line {line_number} field 'silence_after_ms' must be a non-negative integer",
            EXIT_INPUT_ERROR,
        )
    if value < 0:
        raise ConcatFileError(
            f"concat file line {line_number} field 'silence_after_ms' must be a non-negative integer",
            EXIT_INPUT_ERROR,
        )
    return value


def _read_concat_wav_format(audio_path, line_number):
    if not audio_path.is_file():
        raise ConcatFileError(
            f"concat file line {line_number} audio file does not exist: {audio_path}",
            EXIT_MISSING_RESOURCE,
        )
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            audio_format = (
                wav_file.getframerate(),
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
            )
            frame_count = wav_file.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        raise ConcatFileError(
            f"concat file line {line_number} audio file is not a readable WAV: {audio_path}",
            EXIT_INPUT_ERROR,
        ) from exc
    if frame_count <= 0:
        raise ConcatFileError(
            f"concat file line {line_number} audio file is empty: {audio_path}",
            EXIT_INPUT_ERROR,
        )
    return audio_format


def _parse_emotion_vector(value, label="--emotion-vector"):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise InputValidationError(f"{label} must not be empty")
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if not value.strip():
            raise InputValidationError(f"{label} must not be empty")
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        if not value:
            raise InputValidationError(f"{label} must not be empty")
        if any(isinstance(part, bool) for part in value):
            raise InputValidationError(f"{label} entries must be numeric")
        parts = value
    else:
        raise InputValidationError(f"{label} must be a string or JSON array")
    try:
        emotion_vector = [float(part) for part in parts]
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{label} entries must be numeric") from exc
    if len(emotion_vector) != 8:
        raise InputValidationError(f"{label} must contain exactly 8 values; got {len(emotion_vector)}")
    out_of_range = [item for item in emotion_vector if not math.isfinite(item) or item < 0.0 or item > 1.0]
    if out_of_range:
        raise InputValidationError(f"{label} values must be between 0.0 and 1.0")
    vector_sum = sum(emotion_vector)
    if vector_sum > 0.8:
        raise InputValidationError(f"{label} sum must be <= 0.8; got {vector_sum:g}")
    return emotion_vector


def _parse_emotion_weight(value, label):
    if isinstance(value, bool):
        raise InputValidationError(f"{label} must be a float: {value}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{label} must be a float: {value}") from exc


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
