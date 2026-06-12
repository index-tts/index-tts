import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_REQUIRED_MODEL_FILES = (
    "bpe.model",
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
)

_MODEL_REPO = "IndexTeam/IndexTTS"


def _cmd_download(args):
    """Download IndexTTS v1 model files."""
    model_dir = args.model_dir

    missing = [f for f in _REQUIRED_MODEL_FILES if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        print(f">> Downloading IndexTTS model to {model_dir}...")
        from indextts.utils.model_download import snapshot_download
        snapshot_download(_MODEL_REPO, local_dir=model_dir)

        still_missing = [f for f in _REQUIRED_MODEL_FILES if not os.path.exists(os.path.join(model_dir, f))]
        if still_missing:
            print(f"ERROR: Still missing after download: {', '.join(still_missing)}")
            sys.exit(1)
    else:
        print(f">> Main model files already present in {model_dir}.")

    from indextts.utils.model_download import ensure_config_available
    ensure_config_available(model_dir)

    print(f">> IndexTTS models downloaded successfully.")


def _cmd_infer(args):
    """Run TTS inference."""
    if len(args.text.strip()) == 0:
        print("ERROR: Text is empty.")
        sys.exit(1)
    if not os.path.exists(args.voice):
        print(f"Audio prompt file {args.voice} does not exist.")
        sys.exit(1)

    requested_config = args.config
    if not os.path.exists(requested_config):
        from indextts.utils.model_download import ensure_config_available
        config_dir = os.path.dirname(requested_config) or "."
        try:
            ensure_config_available(config_dir)
        except Exception as e:
            print(f"Failed to download config.yaml: {e}")
        downloaded_config = os.path.join(config_dir, "config.yaml")
        if os.path.exists(requested_config):
            args.config = requested_config
        elif os.path.exists(downloaded_config):
            print(f"Config file {requested_config} does not exist. Using {downloaded_config} instead.")
            args.config = downloaded_config
        else:
            print(f"Config file {requested_config} does not exist.")
            sys.exit(1)

    output_path = args.output_path
    if os.path.exists(output_path):
        if not args.force:
            print(f"ERROR: Output file {output_path} already exists. Use --force to overwrite.")
            sys.exit(1)
        else:
            os.remove(output_path)

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it first.")
        sys.exit(1)

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda:0"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            args.device = "xpu"
        elif hasattr(torch, "mps") and torch.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
            args.fp16 = False
            print("WARNING: Running on CPU may be slow.")

    from indextts.infer import IndexTTS
    tts = IndexTTS(cfg_path=args.config, model_dir=args.model_dir, use_fp16=args.fp16, device=args.device)
    tts.infer(audio_prompt=args.voice, text=args.text.strip(), output_path=output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS Command Line")
    subparsers = parser.add_subparsers(dest="command")

    # -- download subcommand --
    dl_parser = subparsers.add_parser("download", help="Download model files")
    dl_parser.add_argument("--model-dir", type=str, default="checkpoints", help="Model directory")

    # -- infer subcommand --
    infer_parser = subparsers.add_parser("infer", help="Run TTS inference")
    infer_parser.add_argument("text", type=str, help="Text to be synthesized")
    infer_parser.add_argument("-v", "--voice", type=str, required=True, help="Path to the audio prompt file")
    infer_parser.add_argument("-o", "--output_path", type=str, default="gen.wav", help="Path to the output wav file")
    infer_parser.add_argument("-c", "--config", type=str, default="checkpoints/config.yaml", help="Path to the config file")
    infer_parser.add_argument("--model-dir", type=str, default="checkpoints", help="Path to the model directory")
    infer_parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference")
    infer_parser.add_argument("-f", "--force", action="store_true", default=False, help="Overwrite output file if exists")
    infer_parser.add_argument("-d", "--device", type=str, default=None, help="Device (cpu, cuda, mps, xpu)")

    args = parser.parse_args()

    if args.command == "download":
        _cmd_download(args)
    elif args.command == "infer":
        _cmd_infer(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
