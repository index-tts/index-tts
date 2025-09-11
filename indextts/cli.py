import os
import sys
import warnings
# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
def main():
    import argparse
    parser = argparse.ArgumentParser(description="IndexTTS Command Line")
    parser.add_argument("text", type=str, help="Text to be synthesized")
    parser.add_argument("-v", "--voice", type=str, required=True, help="Path to the audio prompt file (wav format)")
    parser.add_argument("-o", "--output_path", type=str, default="gen.wav", help="Path to the output wav file")
    parser.add_argument("-c", "--config", type=str, default="checkpoints/config.yaml", help="Path to the config file. Default is 'checkpoints/config.yaml'")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Path to the model directory. Default is 'checkpoints'")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
    parser.add_argument("-f", "--force", action="store_true", default=False, help="Force to overwrite the output file if it exists")
    parser.add_argument("-d", "--device", type=str, default=None, help="Device to run the model on (cpu, cuda, mps, xpu)." )
    args = parser.parse_args()
    if len(args.text.strip()) == 0:
        print("ERROR: Text is empty.")
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.voice):
        print(f"Audio prompt file {args.voice} does not exist.")
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        parser.print_help()
        sys.exit(1)

    output_path = args.output_path
    if os.path.exists(output_path):
        if not args.force:
            print(f"ERROR: Output file {output_path} already exists. Use --force to overwrite.")
            parser.print_help()
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
            args.fp16 = False # Disable FP16 on CPU
            print("WARNING: Running on CPU may be slow.")

    # TODO: Add CLI support for IndexTTS2.
    from indextts.infer import IndexTTS
    tts = IndexTTS(cfg_path=args.config, model_dir=args.model_dir, use_fp16=args.fp16, device=args.device)
    tts.infer(audio_prompt=args.voice, text=args.text.strip(), output_path=output_path)

if __name__ == "__main__":
    main()