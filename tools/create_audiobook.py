import argparse
import os
import sys
import time

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from indextts.infer_v2 import IndexTTS2

def main():
    parser = argparse.ArgumentParser(
        description="Audiobook Creation Suite using IndexTTS2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text-file", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--voice-file", type=str, required=True, help="Path to the reference voice audio file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the generated audiobook.")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory.")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available.")
    parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available.")
    parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode.")
    parser.add_argument("--max-text-tokens", type=int, default=120, help="Max tokens per generation segment.")

    args = parser.parse_args()

    print("--- Audiobook Creation Suite ---")
    print(f"Text file: {args.text_file}")
    print(f"Voice file: {args.voice_file}")
    print(f"Output file: {args.output_file}")
    print("---------------------------------")

    # Check if input files exist
    if not os.path.exists(args.text_file):
        print(f"Error: Text file not found at '{args.text_file}'")
        sys.exit(1)
    if not os.path.exists(args.voice_file):
        print(f"Error: Voice file not found at '{args.voice_file}'")
        sys.exit(1)

    # Read the text content from the file
    try:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        print("Successfully read text file.")
    except Exception as e:
        print(f"Error reading text file: {e}")
        sys.exit(1)

    if not text_content.strip():
        print("Error: The text file is empty.")
        sys.exit(1)

    print("Initializing IndexTTS2 model... (This may take a moment)")
    start_time = time.time()

    # Initialize the TTS model
    tts = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        use_fp16=args.fp16,
        use_deepspeed=args.deepspeed,
        use_cuda_kernel=args.cuda_kernel,
    )

    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    print("Starting audiobook generation...")
    generation_start_time = time.time()

    # Generate the audiobook
    try:
        tts.infer(
            spk_audio_prompt=args.voice_file,
            text=text_content,
            output_path=args.output_file,
            verbose=args.verbose,
            max_text_tokens_per_segment=args.max_text_tokens,
        )
        print(f"Audiobook generation completed in {time.time() - generation_start_time:.2f} seconds.")
        print(f"Audiobook saved to: {args.output_file}")
    except Exception as e:
        print(f"An error occurred during audiobook generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()