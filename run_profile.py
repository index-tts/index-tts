if __name__ == "__main__":
    import csv
    import os
    import torch
    from torch.profiler import profile, ProfilerActivity
    from indextts.infer_v2 import IndexTTS2

    torch.cuda.reset_peak_memory_stats()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    traces_dir = os.path.join(root_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    prompt_wav = os.path.join(root_dir, "examples/voice_01.wav")
    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    tts = IndexTTS2(
        cfg_path=os.path.join(root_dir, "checkpoints/config.yaml"),
        model_dir=os.path.join(root_dir, "checkpoints"),
        use_cuda_kernel=False,
        use_torch_compile=False,
        use_fp16=True,
        use_int8=True,
    )

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory used (after model load): {peak_mb:.0f} MB")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
    ) as prof:
        tts.infer(
            spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True
        )

    # Save Chrome trace JSON
    prof.export_chrome_trace(os.path.join(traces_dir, "trace.json"))
    print(f"Trace saved to {traces_dir}/trace.json")

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory used (total): {peak_mb:.0f} MB")
