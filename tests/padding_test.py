import torch
import torchaudio
from indextts.infer import IndexTTS
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from torch.nn import functional as F

if __name__ == "__main__":
    """
    Test the padding of text tokens in inference.
    ```
    python tests/padding_test.py checkpoints
    python tests/padding_test.py IndexTTS-1.5
    ```
    """
    import transformers
    transformers.set_seed(42)
    import sys
    sys.path.append("..")
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = "checkpoints"
    audio_prompt="tests/sample_prompt.wav"
    tts = IndexTTS(cfg_path=f"{model_dir}/config.yaml", model_dir=model_dir, use_fp16=False, use_cuda_kernel=False)
    text = "晕 XUAN4 是 一 种 not very good GAN3 觉"
    text_tokens = tts.tokenizer.encode(text)
    text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=tts.device).unsqueeze(0) # [1, L]

    audio, sr = torchaudio.load(audio_prompt)
    audio = torch.mean(audio, dim=0, keepdim=True)
    audio = torchaudio.transforms.Resample(sr, 24000)(audio)
    auto_conditioning = MelSpectrogramFeatures()(audio).to(tts.device)
    cond_mel_lengths = torch.tensor([auto_conditioning.shape[-1]]).to(tts.device)
    with torch.no_grad():
        kwargs = {
            "cond_mel_lengths": cond_mel_lengths,
            "do_sample": False,
            "top_p": 0.8,
            "top_k": None,
            "temperature": 1.0,
            "num_return_sequences": 1,
            "length_penalty": 0.0,
            "num_beams": 1,
            "repetition_penalty": 10.0,
            "max_generate_length": 100,
        }
        # baseline for non-pad
        baseline = tts.gpt.inference_speech(auto_conditioning, text_tokens, **kwargs)
        baseline = baseline.squeeze(0)
        print("Inference padded text tokens...")
        pad_text_tokens = [
            F.pad(text_tokens, (8, 0), value=0), # left bos
            F.pad(text_tokens, (0, 8), value=1), # right eos
            F.pad(F.pad(text_tokens, (4, 0), value=0), (0, 4), value=1), # both side
            F.pad(F.pad(text_tokens, (6, 0), value=0), (0, 2), value=1),
            F.pad(F.pad(text_tokens, (0, 4), value=0), (0, 4), value=1),
        ]
        output_for_padded = []
        for t in pad_text_tokens:
            # test for each padded text
            out = tts.gpt.inference_speech(auto_conditioning, text_tokens, **kwargs)
            output_for_padded.append(out.squeeze(0))
        # batched inference
        print("Inference padded text tokens as one batch...")
        batched_text_tokens = torch.cat(pad_text_tokens, dim=0).to(tts.device)
        assert len(pad_text_tokens) == batched_text_tokens.shape[0] and batched_text_tokens.ndim == 2
        batch_output = tts.gpt.inference_speech(auto_conditioning, batched_text_tokens, **kwargs)
        del pad_text_tokens
    mismatch_idx = []
    print("baseline:", baseline.shape, baseline)
    print("--"*10)
    print("baseline vs padded output:")
    for i in range(len(output_for_padded)):
        if not baseline.equal(output_for_padded[i]):
            mismatch_idx.append(i)
    
    if len(mismatch_idx) > 0:
        print("mismatch:", mismatch_idx)
        for i in mismatch_idx:
            print(f"[{i}]: {output_for_padded[i]}")
    else:
        print("all matched")
    
    del output_for_padded
    print("--"*10)
    print("baseline vs batched output:")
    mismatch_idx = []
    for i in range(batch_output.shape[0]):
        if not baseline.equal(batch_output[i]):
            mismatch_idx.append(i)
    if len(mismatch_idx) > 0:
        print("mismatch:", mismatch_idx)
        for i in mismatch_idx:
            print(f"[{i}]: {batch_output[i]}")
    
    else:
        print("all matched")
    
    print("Test finished.")