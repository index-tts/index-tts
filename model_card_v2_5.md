---
language:
  - zh
  - en
  - ja
  - es
  - ar
license: other
library_name: indextts
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - tts
  - zero-shot
  - voice-cloning
  - multilingual
  - cross-lingual
  - emotion-controllable
---

# IndexTTS-2.5

IndexTTS-2.5 is a zero-shot text-to-speech model that performs voice cloning from a single reference audio. It supports **Chinese, English, Japanese, Spanish, and Arabic**, with cross-lingual voice transfer and disentangled emotion control.

Compared to IndexTTS-2, it adds Japanese/Spanish/Arabic support, improves inference speed, and enhances controllability of Chinese Pinyin, English CMU phonemes, and Japanese Kana.

## Model Details

- **Developed by:** IndexTeam, Bilibili
- **Model type:** Autoregressive zero-shot TTS (GPT + DiT + BigVGAN)
- **Languages:** Chinese, English, Japanese, Spanish, Arabic
- **License:** Other
- **Finetuned from:** —

### Model Sources

- **Repository:** [github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)
- **Demo:** [IndexTTS-2.5 Demo Page](https://index-tts.github.io/index-tts2-5.github.io/)

## How to Get Started

### Installation

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
git lfs pull
pip install -U uv
uv sync --all-extras
```

### Download Model Weights

```bash
# HuggingFace
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints

# ModelScope
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints
```

### Quick Inference

```python
from indextts.infer_v2_5 import IndexTTS2

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True)

# Basic voice cloning
tts.infer(
    spk_audio_prompt="prompt.wav",
    text="Hello, this is a voice cloning demo.",
    lang="EN",
    output_path="output.wav",
)

# With emotion control
tts.infer(
    spk_audio_prompt="prompt.wav",
    text="快躲起来！是他要来了！",
    lang="ZH",
    output_path="output.wav",
    emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0],
)

# With Pinyin/phoneme annotation
tts.infer(
    spk_audio_prompt="prompt.wav",
    text="他在银<行|XING2>里<行|HANG2>走了半天。",
    lang="ZH",
    output_path="output.wav",
)
```

### Web Demo

```bash
uv run webui.py --version 2.5 --model_dir ./checkpoints
```

## Uses

### Direct Use

- Zero-shot voice cloning from a single reference audio
- Multilingual speech synthesis (Chinese, English, Japanese, Spanish, Arabic)
- Cross-lingual voice transfer (e.g., Chinese speaker voice → English output)
- Emotion-controllable speech synthesis via emotion vectors, emotion reference audio, or text-based emotion detection

### Downstream Use

- Audiobook and podcast production
- Voice dubbing and localization
- Conversational AI and virtual assistants

### Out-of-Scope Use

- Impersonation or deception without consent
- Generating misleading or fraudulent audio content
- Any use that violates applicable laws or regulations

## Bias, Risks, and Limitations

- Voice cloning quality may vary across speakers and languages.
- The model may produce artifacts or unnatural prosody for very long or highly complex text.
- Cross-lingual transfer quality depends on the target language and speaker characteristics.
- The model does not verify speaker identity or consent. Users are responsible for ethical use.

## Evaluation

### Zero-Shot TTS (CV3-Eval)

| Model | Params | zh WER↓ | zh SS↑ | en WER↓ | en SS↑ | es WER↓ | es SS↑ | ja WER↓ | ja SS↑ | ar WER↓ | ar SS↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VoxCPM2 | 2B | 3.88 | 74.99 | 5.13 | 71.57 | 5.49 | 74.67 | 6.69 | 72.90 | 14.94 | 65.99 |
| CosyVoice3-0.5B | 0.5B | 3.84 | 80.01 | 4.88 | 74.16 | 4.04 | 78.85 | - | 76.36 | - | - |
| Fish Audio S2 Pro | 4B | 3.62 | 67.79 | 3.83 | 61.66 | 2.93 | 67.44 | 5.15 | 66.15 | 14.15 | 59.43 |
| Qwen3-TTS | 1.7B | 3.27 | 73.02 | 5.06 | 67.17 | 2.87 | 73.17 | 5.89 | 70.18 | - | - |
| **IndexTTS2.5** | **0.8B** | 4.36 | 77.10 | 5.12 | 68.06 | 3.75 | 76.39 | 5.66 | 74.62 | 14.88 | 69.74 |
| **IndexTTS2.5-RL** | **0.8B** | 3.93 | 77.92 | 3.89 | 67.79 | 3.33 | 76.68 | 5.30 | 75.41 | 13.58 | 70.36 |

### Cross-Lingual TTS (Chinese prompt → target language)

| Model | Params | zh→en WER↓ | zh→en SS↑ | zh→es WER↓ | zh→es SS↑ | zh→ja WER↓ | zh→ja SS↑ | zh→ar WER↓ | zh→ar SS↑ |
|---|---|---|---|---|---|---|---|---|---|
| VoxCPM2 | 2B | 4.48 | 64.25 | 16.38 | 64.89 | 11.84 | 71.54 | 11.09 | 67.62 |
| CosyVoice3-0.5B | 0.5B | 3.23 | 62.79 | 4.58 | 64.04 | - | - | - | - |
| **IndexTTS2.5** | **0.8B** | 3.62 | 63.83 | 5.17 | 65.48 | 6.57 | 74.16 | 9.51 | 71.02 |
| **IndexTTS2.5-RL** | **0.8B** | 3.55 | 67.47 | 4.86 | 64.47 | 6.38 | 75.82 | 9.89 | 73.05 |

## Citation

```bibtex
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```
