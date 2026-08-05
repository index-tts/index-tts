

<div align="center">
<img src='assets/index_icon.png' width="250"/>
</div>

<div align="center">
<a href="docs/README_zh.md" style="font-size: 24px">简体中文</a> | 
<a href="README.md" style="font-size: 24px">English</a>
</div>

## 👉🏻 IndexTTS 👈🏻

<!-- |**HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
|| [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2) | [IndexTTS-2.5](https://modelscope.cn/models/IndexTeam/IndexTTS-2) |
| [IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | [IndexTTS-2](https://modelscope.cn/models/IndexTeam/IndexTTS-2) |
| [IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) | -->

| Model | Demos | Paper | Modelscope | HuggingFace |
| :--- | :---: | :---: | :---: | :---: |
| **IndexTTS-2.5** | [Demos](https://index-tts.github.io/index-tts2-5.github.io/) | [Paper](https://arxiv.org/abs/2601.03888) | [Modelscope](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5) | [HuggingFace](https://huggingface.co/IndexTeam/IndexTTS-2.5) |
| **IndexTTS-2** | [Demos](https://index-tts.github.io/index-tts2.github.io/) | [Paper](https://arxiv.org/abs/2506.21619) | [Modelscope](https://modelscope.cn/models/IndexTeam/IndexTTS-2) | [HuggingFace](https://huggingface.co/IndexTeam/IndexTTS-2) |
| **IndexTTS-1.5** | [Demos](https://index-tts.github.io/) | [Paper](https://arxiv.org/abs/2502.05512) | [Modelscope](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) | [HuggingFace](https://huggingface.co/IndexTeam/IndexTTS-1.5) |
| **IndexTTS** | [Demos](https://index-tts.github.io/) | [Paper](https://arxiv.org/abs/2502.05512) | [Modelscope](https://modelscope.cn/models/IndexTeam/Index-TTS) | [HuggingFace](https://huggingface.co/IndexTeam/Index-TTS) |

## 📣 Updates

- `2026/07/17` 🔥 We release **IndexTTS-2.5**
    - The model now supports Chinese, English, Japanese, Spanish and Arabic, with faster inference speed compared to IndexTTS-2, while maitaining the cross-lingual and timbre-emotion disentanglement capabilities.
    - The model improves the controbility of Chinese Pinyin and English CMU phonemes and Japanese Kana. 
- `2025/09/08` 🔥 We release **IndexTTS-2**
    - The first autoregressive TTS model with precise synthesis duration control, supporting both controllable and uncontrollable modes. <i>This functionality is not yet enabled in this release.</i>
    - The model achieves highly expressive emotional speech synthesis, with emotion-controllable capabilities enabled through multiple input modalities.
- `2025/05/14` 🔥 We release **IndexTTS-1.5**, significantly improving the model's stability and its performance in the English language.
- `2025/03/25` 🔥 We release **IndexTTS-1.0** with model weights and inference code.
- `2025/02/12` 🎉 We submitted our paper to arXiv, and released our demos and test sets.
### Feel IndexTTS

<div align="center">

**IndexTTS2.5: The Future of Voice, Now Generating**

[![IndexTTS2.5 Demo](assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)


**IndexTTS2: The Future of Voice, Now Generating**

[![IndexTTS2 Demo](assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)

</div>

## Evaluation

Table 1: Zero-shot TTS evaluation results on CV3-Eval test set (Arabic uses an in-house test set). WER (%) ↓ and Speaker Similarity (SS) ↑ are reported. †Results cited from the original paper.
| Model | Params | test-zh<br>WER (%) ↓ | test-zh<br>SS (%) ↑ | test-en<br>WER (%) ↓ | test-en<br>SS (%) <br> ↑ | test-es<br>WER (%) ↓ | test-es<br>SS (%) ↑ | test-ja<br>WER (%) ↓ | test-ja<br>SS (%) ↑ | test-ar<br>WER (%) ↓ | test-ar<br>SS (%) ↑ | Average<br>WER (%) <br> ↓ | Average<br>SS <br> (%) <br> ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VoxCPM2 | 2B | 3.88 | 74.99 | 5.13 | 71.57 | 5.49 | 74.67 | 6.69 | 72.90 | 14.94 | 65.99 | 7.22 | 72.02 |
| OmniVoice | 0.8B | 3.41 | 72.99 | 3.62 | 70.13 | 3.52 | 74.14 | 5.38 | 70.49 | 17.88 | 64.22 | 6.76 | 70.39 |
| Moss-TTS 1.5 | 8B | 4.02 | 72.68 | 4.45 | 67.46 | 3.83 | 71.75 | 10.97 | 68.71 | 23.71 | 62.21 | 9.40 | 68.56 |
| CosyVoice3-0.5B | 0.5B | 3.84 | 80.01 | 4.88 | 74.16 | 4.04 | 78.85 | - | 76.36 | - | - | - | - |
| CosyVoice3-1.5B | 1.5B | 3.91† | - | 4.99† | - | 4.47† | - | 7.57† | - | - | - | - | - |
| FireRedTTS-2 | 1.5B | 8.22 | 68.10 | 14.92 | 56.93 | - | - | - | - | - | - | - | - |
| Fish Audio S2 Pro | 4B | 3.62 | 67.79 | 3.83 | 61.66 | 2.93 | 67.44 | 5.15 | 66.15 | 14.15 | 59.43 | 5.94 | 64.49 |
| Qwen3-TTS | 1.7B | 3.27 | 73.02 | 5.06 | 67.17 | 2.87 | 73.17 | 5.89 | 70.18 | - | - | - | - |
| IndexTTS2.5 | 0.8B | 4.36 | 77.10 | 5.12 | 68.06 | 3.75 | 76.39 | 5.66 | 74.62 | 14.88 | 69.74 | 6.75 | 73.18 |
| IndexTTS2.5-RL | 0.8B | 3.93 | 77.92 | 3.89 | 67.79 | 3.33 | 76.68 | 5.30 | 75.41 | 13.58 | 70.36 | 6.00 | 73.63 |

Table 2: Cross-lingual TTS evaluation on CV3-Eval test set (Chinese prompt → target language, Arabic uses an in-house test set). WER (%) ↓ and Speaker Similarity (SS) ↑ are reported.
| Model | Params | zh→en<br>WER (%) ↓ | zh→en<br>SS (%) ↑ | zh→es<br>WER (%) ↓ | zh→es<br>SS (%) ↑ | zh→ja<br>WER (%) ↓ | zh→ja<br>SS (%) ↑ | zh→ar<br>WER (%) ↓ | zh→ar<br>SS (%) ↑ | Average<br>WER (%) ↓ | Average<br>SS <br> (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VoxCPM2 | 2B | 4.48 | 64.25 | 16.38 | 64.89 | 11.84 | 71.54 | 11.09 | 67.62 | 10.95 | 67.08 |
| OmniVoice | 0.8B | 3.74 | 64.91 | 5.84 | 62.08 | 9.09 | 69.06 | 19.80 | 65.27 | 9.62 | 65.33 |
| Moss-TTS 1.5 | 8B | 6.13 | 59.23 | 4.32 | 56.63 | 11.52 | 65.54 | 17.03 | 62.93 | 9.75 | 61.08 |
| CosyVoice3-0.5B | 0.5B | 3.23 | 62.79 | 4.58 | 64.04 | - | - | - | - | - | - |
| CosyVoice3-1.5B | 1.5B | 4.32 | - | - | - | 13.70 | - | - | - | - | - |
| FireRedTTS-2 | 1.5B | 9.34 | 53.19 | 12.25 | 58.31 | 19.05 | 64.12 | - | - | - | - |
| Fish Audio S2 Pro | 4B | 4.14 | 55.89 | 4.46 | 55.57 | 10.48 | 61.74 | 14.49 | 59.80 | 8.39 | 58.25 |
| Qwen3-TTS | 1.7B | 5.74 | 63.04 | 5.15 | 68.02 | 36.09 | 65.71 | - | - | - | - |
| IndexTTS2.5 | 0.8B | 3.62 | 63.83 | 5.17 | 65.48 | 6.57 | 74.16 | 9.51 | 71.02 | 6.22 | 68.62 |
| IndexTTS2.5-RL | 0.8B | 3.55 | 67.47 | 4.86 | 64.47 | 6.38 | 75.82 | 9.89 | 73.05 | 6.17 | 70.20 |


### Contact

QQ Group：663272642(No.4) 1013410623(No.5)  \
Discord：https://discord.gg/uT32E7KDmy  \
Email：indexspeech@bilibili.com  \
You are welcome to join our community! 🌏  \
欢迎大家来交流讨论！

> [!CAUTION]
> Thank you for your support of the bilibili indextts project!
> Please note that the **only official channel** maintained by the core team is: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts).
> ***Any other websites or services are not official***, and we cannot guarantee their security, accuracy, or timeliness.
> For the latest updates, please always refer to this official repository.

**Tips:** Please contact the authors for more detailed information. For commercial usage and cooperation, please contact <u>indexspeech@bilibili.com</u>.

## Usage Instructions

### ⚙️ Environment Setup

1. Ensure that you have both [git](https://git-scm.com/downloads)
   and [git-lfs](https://git-lfs.com/) on your system.

The Git-LFS plugin must also be enabled on your current user account:

```bash
git lfs install
```

2. Download this repository:

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
git lfs pull  # download large repository files
```

3. Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/).
   It is *required* for a reliable, modern installation environment.

> [!TIP]
> **Quick & Easy Installation Method:**
> 
> There are many convenient ways to install the `uv` command on your computer.
> Please check the link above to see all options. Alternatively, if you want
> a very quick and easy method, you can install it as follows:
> 
> ```bash
> pip install -U uv
> ```

> [!WARNING]
> We **only** support the `uv` installation method. Other tools, such as `conda`
> or `pip`, don't provide any guarantees that they will install the correct
> dependency versions. You will almost certainly have *random bugs, error messages,*
> ***missing GPU acceleration**, and various other problems* if you don't use `uv`.
> Please *do not report any issues* if you use non-standard installations, since
> almost all such issues are invalid.
> 
> Furthermore, `uv` is [up to 115x faster](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md)
> than `pip`, which is another *great* reason to embrace the new industry-standard
> for Python project management.

4. Install required dependencies:

We use `uv` to manage the project's dependency environment. The following command
will *automatically* create a `.venv` project-directory and then installs the correct
versions of Python and all required dependencies:

```bash
uv sync --all-extras
```

If the download is slow, please try a *local mirror*, for example any of these
local mirrors in China (choose one mirror from the list below):

```bash
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"

uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

> [!TIP]
> **Available Extra Features:**
> 
> - `--all-extras`: Automatically adds *every* extra feature listed below. You can
>   remove this flag if you want to customize your installation choices.
> - `--extra webui`: Adds WebUI support (recommended).
> - `--extra deepspeed`: Adds DeepSpeed support (may speed up inference on some
>   systems).

> [!IMPORTANT]
> **Important (Windows):** The DeepSpeed library may be difficult to install for
> some Windows users. You can skip it by removing the `--all-extras` flag. If you
> want any of the other extra features above, you can manually add their specific
> feature flags instead.
> 
> **Important (Linux/Windows):** If you see an error about CUDA during the installation,
> please ensure that you have installed NVIDIA's [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
> version **12.8** (or newer) on your system.

5. Download the required models via [uv tool](https://docs.astral.sh/uv/guides/tools/#installing-tools):

Download via `huggingface-cli`:

```bash
uv tool install "huggingface-hub[cli,hf_xet]"

hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

Or download via `modelscope`:

```bash
uv tool install "modelscope"

modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

> [!IMPORTANT]
> If the commands above aren't available, please carefully read the `uv tool`
> output. It will tell you how to add the tools to your system's path.

> [!NOTE]
> In addition to the above models, some small models will also be automatically
> downloaded when the project is run for the first time. If your network environment
> has slow access to HuggingFace, it is recommended to execute the following
> command before running the code:
> 
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```


#### 🖥️ Checking PyTorch GPU Acceleration

If you need to diagnose your environment to see which GPUs are detected,
you can use our included utility to check your system:

```bash
uv run tools/gpu_check.py
```


### 🔥 Quickstart

#### 🌐 Web Demo

```bash
# IndexTTS2 (default)
uv run webui.py

# IndexTTS2.5
uv run webui.py --version 2.5 --model_dir ./checkpoints_25
```

Open your browser and visit `http://127.0.0.1:7860` to see the demo.

You can also adjust the settings to enable features such as BF16(IndexTTS 2.5)/FP16(IndexTTS 2) inference (lower
VRAM usage), DeepSpeed acceleration, compiled CUDA kernels for speed, etc. All
available options can be seen via the following command:

```bash
uv run webui.py -h
```

Have fun!

> [!IMPORTANT]
> It can be very helpful to use **FP16/BF16** (half-precision) inference. It is faster
> and uses less VRAM, with a very small quality loss.
> 
> **DeepSpeed** *may* also speed up inference on some systems, but it could also
> make it slower. The performance impact is highly dependent on your specific
> hardware, drivers and operating system. Please try with and without it,
> to discover what works best on your personal system.
> 
> Lastly, be aware that *all* `uv` commands will **automatically activate** the correct
> per-project virtual environments. Do *not* manually activate any environments
> before running `uv` commands, since that could lead to dependency conflicts!


#### 📝 Using in Python

To run scripts, you *must* use the `uv run <file.py>` command to ensure that
the code runs inside your current "uv" environment. It *may* sometimes also be
necessary to add the current directory to your `PYTHONPATH`, to help it find
the IndexTTS modules.

Example of running a script via `uv`:

```bash
# IndexTTS2
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2.py

# IndexTTS2.5
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2_5.py \
  --cfg_path checkpoints/config_v2_5.yaml \
  --model_dir checkpoints \
  --text "Hello world" \
  --lang EN
```

Here are several examples of how to use in your own scripts:

0. Initialize IndexTTS
```python
# IndexTTS2
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)

# IndexTTS2.5
from indextts.infer_v2_5 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints_25/config_v2_5.yaml", model_dir="checkpoints_25", use_bf16=True)
```
1. Synthesize new speech with a single reference audio file (voice cloning):

```python
text = "Translate for me, what is a surprise!"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)

# IndexTTS2.5 (multilingual, with language selection)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="EN", output_path="gen.wav", verbose=True)
```

2. Using a separate, emotional reference audio file to condition the speech synthesis:

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, lang="ZH", output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)

```

3. When an emotional reference audio file is specified, you can optionally set
   the `emo_alpha` to adjust how much it affects the output.
   Valid range is `0.0 - 1.0`, and the default value is `1.0` (100%):

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", lang="ZH", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)
```

4. It's also possible to omit the emotional reference audio and instead provide
   an 8-float list specifying the intensity of each emotion, in the following order:
   `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.
   You can additionally use the `use_random` parameter to introduce stochasticity
   during inference; the default is `False`, and setting it to `True` enables
   randomness:

> [!NOTE]
> Enabling random sampling will reduce the voice cloning fidelity of the speech
> synthesis.

```python
text = "对不起嘛！我的记性真的不太好，但是和你在一起的事情，我都会努力记住的~"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/09.wav', text=text, output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/09.wav', text=text, lang="ZH", output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)
```

5. Alternatively, you can enable `use_emo_text` to guide the emotions based on
   your provided `text` script. Your text script will then automatically
   be converted into emotion vectors.
   It's recommended to use `emo_alpha` around 0.6 (or lower) when using the text
   emotion modes, for more natural sounding speech.
   You can introduce randomness with `use_random` (default: `False`;
   `True` enables randomness):

```python
text = "快躲起来！是他要来了！他要来抓我们了！"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", utput_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```

6. It's also possible to directly provide a specific text emotion description
   via the `emo_text` parameter. Your emotion text will then automatically be
   converted into emotion vectors. This gives you separate control of the text
   script and the text emotion description:

```python
text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
```

> [!TIP]
>
> **IndexTTS2.5 Pinyin/English phonemes/Japan Kana Usage Notes:**
> 
> IndexTTS2.5 now can support these character replacement, with better instruction-following capability.
> For the full list of valid entries, please refer to `checkpoints/pinyin.vocab` for Pinyin, and 'https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b' for CMU dictionary.
>
> Example:
> ```
> 他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。
>
> He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details of the contract.
>
> 彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
> ```
> **IndexTTS2 Pinyin Usage Notes:**
> 
> IndexTTS2 still supports mixed modeling of Chinese characters and Pinyin.
> When you need precise pronunciation control, please provide text with specific Pinyin annotations to activate the Pinyin control feature.
> Note that Pinyin control does not work for every possible consonant–vowel combination; only valid Chinese Pinyin cases are supported.
> For the full list of valid entries, please refer to `checkpoints/pinyin.vocab`.
>
> Example:
> ```
> 之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。
> ```
> **IndexTTS1 Usage Notes:**
>
>You can also use our previous IndexTTS1 model by importing a different module:
>
>```python
>from indextts.infer import IndexTTS
>tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
>voice = "examples/voice_07.wav"
>text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比>如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可>以访问 bilibili studio，相信我，你们也会吃惊的。"
>tts.infer(voice, text, 'gen.wav')
>```
>
>For more detailed information, see [README_INDEXTTS_1_5](archive/README_INDEXTTS_1_5.md),
or visit the IndexTTS1 repository at <a href="https://github.com/index-tts/index-tts/tree/v1.5.0">index-tts:v1.5.0</a>.



## Acknowledgements

1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)
6. [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
7. [seed-vc](https://github.com/Plachtaa/seed-vc)

## 📚 Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.

IndexTTS2.5:

```
@misc{li2026indextts25technicalreport,
      title={IndexTTS 2.5 Technical Report}, 
      author={Yunpei Li and Xun Zhou and Jinchao Wang and Lu Wang and Yong Wu and Siyi Zhou and Yiquan Zhou and Jingchen Shu},
      year={2026},
      eprint={2601.03888},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.03888}, 
}
```

IndexTTS2:

```
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```


IndexTTS:

```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025},
  doi={10.48550/arXiv.2502.05512},
  url={https://arxiv.org/abs/2502.05512}
}
```
