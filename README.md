

<div align="center">
<img src='assets/index_icon.png' width="250"/>
</div>

<div align="center">
<a href="docs/README_zh.md" style="font-size: 24px">简体中文</a> | 
<a href="README.md" style="font-size: 24px">English</a>
</div>

## 👉🏻 IndexTTS2 👈🏻

<center><h3>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h3></center>

[![IndexTTS2](assets/IndexTTS2_banner.png)](assets/IndexTTS2_banner.png)


<div align="center">
  <a href='https://arxiv.org/abs/2506.21619'>
    <img src='https://img.shields.io/badge/ArXiv-2506.21619-red?logo=arxiv'/>
  </a>
  <br/>
  <a href='https://github.com/index-tts/index-tts'>
    <img src='https://img.shields.io/badge/GitHub-Code-orange?logo=github'/>
  </a>
  <a href='https://index-tts.github.io/index-tts2.github.io/'>
    <img src='https://img.shields.io/badge/GitHub-Demo-orange?logo=github'/>
  </a>
  <br/>
  <a href='https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo'>
    <img src='https://img.shields.io/badge/HuggingFace-Demo-blue?logo=huggingface'/>
  </a>
  <a href='https://huggingface.co/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface' />
  </a>
  <br/>
  <a href='https://modelscope.cn/studios/IndexTeam/IndexTTS-2-Demo'>
    <img src='https://img.shields.io/badge/ModelScope-Demo-purple?logo=modelscope'/>
  </>
  <a href='https://modelscope.cn/models/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope'/>
  </a>
</div>


### Abstract

Existing autoregressive large-scale text-to-speech (TTS) models have advantages in speech naturalness, but their token-by-token generation mechanism makes it difficult to precisely control the duration of synthesized speech. This becomes a significant limitation in applications requiring strict audio-visual synchronization, such as video dubbing.

This paper introduces IndexTTS2, which proposes a novel, general, and autoregressive model-friendly method for speech duration control.

The method supports two generation modes: one explicitly specifies the number of generated tokens to precisely control speech duration; the other freely generates speech in an autoregressive manner without specifying the number of tokens, while faithfully reproducing the prosodic features of the input prompt.

Furthermore, IndexTTS2 achieves disentanglement between emotional expression and speaker identity, enabling independent control over timbre and emotion. In the zero-shot setting, the model can accurately reconstruct the target timbre (from the timbre prompt) while perfectly reproducing the specified emotional tone (from the style prompt).

To enhance speech clarity in highly emotional expressions, we incorporate GPT latent representations and design a novel three-stage training paradigm to improve the stability of the generated speech. Additionally, to lower the barrier for emotional control, we designed a soft instruction mechanism based on text descriptions by fine-tuning Qwen3, effectively guiding the generation of speech with the desired emotional orientation.

Finally, experimental results on multiple datasets show that IndexTTS2 outperforms state-of-the-art zero-shot TTS models in terms of word error rate, speaker similarity, and emotional fidelity. Audio samples are available at: <a href="https://index-tts.github.io/index-tts2.github.io/">IndexTTS2 demo page</a>.

**Tips:** Please contact the authors for more detailed information. For commercial usage and cooperation, please contact <u>indexspeech@bilibili.com</u>.


### Feel IndexTTS2

<div align="center">

**IndexTTS2: The Future of Voice, Now Generating**

[![IndexTTS2 Demo](assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)

*Click the image to watch the IndexTTS2 introduction video.*

</div>


### Contact

QQ Group：553460296(No.1) 663272642(No.4)  \
Discord：https://discord.gg/uT32E7KDmy  \
Email：indexspeech@bilibili.com  \
You are welcome to join our community! 🌏  \
欢迎大家来交流讨论！

> [!CAUTION]
> Thank you for your support of the bilibili indextts project!
> Please note that the **only official channel** maintained by the core team is: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts).
> ***Any other websites or services are not official***, and we cannot guarantee their security, accuracy, or timeliness.
> For the latest updates, please always refer to this official repository.


## 📣 Updates

- `2025/09/08` 🔥🔥🔥  We release **IndexTTS-2** to the world!
    - The first autoregressive TTS model with precise synthesis duration control, supporting both controllable and uncontrollable modes. <i>This functionality is not yet enabled in this release.</i>
    - The model achieves highly expressive emotional speech synthesis, with emotion-controllable capabilities enabled through multiple input modalities.
- `2025/05/14` 🔥🔥 We release **IndexTTS-1.5**, significantly improving the model's stability and its performance in the English language.
- `2025/03/25` 🔥 We release **IndexTTS-1.0** with model weights and inference code.
- `2025/02/12` 🔥 We submitted our paper to arXiv, and released our demos and test sets.


## 🖥️ Neural Network Architecture

Architectural overview of IndexTTS2, our state-of-the art speech model:

<picture>
  <img src="assets/IndexTTS2.png"  width="800"/>
</picture>


The key contributions of **IndexTTS2** are summarized as follows:

 - We propose a duration adaptation scheme for autoregressive TTS models. IndexTTS2 is the first autoregressive zero-shot TTS model to combine precise duration control with natural duration generation, and the method is scalable for any autoregressive large-scale TTS model.  
 - The emotional and speaker-related features are decoupled from the prompts, and a feature fusion strategy is designed to maintain semantic fluency and pronunciation clarity during emotionally rich expressions. Furthermore, a tool was developed for emotion control, utilizing natural language descriptions for the benefit of users.  
 - To address the lack of highly expressive speech data, we propose an effective training strategy, significantly enhancing the emotional expressiveness of zeroshot TTS to State-of-the-Art (SOTA) level.  
 - We will publicly release the code and pre-trained weights to facilitate future research and practical applications.  


## Model Download

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [😁 IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | [IndexTTS-2](https://modelscope.cn/models/IndexTeam/IndexTTS-2) |
| [IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |


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


### 🔥 IndexTTS2 Quickstart

#### 🌐 Web Demo

```bash
uv run webui.py
```

Open your browser and visit `http://127.0.0.1:7860` to see the demo.

You can also adjust the settings to enable features such as FP16 inference (lower
VRAM usage), DeepSpeed acceleration, compiled CUDA kernels for speed, etc. All
available options can be seen via the following command:

```bash
uv run webui.py -h
```

Have fun!

> [!IMPORTANT]
> It can be very helpful to use **FP16** (half-precision) inference. It is faster
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


#### 📝 Using IndexTTS2 in Python

To run scripts, you *must* use the `uv run <file.py>` command to ensure that
the code runs inside your current "uv" environment. It *may* sometimes also be
necessary to add the current directory to your `PYTHONPATH`, to help it find
the IndexTTS modules.

Example of running a script via `uv`:

```bash
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2.py
```

Here are several examples of how to use IndexTTS2 in your own scripts:

1. Synthesize new speech with a single reference audio file (voice cloning):

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "Translate for me, what is a surprise!"
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)
```

2. Using a separate, emotional reference audio file to condition the speech synthesis:

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)
```

3. When an emotional reference audio file is specified, you can optionally set
   the `emo_alpha` to adjust how much it affects the output.
   Valid range is `0.0 - 1.0`, and the default value is `1.0` (100%):

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)
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
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "哇塞！这个爆率也太高了！欧皇附体了！"
tts.infer(spk_audio_prompt='examples/voice_10.wav', text=text, output_path="gen.wav", emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0], use_random=False, verbose=True)
```

5. Alternatively, you can enable `use_emo_text` to guide the emotions based on
   your provided `text` script. Your text script will then automatically
   be converted into emotion vectors.
   It's recommended to use `emo_alpha` around 0.6 (or lower) when using the text
   emotion modes, for more natural sounding speech.
   You can introduce randomness with `use_random` (default: `False`;
   `True` enables randomness):

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "快躲起来！是他要来了！他要来抓我们了！"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```

6. It's also possible to directly provide a specific text emotion description
   via the `emo_text` parameter. Your emotion text will then automatically be
   converted into emotion vectors. This gives you separate control of the text
   script and the text emotion description:

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
```

> [!TIP]
> **Pinyin Usage Notes:**
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

### Legacy: IndexTTS1 User Guide

You can also use our previous IndexTTS1 model by importing a different module:

```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
voice = "examples/voice_07.wav"
text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, 'gen.wav')
```

For more detailed information, see [README_INDEXTTS_1_5](archive/README_INDEXTTS_1_5.md),
or visit the IndexTTS1 repository at <a href="https://github.com/index-tts/index-tts/tree/v1.5.0">index-tts:v1.5.0</a>.


## Our Releases and Demos

### IndexTTS2: [[Paper]](https://arxiv.org/abs/2506.21619); [[Demo]](https://index-tts.github.io/index-tts2.github.io/); [[ModelScope]](https://modelscope.cn/studios/IndexTeam/IndexTTS-2-Demo); [[HuggingFace]](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo)

### IndexTTS1: [[Paper]](https://arxiv.org/abs/2502.05512); [[Demo]](https://index-tts.github.io/); [[ModelScope]](https://modelscope.cn/studios/IndexTeam/IndexTTS-Demo); [[HuggingFace]](https://huggingface.co/spaces/IndexTeam/IndexTTS)


## Acknowledgements

1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)
6. [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
7. [seed-vc](https://github.com/Plachtaa/seed-vc)

## Contributors in Bilibili
We sincerely thank colleagues from different roles at Bilibili, whose combined efforts made the IndexTTS series possible.

### Core Authors
 - **Wei Deng** - Core author; Initiated the IndexTTS project, led the development of the IndexTTS1 data pipeline, model architecture design and training, as well as iterative optimization of the IndexTTS series of models, focusing on fundamental capability building and performance optimization.
 - **Siyi Zhou** – Core author; in IndexTTS2, led model architecture design and training pipeline optimization, focusing on key features such as multilingual and emotional synthesis.
 - **Jingchen Shu** - Core author; worked on overall architecture design, cross-lingual modeling solutions, and training strategy optimization, driving model iteration.
 - **Xun Zhou** - Core author; worked on cross-lingual data processing and experiments, explored multilingual training strategies, and contributed to audio quality improvement and stability evaluation.
 - **Jinchao Wang** - Core author; worked on model development and deployment, building the inference framework and supporting system integration.
 - **Yiquan Zhou** - Core author; contributed to model experiments and validation, and proposed and implemented text-based emotion control.
 - **Yi He** - Core author; contributed to model experiments and validation.
 - **Lu Wang** – Core author; worked on data processing and model evaluation, supporting model training and performance verification.

### Technical Contributors
 - **Yining Wang** - Supporting contributor; contributed to open-source code implementation and maintenance, supporting feature adaptation and community release.
 - **Yong Wu** - Supporting contributor; worked on data processing and experimental support, ensuring data quality and efficiency for model training and iteration.
 - **Yaqin Huang** – Supporting contributor; contributed to systematic model evaluation and effect tracking, providing feedback to support iterative improvements.
 - **Yunhan Xu** – Supporting contributor; provided guidance in recording and data collection, while also offering feedback from a product and operations perspective to improve usability and practical application.
 - **Yuelang Sun** – Supporting contributor; provided professional support in audio recording and data collection, ensuring high-quality data for model training and evaluation.
 - **Yihuang Liang** - Supporting contributor; worked on systematic model evaluation and project promotion, helping IndexTTS expand its reach and engagement.

### Technical Guidance
 - **Huyang Sun** - Provided strong support for the IndexTTS project, ensuring strategic alignment and resource backing.
 - **Bin Xia** - Contributed to the review, optimization, and follow-up of technical solutions, focusing on ensuring model effectiveness.


## 📚 Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.


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
