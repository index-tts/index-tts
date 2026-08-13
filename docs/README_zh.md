<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/indextts_icon_dark.png"/>
  <img src="../assets/indextts_icon_light.png" width="300"/>
</picture>

**工业级可控、高效的零样本文本转语音系统**

简体中文 | [English](../README.md) | [日本語](README_ja.md) | [Español](README_es.md) | [العربية](README_ar.md)

[![GitHub Stars](https://img.shields.io/github/stars/index-tts/index-tts?style=flat&logo=github)](https://github.com/index-tts/index-tts/stargazers)
[![arXiv](https://img.shields.io/badge/arXiv-2601.03888-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.03888)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/uT32E7KDmy)

</div>

IndexTTS 是一个零样本文本转语音（TTS）系统，只需一段参考音频即可克隆音色。
最新发布的 **IndexTTS-2.5** 支持中文、英文、日语、西班牙语和阿拉伯语，
具备细粒度情感控制、语速控制、发音控制（拼音 / CMU 音素 / 日语假名）能力，
推理速度较 IndexTTS-2 更快。

---

## 🗂️ 模型列表

| 模型 | 演示 | 论文 | ModelScope | HuggingFace |
| :--- | :---: | :---: | :---: | :---: |
| **IndexTTS-2.5** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/index-tts2-5.github.io/) [![Studio](https://img.shields.io/badge/Studio-ModelScope-purple?logo=modelscope)](https://modelscope.cn/studios/IndexTeam/IndexTTS-2.5) [![Space](https://img.shields.io/badge/Space-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/IndexTeam/IndexTTS-2.5-Demo) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2601.03888) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-2.5) |
| **IndexTTS-2** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/index-tts2.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2506.21619) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-2) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-2) |
| **IndexTTS-1.5** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2502.05512) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-1.5) |
| **IndexTTS** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2502.05512) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/Index-TTS) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/Index-TTS) |

## 📣 更新日志

- `2026/08/10` 🔥 **IndexTTS-2.5** 全球发布！
  - 模型现已支持中文、英文、日语、西班牙语和阿拉伯语，推理速度较 IndexTTS-2 更快，同时保持跨语言合成与音色-情感解耦能力。
  - 模型提升了中文拼音、英文 CMU 音素和日语假名的可控性。
  - 支持通过 `duration_factor` 控制语速（0.5–2.0 倍时长）。
  - 支持通过 [vLLM](https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5) 进行生产环境部署。
- `2025/09/08` 🔥 **IndexTTS-2** 全球发布！
  - 首个支持精确合成时长控制的自回归 TTS 模型，支持可控与非可控模式。<i>本版本暂未开放该功能。</i>
  - 模型实现高度情感表达的语音合成，支持多模态情感控制。
- `2025/05/14` 🔥 **IndexTTS-1.5** 发布，显著提升模型稳定性及英文表现。
- `2025/03/25` 🔥 **IndexTTS-1.0** 发布，开放模型权重与推理代码。
- `2025/02/12` 🎉 论文提交 arXiv，发布演示与测试集。

## 🎬 演示视频

<div align="center">

**IndexTTS-2.5：语音未来，现已生成**

[![IndexTTS2.5 Demo](../assets/index2.5_video_cover.png)](https://www.bilibili.com/video/BV1uvMk6ZEdK/)

**IndexTTS-2：语音未来，现已生成**

[![IndexTTS2 Demo](../assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)

</div>

## 🚀 快速开始

### 1. 环境准备

请确保已安装 [git](https://git-scm.com/downloads)，然后下载本仓库：

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
```

示例音频会在首次运行时按需从 HuggingFace/ModelScope 自动下载，无需 Git LFS。

### 2. 安装依赖

我们使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理项目依赖环境，
这是保证安装可靠的**必需**工具：

```bash
pip install -U uv  # 其他安装方式见上方官网链接
```

```bash
uv sync --all-extras
```

该命令会*自动*创建 `.venv` 虚拟环境，并安装正确版本的 Python 及所有依赖。

如下载缓慢，可选用国内镜像：

```bash
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"

uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

> [!TIP]
> **可选功能：**
>
> - `--all-extras`：自动安装下方列出的全部可选功能。可去除此选项自定义安装。
> - `--extra webui`：安装 WebUI 支持（推荐）。
> - `--extra deepspeed`：安装 DeepSpeed 加速（部分环境可加速推理）。

> [!IMPORTANT]
> **Windows 注意：** DeepSpeed 在部分 Windows 环境较难安装，可去除 `--all-extras`，
> 手动添加所需的其他功能选项。
>
> **Linux/Windows 注意：** 如遇 CUDA 相关报错，请确保已安装 NVIDIA
> [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) **12.8** 及以上版本。

### 3. 下载模型

通过 [uv tool](https://docs.astral.sh/uv/guides/tools/#installing-tools) 下载所需模型：

HuggingFace 下载：

```bash
uv tool install "huggingface-hub"

# IndexTTS-2.5
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints

# IndexTTS-2
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints_2
```

ModelScope 下载：

```bash
uv tool install "modelscope"

# IndexTTS-2.5
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints

# IndexTTS-2
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints_2
```

> [!IMPORTANT]
> 如上述命令无法运行，请仔细阅读 `uv tool` 输出信息，按提示将工具添加到系统 PATH。

> [!NOTE]
> 项目首次运行还会自动下载部分小模型。如网络访问 HuggingFace 较慢，建议提前设置镜像：
>
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```

### 4. 检测 GPU 加速

如需诊断环境、查看识别到的 GPU，可运行内置工具：

```bash
uv run tools/gpu_check.py
```

## 💻 使用说明

### 🌐 Web 演示

```bash
# IndexTTS-2.5（默认）
uv run webui.py

# IndexTTS-2
uv run webui.py --version 2 --model_dir ./checkpoints_2
```

浏览器访问 `http://127.0.0.1:7860` 查看演示。

可通过命令行参数开启 BF16（IndexTTS-2.5）/ FP16（IndexTTS-2）推理（降低显存占用）、
DeepSpeed 加速、CUDA 内核编译加速等。运行以下命令查看所有可用选项：

```bash
uv run webui.py -h
```

> [!IMPORTANT]
> 使用 **FP16/BF16**（半精度）推理非常有益：推理更快、显存占用更低，质量损失极小。
>
> **DeepSpeed** *可能*在部分系统上加速推理，但也可能变慢，效果取决于具体硬件、驱动及操作系统，
> 建议分别开启和关闭测试，找到最适合自己环境的配置。
>
> 注意：所有 `uv` 命令会**自动激活**对应项目的虚拟环境。请*不要*手动激活环境后再运行
> `uv` 命令，否则可能导致依赖冲突！

### 🚀 使用 vLLM 部署

生产环境部署请参考 [IndexTTS 的 vLLM 部署方案](https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5)。

### 📝 Python 脚本调用

运行脚本时请使用 `uv run <file.py>`，保证程序在 uv 创建的虚拟环境下运行。
部分情况可能需要将当前目录加入 `PYTHONPATH`：

```bash
# IndexTTS2
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2.py

# IndexTTS2.5
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2_5.py \
  --cfg_path checkpoints/config.yaml \
  --model_dir checkpoints \
  --text "Hello world" \
  --lang EN
```

#### 0. 初始化 IndexTTS

```python
# IndexTTS2
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints_2/config.yaml", model_dir="checkpoints_2", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)

# IndexTTS2.5
from indextts.infer_v2_5 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True)
```

#### 1. 单一参考音频音色克隆

```python
text = "Translate for me, what is a surprise!"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)

# IndexTTS2.5（多语言，需指定语言）
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="EN", output_path="gen.wav", verbose=True)
```

#### 2. 使用独立的情感参考音频控制情感

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, lang="ZH", output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)
```

#### 3. 通过 `emo_alpha` 调节情感强度

指定情感参考音频时，可通过 `emo_alpha` 调节情感对输出的影响强度。
有效范围 `0.0 - 1.0`，默认值为 `1.0`（100%）。

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", lang="ZH", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)
```

#### 4. 使用情感向量控制情感

也可以不使用情感参考音频，直接提供 8 维情感向量，按以下顺序指定各情感强度：
`[高兴, 愤怒, 悲伤, 害怕, 厌恶, 忧郁, 惊讶, 平静]`。
可使用 `use_random` 参数开启推理随机性（默认 `False`，设为 `True` 开启随机采样）。

> [!NOTE]
> 开启随机采样会降低音色克隆的还原度。

```python
text = "对不起嘛！我的记性真的不太好，但是和你在一起的事情，我都会努力记住的~"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_09.wav', text=text, output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_09.wav', text=text, lang="ZH", output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)
```

#### 5. 根据文本自动生成情感（`use_emo_text`）

开启 `use_emo_text` 后，输入文本会自动转换为情感向量。
建议将 `emo_alpha` 设为 0.6 左右（或更低），以获得更自然的语音效果。
可使用 `use_random` 开启随机性（默认 `False`）。

```python
text = "快躲起来！是他要来了！他要来抓我们了！"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```

#### 6. 使用显式情感描述文本（`emo_text`）

通过 `emo_text` 参数直接提供情感描述文本，情感文本会自动转换为情感向量，
从而实现文本内容与情感描述的分别控制：

```python
text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
```

#### 7. 语速控制（`duration_factor`）

取值大于 `1.0` 时语速变慢，小于 `1.0` 时语速变快。默认值为 `1.0`（正常语速）。
有效范围 `0.5 - 2.0`。

```python
text = "大家好，欢迎来到IndexTTS的语速控制演示。"

# IndexTTS2.5
# 放慢（1.2 倍时长）
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="ZH", output_path="gen_slow.wav", duration_factor=1.2, verbose=True)

# 加快（0.8 倍时长）
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="ZH", output_path="gen_fast.wav", duration_factor=0.8, verbose=True)
```

### 🗣️ 发音控制

**IndexTTS2.5 —— 拼音 / CMU 音素 / 日语假名：**

IndexTTS2.5 支持以下字符替换写法，具有更好的指令跟随能力。
完整合法拼音列表请参考 `checkpoints/pinyin.vocab`；英文音素请参考
[CMU 词典](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b)。

```
他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。

He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details of the contract.

彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
```

**IndexTTS2 —— 拼音：**

IndexTTS2 支持中文字符与拼音混合建模。如需精确的发音控制，请输入包含特定拼音标注的
文本来触发拼音控制功能。需要注意：拼音控制并非对所有声母韵母组合都生效，系统仅支持
中文合法拼音，具体可参考 `checkpoints/pinyin.vocab` 文件。

```
之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。
```

### 🕰️ IndexTTS-1.5（旧版）

如需使用旧的 IndexTTS1 模型，可以 import 旧模块：

```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")
voice = "examples/voice_07.wav"
text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, 'gen.wav')
```

详细信息见 [README_INDEXTTS_1_5](../archive/README_INDEXTTS_1_5.md)，
或访问 IndexTTS1 仓库 [index-tts:v1.5.0](https://github.com/index-tts/index-tts/tree/v1.5.0)。

## 📊 评测结果

**表 1：基于 CV3-Eval 测试集的零样本 TTS 评测结果**（阿拉伯语使用内部测试集）。†为原始论文引用结果。

<table>
<thead>
<tr>
<th rowspan="2">模型</th>
<th rowspan="2">参数量</th>
<th colspan="2">zh</th>
<th colspan="2">en</th>
<th colspan="2">es</th>
<th colspan="2">ja</th>
<th colspan="2">ar</th>
<th colspan="2">平均</th>
</tr>
<tr>
<th>WER↓</th><th>SS↑</th>
<th>WER↓</th><th>SS↑</th>
<th>WER↓</th><th>SS↑</th>
<th>WER↓</th><th>SS↑</th>
<th>WER↓</th><th>SS↑</th>
<th>WER↓</th><th>SS↑</th>
</tr>
</thead>
<tbody>
<tr><td>VoxCPM2</td><td>2B</td><td>3.88</td><td>74.99</td><td>5.13</td><td>71.57</td><td>5.49</td><td>74.67</td><td>6.69</td><td>72.90</td><td>14.94</td><td>65.99</td><td>7.22</td><td>72.02</td></tr>
<tr><td>OmniVoice</td><td>0.8B</td><td>3.41</td><td>72.99</td><td>3.62</td><td>70.13</td><td>3.52</td><td>74.14</td><td>5.38</td><td>70.49</td><td>17.88</td><td>64.22</td><td>6.76</td><td>70.39</td></tr>
<tr><td>Moss-TTS 1.5</td><td>8B</td><td>4.02</td><td>72.68</td><td>4.45</td><td>67.46</td><td>3.83</td><td>71.75</td><td>10.97</td><td>68.71</td><td>23.71</td><td>62.21</td><td>9.40</td><td>68.56</td></tr>
<tr><td>CosyVoice3-0.5B</td><td>0.5B</td><td>3.84</td><td>80.01</td><td>4.88</td><td>74.16</td><td>4.04</td><td>78.85</td><td>-</td><td>76.36</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>CosyVoice3-1.5B</td><td>1.5B</td><td>3.91†</td><td>-</td><td>4.99†</td><td>-</td><td>4.47†</td><td>-</td><td>7.57†</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>FireRedTTS-2</td><td>1.5B</td><td>8.22</td><td>68.10</td><td>14.92</td><td>56.93</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>Fish Audio S2 Pro</td><td>4B</td><td>3.62</td><td>67.79</td><td>3.83</td><td>61.66</td><td>2.93</td><td>67.44</td><td>5.15</td><td>66.15</td><td>14.15</td><td>59.43</td><td>5.94</td><td>64.49</td></tr>
<tr><td>Qwen3-TTS</td><td>1.7B</td><td>3.27</td><td>73.02</td><td>5.06</td><td>67.17</td><td>2.87</td><td>73.17</td><td>5.89</td><td>70.18</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td><b>IndexTTS2.5</b></td><td>0.8B</td><td>4.36</td><td>77.10</td><td>5.12</td><td>68.06</td><td>3.75</td><td>76.39</td><td>5.66</td><td>74.62</td><td>14.88</td><td>69.74</td><td>6.75</td><td>73.18</td></tr>
<tr><td><b>IndexTTS2.5-RL</b></td><td>0.8B</td><td>3.93</td><td>77.92</td><td>3.89</td><td>67.79</td><td>3.33</td><td>76.68</td><td>5.30</td><td>75.41</td><td>13.58</td><td>70.36</td><td>6.00</td><td>73.63</td></tr>
</tbody>
</table>

**表 2：基于 CV3-Eval 测试集的跨语言 TTS 评测**（中文提示 → 目标语言，阿拉伯语使用内部测试集）。

<table>
<thead>
<tr>
<th rowspan="2">模型</th>
<th rowspan="2">参数量</th>
<th colspan="2">zh→en</th>
<th colspan="2">zh→es</th>
<th colspan="2">zh→ja</th>
<th colspan="2">zh→ar</th>
<th colspan="2">平均</th>
</tr>
<tr>
<th>WER↓</th><th>SS↑</th>
<th>WER↓</th><th>SS↑</th>
<th>WER↓</th><th>SS↑</th>
<th>WER↓</th><th>SS↑</th>
<th>WER↓</th><th>SS↑</th>
</tr>
</thead>
<tbody>
<tr><td>VoxCPM2</td><td>2B</td><td>4.48</td><td>64.25</td><td>16.38</td><td>64.89</td><td>11.84</td><td>71.54</td><td>11.09</td><td>67.62</td><td>10.95</td><td>67.08</td></tr>
<tr><td>OmniVoice</td><td>0.8B</td><td>3.74</td><td>64.91</td><td>5.84</td><td>62.08</td><td>9.09</td><td>69.06</td><td>19.80</td><td>65.27</td><td>9.62</td><td>65.33</td></tr>
<tr><td>Moss-TTS 1.5</td><td>8B</td><td>6.13</td><td>59.23</td><td>4.32</td><td>56.63</td><td>11.52</td><td>65.54</td><td>17.03</td><td>62.93</td><td>9.75</td><td>61.08</td></tr>
<tr><td>CosyVoice3-0.5B</td><td>0.5B</td><td>3.23</td><td>62.79</td><td>4.58</td><td>64.04</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>CosyVoice3-1.5B</td><td>1.5B</td><td>4.32</td><td>-</td><td>-</td><td>-</td><td>13.70</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>FireRedTTS-2</td><td>1.5B</td><td>9.34</td><td>53.19</td><td>12.25</td><td>58.31</td><td>19.05</td><td>64.12</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>Fish Audio S2 Pro</td><td>4B</td><td>4.14</td><td>55.89</td><td>4.46</td><td>55.57</td><td>10.48</td><td>61.74</td><td>14.49</td><td>59.80</td><td>8.39</td><td>58.25</td></tr>
<tr><td>Qwen3-TTS</td><td>1.7B</td><td>5.74</td><td>63.04</td><td>5.15</td><td>68.02</td><td>36.09</td><td>65.71</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td><b>IndexTTS2.5</b></td><td>0.8B</td><td>3.62</td><td>63.83</td><td>5.17</td><td>65.48</td><td>6.57</td><td>74.16</td><td>9.51</td><td>71.02</td><td>6.22</td><td>68.62</td></tr>
<tr><td><b>IndexTTS2.5-RL</b></td><td>0.8B</td><td>3.55</td><td>67.47</td><td>4.86</td><td>64.47</td><td>6.38</td><td>75.82</td><td>9.89</td><td>73.05</td><td>6.17</td><td>70.20</td></tr>
</tbody>
</table>

## ⚡ 推理速度

RTF（墙钟时间 / 生成音频时长，越低越快），NVIDIA RTX 4090，`kv_cache=True`。

| 文本 | 2.0 fp16 | 2.0 fp32 | 2.5 bf16 | 2.5 fp32 |
|---|---|---|---|---|
| 7 字 | 0.4004 | 0.3748 | 0.2871 | 0.2547 |
| 16 字 | 0.3322 | 0.3389 | 0.2155 | 0.1981 |
| 28 字 | 0.3257 | 0.3480 | 0.2065 | 0.1927 |
| 80 字 | 0.3229 | 0.3754 | 0.1997 | 0.2060 |
| 200 字 | 0.3244 | 0.3990 | 0.1997 | 0.2144 |
| **overall** | **0.3257** | **0.3748** | **0.2065** | **0.2060** |

## 🤝 社区与联系方式

- **QQ 群：** 663272642（4 群）、1013410623（5 群）
- **Discord：** https://discord.gg/uT32E7KDmy
- **邮箱：** indexspeech@bilibili.com

欢迎加入我们的社区！🌏 欢迎大家交流讨论！

> [!CAUTION]
> 感谢大家对 bilibili IndexTTS 项目的支持与关注！
> 请注意，目前由核心团队直接维护的**官方渠道仅有**: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)。
> ***其他任何网站或服务均非官方提供***，我们对其内容及安全性、准确性和及时性不作任何担保。
> 为了保障您的权益，建议通过上述官方渠道获取 bilibili IndexTTS 项目的最新进展与更新。

商业合作请联系 <u>indexspeech@bilibili.com</u>。

## 📚 论文引用

🌟 如果本项目对您有帮助，请为我们点 star 并引用论文。

IndexTTS2.5:

```bibtex
@misc{li2026indextts25technicalreport,
      title={IndexTTS 2.5 Technical Report},
      author={Yunpei Li and Xun Zhou and Jinchao Wang and Lu Wang and Yong Wu and Siyi Zhou and Yiquan Zhou and Yining Wang and Yaogen Yang and Zhetao Hu and Shiyao Duan and Jiacheng Xu and Bin Xia and Jingchen Shu},
      year={2026},
      eprint={2601.03888},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.03888},
}
```

IndexTTS2:

```bibtex
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou and Yiquan Zhou and Yi He and Xun Zhou and Jinchao Wang and Wei Deng and Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```

IndexTTS:

```bibtex
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng and Siyi Zhou and Jingchen Shu and Jinchao Wang and Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025},
  doi={10.48550/arXiv.2502.05512},
  url={https://arxiv.org/abs/2502.05512}
}
```

## 🙏 致谢

1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)
6. [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
7. [seed-vc](https://github.com/Plachtaa/seed-vc)

## 📄 许可证

本项目基于 [bilibili 模型使用许可协议](../LICENSE) 发布。
使用前请同时阅读[免责声明](../DISCLAIMER)。
