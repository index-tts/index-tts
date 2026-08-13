<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/indextts_icon_dark.png"/>
  <img src="../assets/indextts_icon_light.png" width="300"/>
</picture>

**産業レベルの制御可能で効率的なゼロショット・テキスト読み上げシステム**

[简体中文](README_zh.md) | [English](../README.md) | 日本語 | [Español](README_es.md) | [العربية](README_ar.md)

[![GitHub Stars](https://img.shields.io/github/stars/index-tts/index-tts?style=flat&logo=github)](https://github.com/index-tts/index-tts/stargazers)
[![arXiv](https://img.shields.io/badge/arXiv-2601.03888-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.03888)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/uT32E7KDmy)

</div>

IndexTTS は、1つの参照音声クリップから声をクローンするゼロショット・テキスト
読み上げシステムです。最新リリースの **IndexTTS-2.5** は、中国語、英語、日本語、
スペイン語、アラビア語をサポートし、きめ細かな感情コントロール、話速コントロール、
発音コントロール（ピンイン / CMU 音素 / 日本語の仮名）を備え、IndexTTS-2 よりも
高速な推論を実現しています。

---

## 🗂️ モデル一覧

| モデル | デモ | 論文 | ModelScope | HuggingFace |
| :--- | :---: | :---: | :---: | :---: |
| **IndexTTS-2.5** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/index-tts2-5.github.io/) [![Studio](https://img.shields.io/badge/Studio-ModelScope-purple?logo=modelscope)](https://modelscope.cn/studios/IndexTeam/IndexTTS-2.5) [![Space](https://img.shields.io/badge/Space-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/IndexTeam/IndexTTS-2.5-Demo) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2601.03888) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-2.5) |
| **IndexTTS-2** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/index-tts2.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2506.21619) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-2) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-2) |
| **IndexTTS-1.5** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2502.05512) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-1.5) |
| **IndexTTS** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2502.05512) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/Index-TTS) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/Index-TTS) |

## 📣 ニュース

- `2026/08/10` 🔥 **IndexTTS-2.5** をリリースしました
  - 中国語、英語、日本語、スペイン語、アラビア語をサポートし、IndexTTS-2 よりも高速な推論を実現しながら、言語横断的および音色・感情分離の能力を維持しています。
  - 中国語ピンイン、英語 CMU 音素、日本語仮名の制御性が向上しました。
  - `duration_factor` による話速コントロール（0.5倍～2.0倍の長さ）。
- `2025/09/08` 🔥 **IndexTTS-2** をリリースしました
  - 精密な合成時間制御を備えた初の自己回帰型 TTS モデルで、制御可能モードと非制御モードの両方をサポートします。<i>この機能は本リリースではまだ有効になっていません。</i>
  - 高い表現力を持つ感情音声合成。複数の入力モダリティによる感情コントロールが可能です。
- `2025/05/14` 🔥 **IndexTTS-1.5** をリリースしました。モデルの安定性と英語での性能が大幅に向上しています。
- `2025/03/25` 🔥 **IndexTTS-1.0** をモデルウェイトおよび推論コードとともにリリースしました。
- `2025/02/12` 🎉 論文を arXiv に投稿し、デモとテストセットを公開しました。

## 🎬 デモ

<div align="center">

**IndexTTS-2.5: The Future of Voice, Now Generating**

[![IndexTTS2.5 Demo](../assets/index2.5_video_cover.png)](https://www.bilibili.com/video/BV1uvMk6ZEdK/)

**IndexTTS-2: The Future of Voice, Now Generating**

[![IndexTTS2 Demo](../assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)

</div>

## 🚀 はじめに

### 1. 前提条件

[git](https://git-scm.com/downloads) がインストールされていることを確認してから、
このリポジトリをダウンロードしてください：

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
```

サンプル音声ファイルは初回実行時に HuggingFace/ModelScope からオンデマンドで
ダウンロードされるため、Git LFS は不要になりました。

### 2. 依存関係のインストール

プロジェクトの依存環境の管理には [uv](https://docs.astral.sh/uv/getting-started/installation/) を使用しています。確実なインストールのために **必須** です：

```bash
pip install -U uv  # or see the link above for other install methods
```

```bash
uv sync --all-extras
```

これにより `.venv` プロジェクトディレクトリが自動的に作成され、正しいバージョンの
Python と必要なすべての依存関係がインストールされます。

ダウンロードが遅い場合はローカルミラーを使用してください。例えば中国国内の
以下のミラーが利用できます：

```bash
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"

uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

> [!TIP]
> **利用可能な追加機能：**
>
> - `--all-extras`: 以下に挙げる *すべての* 追加機能を自動的に追加します。
>   インストール内容をカスタマイズしたい場合は、このフラグを外してください。
> - `--extra webui`: WebUI サポートを追加します（推奨）。
> - `--extra deepspeed`: DeepSpeed サポートを追加します（一部のシステムで推論が
>   高速化する場合があります）。

> [!IMPORTANT]
> **Windows:** DeepSpeed のインストールが難しい場合があります。`--all-extras`
> フラグを外し、他の機能フラグを個別に追加することでスキップできます。
>
> **Linux/Windows:** インストール中に CUDA エラーが表示された場合は、NVIDIA の
> [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) バージョン **12.8**
> （以降）がシステムにインストールされていることを確認してください。

### 3. モデルのダウンロード

[uv tool](https://docs.astral.sh/uv/guides/tools/#installing-tools) を使って必要なモデルをダウンロードします：

`huggingface-cli` を使う場合：

```bash
uv tool install "huggingface-hub"

# IndexTTS-2.5
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints

# IndexTTS-2
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints_2
```

または `modelscope` を使う場合：

```bash
uv tool install "modelscope"

# IndexTTS-2.5
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints

# IndexTTS-2
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints_2
```

> [!IMPORTANT]
> 上記のコマンドが利用できない場合は、`uv tool` の出力をよく確認してください。
> ツールをシステムの PATH に追加する方法が表示されます。

> [!NOTE]
> 一部の小さなモデルは初回実行時に自動的にダウンロードされます。ネットワークから
> HuggingFace へのアクセスが遅い場合は、コードを実行する前にミラーを設定してください：
>
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```

### 4. GPU アクセラレーションの確認

環境を診断し、どの GPU が検出されているかを確認するには、付属のユーティリティを
使用してください：

```bash
uv run tools/gpu_check.py
```

## 💻 使い方

### 🌐 Web デモ

```bash
# IndexTTS-2.5 (default)
uv run webui.py

# IndexTTS-2
uv run webui.py --version 2 --model_dir ./checkpoints_2
```

ブラウザを開いて `http://127.0.0.1:7860` にアクセスするとデモが表示されます。

設定を調整して、BF16（IndexTTS-2.5）/ FP16（IndexTTS-2）推論（VRAM 使用量の削減）、
DeepSpeed アクセラレーション、高速化のためのコンパイル済み CUDA カーネルなどを
有効にできます。利用可能なすべてのオプションは以下で確認できます：

```bash
uv run webui.py -h
```

> [!IMPORTANT]
> **FP16/BF16**（半精度）推論は高速で VRAM 使用量も少なく、品質の低下は
> ごくわずかです。
>
> **DeepSpeed** は一部のシステムで推論を高速化する*可能性*がありますが、逆に
> 遅くなる場合もあります。ハードウェア、ドライバ、OS に依存します。両方を
> 試してみてください。
>
> すべての `uv` コマンドは、プロジェクトごとの正しい仮想環境を**自動的に
> アクティベート**します。`uv` コマンドを実行する前に手動で環境をアクティベート
> *しないでください*。依存関係の競合を引き起こす可能性があります。

### 🚀 vLLM によるサービング

本番環境へのデプロイについては、[IndexTTS 向け vLLM レシピ](https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5)を参照してください。

### 📝 Python API

スクリプトを実行するには `uv run <file.py>` を使用し、コードが `uv` 環境内で
実行されるようにしてください。カレントディレクトリを `PYTHONPATH` に追加する
必要がある場合もあります：

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

#### 0. IndexTTS の初期化

```python
# IndexTTS2
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints_2/config.yaml", model_dir="checkpoints_2", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)

# IndexTTS2.5
from indextts.infer_v2_5 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True)
```

#### 1. 1つの参照音声によるボイスクローニング

```python
text = "Translate for me, what is a surprise!"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)

# IndexTTS2.5 (multilingual, with language selection)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="EN", output_path="gen.wav", verbose=True)
```

#### 2. 別の感情参照音声による感情コントロール

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, lang="ZH", output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)
```

#### 3. `emo_alpha` による感情強度の調整

感情参照音声が指定されている場合、`emo_alpha` でそれが出力に与える影響の
大きさを調整できます。有効範囲：`0.0 - 1.0`、デフォルト：`1.0`（100%）。

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", lang="ZH", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)
```

#### 4. 感情ベクトルによる感情コントロール

感情参照音声を省略し、代わりに各感情の強度を指定する 8 要素の浮動小数点数リストを
指定できます。順序は
`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`
です。`use_random` を使うと推論時に確率的な揺らぎを導入できます（デフォルト：`False`）。

> [!NOTE]
> ランダムサンプリングを有効にすると、ボイスクローニングの忠実度が低下します。

```python
text = "对不起嘛！我的记性真的不太好，但是和你在一起的事情，我都会努力记住的~"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_09.wav', text=text, output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_09.wav', text=text, lang="ZH", output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)
```

#### 5. テキスト自体からの感情コントロール（`use_emo_text`）

`use_emo_text` を有効にすると、`text` のスクリプトが自動的に感情ベクトルに
変換されます。より自然な音声にするためには、`emo_alpha` を 0.6 前後（または
それ以下）にすることを推奨します。`use_random` でランダム性を導入できます
（デフォルト：`False`）。

```python
text = "快躲起来！是他要来了！他要来抓我们了！"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```

#### 6. 明示的な感情記述による感情コントロール（`emo_text`）

`emo_text` で特定の感情記述テキストを指定すると、それが感情ベクトルに変換
されます。これにより、テキストスクリプトと感情記述を別々にコントロールできます：

```python
text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
```

#### 7. 話速コントロール（`duration_factor`）

`1.0` より大きい値を指定すると音声が遅くなり、`1.0` より小さい値を指定すると
速くなります。デフォルト：`1.0`（通常速度）。有効範囲：`0.5 - 2.0`。

```python
text = "大家好，欢迎来到IndexTTS的语速控制演示。"

# IndexTTS2.5
# Slow down (1.2x duration)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="ZH", output_path="gen_slow.wav", duration_factor=1.2, verbose=True)

# Speed up (0.8x duration)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="ZH", output_path="gen_fast.wav", duration_factor=0.8, verbose=True)
```

### 🗣️ 発音コントロール

**IndexTTS2.5 — ピンイン / CMU 音素 / 日本語仮名：**

IndexTTS2.5 は、より優れた指示追従能力により、これらの文字置換をサポートします。
有効なエントリの完全なリストについては、ピンインは `checkpoints/pinyin.vocab`、
英語音素は [CMU 辞書](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b)を参照してください。

```
他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。

He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details of the contract.

彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
```

**IndexTTS2 — ピンイン：**

IndexTTS2 は、漢字とピンインの混合モデリングをサポートします。ピンイン
コントロールを有効にするには、特定のピンイン注釈を付けたテキストを入力します。
なお、ピンインコントロールはすべての子音・母音の組み合わせで機能するわけでは
ありません。有効な中国語ピンインの場合のみサポートされます
（`checkpoints/pinyin.vocab` を参照）。

```
之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。
```

### 🕰️ IndexTTS-1.5（レガシー）

別のモジュールをインポートすることで、以前の IndexTTS1 モデルを使用することも
できます：

```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")
voice = "examples/voice_07.wav"
text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, 'gen.wav')
```

詳細については [README_INDEXTTS_1_5](../archive/README_INDEXTTS_1_5.md) を参照するか、
[index-tts:v1.5.0](https://github.com/index-tts/index-tts/tree/v1.5.0) の IndexTTS1 リポジトリをご覧ください。

## 📊 評価

**表 1: CV3-Eval におけるゼロショット TTS**（アラビア語は社内テストセットを使用）。†原論文より引用。

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Params</th>
<th colspan="2">zh</th>
<th colspan="2">en</th>
<th colspan="2">es</th>
<th colspan="2">ja</th>
<th colspan="2">ar</th>
<th colspan="2">Avg</th>
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

**表 2: CV3-Eval における言語横断 TTS**（中国語プロンプト → ターゲット言語。アラビア語は社内テストセットを使用）。

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Params</th>
<th colspan="2">zh→en</th>
<th colspan="2">zh→es</th>
<th colspan="2">zh→ja</th>
<th colspan="2">zh→ar</th>
<th colspan="2">Avg</th>
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

## 🤝 コミュニティ & お問い合わせ

- **QQ グループ:** 663272642 (No.4), 1013410623 (No.5)
- **Discord:** https://discord.gg/uT32E7KDmy
- **メール:** indexspeech@bilibili.com

ぜひコミュニティにご参加ください！🌏 皆様のご参加・ご意見をお待ちしております！

> [!CAUTION]
> bilibili IndexTTS プロジェクトをご支援いただきありがとうございます！
> コアチームがメンテナンスしている**唯一の公式チャンネル**は [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts) です。
> ***その他のウェブサイトやサービスは公式ではありません***。その安全性、正確性、適時性について当方は保証できません。
> 最新情報については、常にこの公式リポジトリをご参照ください。

商用利用および協業については、<u>indexspeech@bilibili.com</u> までお問い合わせください。

## 📚 引用

🌟 私たちの成果がお役に立ちましたら、スターを付け、論文を引用していただけると幸いです。

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

## 🙏 謝辞

1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)
6. [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
7. [seed-vc](https://github.com/Plachtaa/seed-vc)

## 📄 ライセンス

このプロジェクトは [bilibili モデル使用許諾契約](../LICENSE) の下で公開されています。
ご使用前に [DISCLAIMER](../DISCLAIMER) もお読みください。
