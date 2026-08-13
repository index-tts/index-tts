<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/indextts_icon_dark.png"/>
  <img src="../assets/indextts_icon_light.png" width="300"/>
</picture>

**Un sistema de texto a voz zero-shot, controlable y eficiente, de nivel industrial**

[简体中文](README_zh.md) | [English](../README.md) | [日本語](README_ja.md) | Español | [العربية](README_ar.md)

[![GitHub Stars](https://img.shields.io/github/stars/index-tts/index-tts?style=flat&logo=github)](https://github.com/index-tts/index-tts/stargazers)
[![arXiv](https://img.shields.io/badge/arXiv-2601.03888-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.03888)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/uT32E7KDmy)

</div>

IndexTTS es un sistema de texto a voz zero-shot que clona una voz a partir de un
único clip de audio de referencia. La última versión, **IndexTTS-2.5**, admite
chino, inglés, japonés, español y árabe, con control de emociones de grano
fino, control de la velocidad de habla, control de la pronunciación (Pinyin /
fonemas CMU / Kana japonés) y una inferencia más rápida que IndexTTS-2.

---

## 🗂️ Catálogo de modelos

| Modelo | Demos | Artículo | ModelScope | HuggingFace |
| :--- | :---: | :---: | :---: | :---: |
| **IndexTTS-2.5** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/index-tts2-5.github.io/) [![Studio](https://img.shields.io/badge/Studio-ModelScope-purple?logo=modelscope)](https://modelscope.cn/studios/IndexTeam/IndexTTS-2.5) [![Space](https://img.shields.io/badge/Space-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/IndexTeam/IndexTTS-2.5-Demo) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2601.03888) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-2.5) |
| **IndexTTS-2** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/index-tts2.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2506.21619) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-2) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-2) |
| **IndexTTS-1.5** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2502.05512) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-1.5) |
| **IndexTTS** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2502.05512) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/Index-TTS) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/Index-TTS) |

## 📣 Novedades

- `2026/08/10` 🔥 Lanzamos **IndexTTS-2.5**
  - Ahora admite chino, inglés, japonés, español y árabe, con una inferencia más rápida que IndexTTS-2, manteniendo las capacidades de síntesis multilingüe y de desacoplamiento timbre-emoción.
  - Mayor controlabilidad del Pinyin chino, los fonemas CMU en inglés y el Kana japonés.
  - Control de la velocidad de habla mediante `duration_factor` (0.5x–2.0x de duración).
- `2025/09/08` 🔥 Lanzamos **IndexTTS-2**
  - El primer modelo TTS autorregresivo con control preciso de la duración de la síntesis, compatible con los modos controlable y no controlable. <i>Esta funcionalidad aún no está habilitada en esta versión.</i>
  - Síntesis de voz con emociones altamente expresivas, con control de emociones a través de múltiples modalidades de entrada.
- `2025/05/14` 🔥 Lanzamos **IndexTTS-1.5**, que mejora significativamente la estabilidad del modelo y su rendimiento en inglés.
- `2025/03/25` 🔥 Lanzamos **IndexTTS-1.0** con los pesos del modelo y el código de inferencia.
- `2025/02/12` 🎉 Enviamos nuestro artículo a arXiv y publicamos nuestras demos y conjuntos de prueba.

## 🎬 Demos

<div align="center">

**IndexTTS-2.5: El futuro de la voz, generándose ahora**

[![IndexTTS2.5 Demo](../assets/index2.5_video_cover.png)](https://www.bilibili.com/video/BV1uvMk6ZEdK/)

**IndexTTS-2: El futuro de la voz, generándose ahora**

[![IndexTTS2 Demo](../assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)

</div>

## 🚀 Primeros pasos

### 1. Requisitos previos

Asegúrate de tener [git](https://git-scm.com/downloads) instalado y luego descarga
este repositorio:

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
```

Los archivos de audio de ejemplo se descargan bajo demanda desde
HuggingFace/ModelScope en la primera ejecución, por lo que Git LFS ya no es
necesario.

### 2. Instalar dependencias

Usamos [uv](https://docs.astral.sh/uv/getting-started/installation/) para
gestionar el entorno de dependencias del proyecto. Es **obligatorio** para una
instalación fiable:

```bash
pip install -U uv  # or see the link above for other install methods
```

```bash
uv sync --all-extras
```

Esto crea automáticamente un directorio de proyecto `.venv` e instala las
versiones correctas de Python y de todas las dependencias necesarias.

Si la descarga es lenta, usa un espejo local, por ejemplo uno de estos espejos
en China:

```bash
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"

uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

> [!TIP]
> **Funciones adicionales disponibles:**
>
> - `--all-extras`: Añade automáticamente *todas* las funciones adicionales
>   enumeradas a continuación. Puedes quitar este indicador si quieres
>   personalizar tu instalación.
> - `--extra webui`: Añade compatibilidad con la WebUI (recomendado).
> - `--extra deepspeed`: Añade compatibilidad con DeepSpeed (puede acelerar la
>   inferencia en algunos sistemas).

> [!IMPORTANT]
> **Windows:** DeepSpeed puede ser difícil de instalar. Puedes omitirlo
> quitando el indicador `--all-extras` y añadiendo manualmente los demás
> indicadores de funciones.
>
> **Linux/Windows:** Si ves un error de CUDA durante la instalación, asegúrate
> de que el [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) de NVIDIA
> versión **12.8** (o superior) esté instalado en tu sistema.

### 3. Descargar los modelos

Descarga los modelos necesarios mediante [uv tool](https://docs.astral.sh/uv/guides/tools/#installing-tools):

Mediante `huggingface-cli`:

```bash
uv tool install "huggingface-hub"

# IndexTTS-2.5
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints

# IndexTTS-2
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints_2
```

O mediante `modelscope`:

```bash
uv tool install "modelscope"

# IndexTTS-2.5
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints

# IndexTTS-2
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints_2
```

> [!IMPORTANT]
> Si los comandos anteriores no están disponibles, lee atentamente la salida de
> `uv tool`: te indicará cómo añadir las herramientas al PATH de tu sistema.

> [!NOTE]
> Algunos modelos pequeños se descargan automáticamente en la primera
> ejecución. Si tu red accede lentamente a HuggingFace, configura un espejo
> antes de ejecutar el código:
>
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```

### 4. Comprobar la aceleración por GPU

Para diagnosticar tu entorno y ver qué GPU se detectan, usa la utilidad
incluida:

```bash
uv run tools/gpu_check.py
```

## 💻 Uso

### 🌐 Demo web

```bash
# IndexTTS-2.5 (default)
uv run webui.py

# IndexTTS-2
uv run webui.py --version 2 --model_dir ./checkpoints_2
```

Abre tu navegador y visita `http://127.0.0.1:7860` para ver la demo.

Puedes ajustar la configuración para habilitar la inferencia en BF16
(IndexTTS-2.5) / FP16 (IndexTTS-2) (menor uso de VRAM), la aceleración con
DeepSpeed, núcleos CUDA compilados para mayor velocidad, etc. Todas las
opciones disponibles se pueden ver con:

```bash
uv run webui.py -h
```

> [!IMPORTANT]
> La inferencia en **FP16/BF16** (media precisión) es más rápida y usa menos
> VRAM, con una pérdida de calidad muy pequeña.
>
> **DeepSpeed** *puede* acelerar la inferencia en algunos sistemas, pero
> también podría hacerla más lenta: depende de tu hardware, controladores y
> sistema operativo. Pruébalo de ambas formas.
>
> Todos los comandos `uv` **activan automáticamente** el entorno virtual
> correcto del proyecto. *No* actives manualmente ningún entorno antes de
> ejecutar comandos `uv`, ya que eso puede causar conflictos de dependencias.

### 🚀 Servicio con vLLM

Para el despliegue en producción, consulta la [receta de vLLM para IndexTTS](https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5).

### 📝 API de Python

Para ejecutar scripts, usa `uv run <file.py>` de modo que el código se ejecute
dentro del entorno de `uv`. Es posible que también necesites añadir el
directorio actual a `PYTHONPATH`:

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

#### 0. Inicializar IndexTTS

```python
# IndexTTS2
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints_2/config.yaml", model_dir="checkpoints_2", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)

# IndexTTS2.5
from indextts.infer_v2_5 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True)
```

#### 1. Clonación de voz con un único audio de referencia

```python
text = "Translate for me, what is a surprise!"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)

# IndexTTS2.5 (multilingual, with language selection)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="EN", output_path="gen.wav", verbose=True)
```

#### 2. Control de emociones con un audio de referencia emocional independiente

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, lang="ZH", output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)
```

#### 3. Ajustar la intensidad de la emoción con `emo_alpha`

Cuando se especifica un audio de referencia emocional, `emo_alpha` ajusta
cuánto afecta al resultado. Rango válido: `0.0 - 1.0`, valor por defecto:
`1.0` (100%).

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", lang="ZH", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)
```

#### 4. Control de emociones con un vector de emociones

Puedes omitir el audio de referencia emocional y, en su lugar, proporcionar una
lista de 8 valores flotantes que especifican la intensidad de cada emoción, en
el orden
`[alegre, enfadado, triste, asustado, disgustado, melancólico, sorprendido, tranquilo]`.
Usa `use_random` para introducir estocasticidad durante la inferencia
(valor por defecto: `False`).

> [!NOTE]
> Activar el muestreo aleatorio reduce la fidelidad de la clonación de voz.

```python
text = "对不起嘛！我的记性真的不太好，但是和你在一起的事情，我都会努力记住的~"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_09.wav', text=text, output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_09.wav', text=text, lang="ZH", output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)
```

#### 5. Control de emociones a partir del propio texto (`use_emo_text`)

Activa `use_emo_text` para convertir automáticamente tu guion `text` en
vectores de emociones. Se recomienda un `emo_alpha` en torno a 0.6 (o menor)
para una voz más natural. Se puede introducir aleatoriedad con `use_random`
(valor por defecto: `False`).

```python
text = "快躲起来！是他要来了！他要来抓我们了！"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```

#### 6. Control de emociones con una descripción explícita de la emoción (`emo_text`)

Proporciona una descripción textual específica de la emoción mediante
`emo_text`, que se convierte en vectores de emociones, lo que te permite
controlar por separado el guion del texto y la descripción de la emoción:

```python
text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
```

#### 7. Control de la velocidad de habla (`duration_factor`)

Un valor mayor que `1.0` ralentiza el habla; un valor menor que `1.0` la
acelera. Valor por defecto: `1.0` (velocidad normal). Rango válido:
`0.5 - 2.0`.

```python
text = "大家好，欢迎来到IndexTTS的语速控制演示。"

# IndexTTS2.5
# Slow down (1.2x duration)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="ZH", output_path="gen_slow.wav", duration_factor=1.2, verbose=True)

# Speed up (0.8x duration)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="ZH", output_path="gen_fast.wav", duration_factor=0.8, verbose=True)
```

### 🗣️ Control de la pronunciación

**IndexTTS2.5 — Pinyin / fonemas CMU / Kana japonés:**

IndexTTS2.5 admite estas sustituciones de caracteres con una mejor capacidad
de seguimiento de instrucciones. Para ver la lista completa de entradas
válidas, consulta `checkpoints/pinyin.vocab` para el Pinyin y el
[diccionario CMU](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b)
para los fonemas en inglés.

```
他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。

He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details of the contract.

彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
```

**IndexTTS2 — Pinyin:**

IndexTTS2 admite el modelado mixto de caracteres chinos y Pinyin. Para activar
el control por Pinyin, proporciona texto con anotaciones específicas de Pinyin.
Ten en cuenta que el control por Pinyin no funciona para todas las
combinaciones posibles de consonante y vocal; solo se admiten los casos de
Pinyin chino válidos (consulta `checkpoints/pinyin.vocab`).

```
之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。
```

### 🕰️ IndexTTS-1.5 (heredado)

También puedes usar el modelo anterior IndexTTS1 importando un módulo
diferente:

```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")
voice = "examples/voice_07.wav"
text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, 'gen.wav')
```

Para más detalles, consulta [README_INDEXTTS_1_5](../archive/README_INDEXTTS_1_5.md),
o visita el repositorio de IndexTTS1 en [index-tts:v1.5.0](https://github.com/index-tts/index-tts/tree/v1.5.0).

## 📊 Evaluación

**Tabla 1: TTS zero-shot en CV3-Eval** (el árabe usa un conjunto de prueba interno). †Citado del artículo original.

<table>
<thead>
<tr>
<th rowspan="2">Modelo</th>
<th rowspan="2">Parámetros</th>
<th colspan="2">zh</th>
<th colspan="2">en</th>
<th colspan="2">es</th>
<th colspan="2">ja</th>
<th colspan="2">ar</th>
<th colspan="2">Prom.</th>
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

**Tabla 2: TTS multilingüe en CV3-Eval** (prompt en chino → idioma de destino; el árabe usa un conjunto de prueba interno).

<table>
<thead>
<tr>
<th rowspan="2">Modelo</th>
<th rowspan="2">Parámetros</th>
<th colspan="2">zh→en</th>
<th colspan="2">zh→es</th>
<th colspan="2">zh→ja</th>
<th colspan="2">zh→ar</th>
<th colspan="2">Prom.</th>
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

## 🤝 Comunidad y contacto

- **Grupos de QQ:** 663272642 (n.º 4), 1013410623 (n.º 5)
- **Discord:** https://discord.gg/uT32E7KDmy
- **Correo electrónico:** indexspeech@bilibili.com

¡Te invitamos a unirte a nuestra comunidad! 🌏 欢迎大家来交流讨论！

> [!CAUTION]
> ¡Gracias por tu apoyo al proyecto IndexTTS de bilibili!
> Ten en cuenta que el **único canal oficial** mantenido por el equipo central es: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts).
> ***Cualquier otro sitio web o servicio no es oficial*** y no podemos garantizar su seguridad, exactitud ni actualidad.
> Para conocer las últimas novedades, consulta siempre este repositorio oficial.

Para uso comercial y colaboraciones, contacta con <u>indexspeech@bilibili.com</u>.

## 📚 Citación

🌟 Si nuestro trabajo te resulta útil, déjanos una estrella y cita nuestros artículos.

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

## 🙏 Agradecimientos

1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)
6. [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
7. [seed-vc](https://github.com/Plachtaa/seed-vc)

## 📄 Licencia

Este proyecto se publica bajo el [Acuerdo de Licencia de Uso de Modelos de bilibili](../LICENSE).
Lee también el [DESCARGO DE RESPONSABILIDAD](../DISCLAIMER) antes de usarlo.
