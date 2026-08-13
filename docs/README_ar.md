<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/indextts_icon_dark.png"/>
  <img src="../assets/indextts_icon_light.png" width="300"/>
</picture>

**نظام صناعي المستوى لتحويل النص إلى كلام بتقنية الاستدلال الصفري (Zero-Shot)، قابل للتحكم وعالي الكفاءة**

[简体中文](README_zh.md) | [English](../README.md) | [日本語](README_ja.md) | [Español](README_es.md) | العربية

[![GitHub Stars](https://img.shields.io/github/stars/index-tts/index-tts?style=flat&logo=github)](https://github.com/index-tts/index-tts/stargazers)
[![arXiv](https://img.shields.io/badge/arXiv-2601.03888-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.03888)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/uT32E7KDmy)

</div>

IndexTTS هو نظام لتحويل النص إلى كلام بتقنية الاستدلال الصفري يستنسخ الصوت من
مقطع صوتي مرجعي واحد. يدعم أحدث إصدار، **IndexTTS-2.5**، اللغات الصينية
والإنجليزية واليابانية والإسبانية والعربية، مع تحكم دقيق في المشاعر، وتحكم
في سرعة الكلام، وتحكم في النطق (البينيين / أصوات CMU / الكانا اليابانية)،
واستدلال أسرع من IndexTTS-2.

---

## 🗂️ مجموعة النماذج

| النموذج | العروض | الورقة البحثية | ModelScope | HuggingFace |
| :--- | :---: | :---: | :---: | :---: |
| **IndexTTS-2.5** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/index-tts2-5.github.io/) [![Studio](https://img.shields.io/badge/Studio-ModelScope-purple?logo=modelscope)](https://modelscope.cn/studios/IndexTeam/IndexTTS-2.5) [![Space](https://img.shields.io/badge/Space-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/IndexTeam/IndexTTS-2.5-Demo) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2601.03888) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-2.5) |
| **IndexTTS-2** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/index-tts2.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2506.21619) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-2) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-2) |
| **IndexTTS-1.5** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2502.05512) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/IndexTTS-1.5) |
| **IndexTTS** | [![Demo](https://img.shields.io/badge/Demo-Page-orange?logo=github)](https://index-tts.github.io/) | [![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2502.05512) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope)](https://modelscope.cn/models/IndexTeam/Index-TTS) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/IndexTeam/Index-TTS) |

## 📣 الأخبار

- `2026/08/10` 🔥 نطلق **IndexTTS-2.5**
  - يدعم الآن الصينية والإنجليزية واليابانية والإسبانية والعربية، مع استدلال أسرع من IndexTTS-2، مع الحفاظ على قدرات التخليق عبر اللغات وفصل نبرة الصوت عن المشاعر.
  - تحسين قابلية التحكم في البينيين الصيني وأصوات CMU الإنجليزية والكانا اليابانية.
  - تحكم في سرعة الكلام عبر `duration_factor` (مدة من 0.5x إلى 2.0x).
- `2025/09/08` 🔥 نطلق **IndexTTS-2**
  - أول نموذج TTS ذاتي الارتداد مع تحكم دقيق في مدة التخليق، يدعم الوضعين القابل للتحكم وغير القابل للتحكم. <i>هذه الميزة غير مفعّلة بعد في هذا الإصدار.</i>
  - تخليق كلام عالي التعبير العاطفي، مع التحكم في المشاعر عبر وسائط إدخال متعددة.
- `2025/05/14` 🔥 نطلق **IndexTTS-1.5**، مع تحسين كبير في استقرار النموذج وأدائه في اللغة الإنجليزية.
- `2025/03/25` 🔥 نطلق **IndexTTS-1.0** مع أوزان النموذج وكود الاستدلال.
- `2025/02/12` 🎉 قدّمنا ورقتنا البحثية إلى arXiv، وأصدرنا العروض التوضيحية ومجموعات الاختبار.

## 🎬 العروض التوضيحية

<div align="center">

**IndexTTS-2.5: مستقبل الصوت، يُولَّد الآن**

[![IndexTTS2.5 Demo](../assets/index2.5_video_cover.png)](https://www.bilibili.com/video/BV1uvMk6ZEdK/)

**IndexTTS-2: مستقبل الصوت، يُولَّد الآن**

[![IndexTTS2 Demo](../assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)

</div>

## 🚀 البدء

### 1. المتطلبات الأساسية

تأكد من تثبيت [git](https://git-scm.com/downloads)، ثم نزّل هذا المستودع:

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
```

يتم تنزيل الملفات الصوتية النموذجية عند الطلب من HuggingFace/ModelScope عند
التشغيل الأول، لذا لم يعد Git LFS مطلوبًا.

### 2. تثبيت التبعيات

نستخدم [uv](https://docs.astral.sh/uv/getting-started/installation/) لإدارة
بيئة تبعيات المشروع. وهو **مطلوب** لتثبيت موثوق:

```bash
pip install -U uv  # or see the link above for other install methods
```

```bash
uv sync --all-extras
```

يؤدي هذا تلقائيًا إلى إنشاء دليل المشروع `.venv` وتثبيت الإصدارات الصحيحة
من Python وجميع التبعيات المطلوبة.

إذا كان التنزيل بطيئًا، استخدم مرآة محلية، مثل إحدى هاتين المرآتين في الصين:

```bash
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"

uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

> [!TIP]
> **الميزات الإضافية المتاحة:**
>
> - `--all-extras`: يضيف تلقائيًا *كل* ميزة إضافية مدرجة أدناه. يمكنك إزالة
>   هذا الخيار إذا أردت تخصيص خيارات التثبيت.
> - `--extra webui`: يضيف دعم واجهة الويب WebUI (موصى به).
> - `--extra deepspeed`: يضيف دعم DeepSpeed (قد يسرّع الاستدلال على بعض
>   الأنظمة).

> [!IMPORTANT]
> **Windows:** قد يكون تثبيت DeepSpeed صعبًا. يمكنك تخطيه بإزالة خيار
> `--all-extras` وإضافة خيارات الميزات الأخرى يدويًا.
>
> **Linux/Windows:** إذا ظهر خطأ CUDA أثناء التثبيت، تأكد من تثبيت الإصدار
> **12.8** (أو أحدث) من [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
> من NVIDIA على نظامك.

### 3. تنزيل النماذج

نزّل النماذج المطلوبة عبر [uv tool](https://docs.astral.sh/uv/guides/tools/#installing-tools):

عبر `huggingface-cli`:

```bash
uv tool install "huggingface-hub"

# IndexTTS-2.5
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints

# IndexTTS-2
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints_2
```

أو عبر `modelscope`:

```bash
uv tool install "modelscope"

# IndexTTS-2.5
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints

# IndexTTS-2
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints_2
```

> [!IMPORTANT]
> إذا لم تكن الأوامر أعلاه متاحة، اقرأ بعناية مخرجات `uv tool` —
> فهي ستخبرك بكيفية إضافة الأدوات إلى PATH في نظامك.

> [!NOTE]
> يتم تنزيل بعض النماذج الصغيرة تلقائيًا عند التشغيل الأول. إذا كانت شبكتك
> بطيئة في الوصول إلى HuggingFace، اضبط مرآة قبل تشغيل الكود:
>
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```

### 4. التحقق من تسريع GPU

لتشخيص بيئتك ومعرفة وحدات GPU المكتشفة، استخدم الأداة المرفقة:

```bash
uv run tools/gpu_check.py
```

## 💻 الاستخدام

### 🌐 عرض الويب

```bash
# IndexTTS-2.5 (default)
uv run webui.py

# IndexTTS-2
uv run webui.py --version 2 --model_dir ./checkpoints_2
```

افتح متصفحك وزر `http://127.0.0.1:7860` لمشاهدة العرض.

يمكنك ضبط الإعدادات لتفعيل استدلال BF16 (IndexTTS-2.5) / FP16 (IndexTTS-2)
(استهلاك أقل لذاكرة VRAM)، وتسريع DeepSpeed، ونوى CUDA المجمّعة للسرعة،
وما إلى ذلك. يمكن الاطلاع على جميع الخيارات المتاحة عبر:

```bash
uv run webui.py -h
```

> [!IMPORTANT]
> استدلال **FP16/BF16** (نصف الدقة) أسرع ويستهلك ذاكرة VRAM أقل، مع فقدان
> ضئيل جدًا في الجودة.
>
> **DeepSpeed** *قد* يسرّع الاستدلال على بعض الأنظمة، لكنه قد يجعله أبطأ
> أيضًا — يعتمد ذلك على عتادك وتعريفاتك ونظام التشغيل. جرّب الطريقتين.
>
> جميع أوامر `uv` **تفعّل تلقائيًا** البيئة الافتراضية الصحيحة الخاصة
> بالمشروع. *لا* تفعّل أي بيئة يدويًا قبل تشغيل أوامر `uv`، لأن ذلك قد
> يسبب تعارضات في التبعيات.

### 🚀 التشغيل باستخدام vLLM

للنشر الإنتاجي، راجع [وصفة vLLM لـ IndexTTS](https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5).

### 📝 واجهة Python البرمجية

لتشغيل السكربتات، استخدم `uv run <file.py>` ليعمل الكود داخل بيئة `uv`.
قد تحتاج أيضًا إلى إضافة الدليل الحالي إلى `PYTHONPATH`:

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

#### 0. تهيئة IndexTTS

```python
# IndexTTS2
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints_2/config.yaml", model_dir="checkpoints_2", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)

# IndexTTS2.5
from indextts.infer_v2_5 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True)
```

#### 1. استنساخ الصوت من مقطع صوتي مرجعي واحد

```python
text = "Translate for me, what is a surprise!"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)

# IndexTTS2.5 (multilingual, with language selection)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="EN", output_path="gen.wav", verbose=True)
```

#### 2. التحكم في المشاعر بصوت مرجعي عاطفي منفصل

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, lang="ZH", output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)
```

#### 3. ضبط شدة المشاعر باستخدام `emo_alpha`

عند تحديد صوت مرجعي عاطفي، يضبط `emo_alpha` مقدار تأثيره على المخرجات.
النطاق الصالح: `0.0 - 1.0`، الافتراضي: `1.0` (100%).

```python
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", lang="ZH", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)
```

#### 4. التحكم في المشاعر باستخدام متجه مشاعر

يمكنك حذف الصوت المرجعي العاطفي وتقديم بدلًا منه قائمة من 8 أرقام عشرية
تحدد شدة كل شعور، بالترتيب
`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.
استخدم `use_random` لإدخال العشوائية أثناء الاستدلال (الافتراضي: `False`).

> [!NOTE]
> تفعيل أخذ العينات العشوائي يقلل من دقة استنساخ الصوت.

```python
text = "对不起嘛！我的记性真的不太好，但是和你在一起的事情，我都会努力记住的~"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_09.wav', text=text, output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_09.wav', text=text, lang="ZH", output_path="gen.wav", emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], use_random=False, verbose=True)
```

#### 5. التحكم في المشاعر من النص نفسه (`use_emo_text`)

فعّل `use_emo_text` لتحويل نص `text` تلقائيًا إلى متجهات مشاعر. يُوصى
بقيمة `emo_alpha` حوالي 0.6 (أو أقل) للحصول على كلام أكثر طبيعية. يمكن
إدخال العشوائية باستخدام `use_random` (الافتراضي: `False`).

```python
text = "快躲起来！是他要来了！他要来抓我们了！"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```

#### 6. التحكم في المشاعر بوصف عاطفي صريح (`emo_text`)

قدّم وصفًا نصيًا محددًا للمشاعر عبر `emo_text`، والذي يتم تحويله إلى
متجهات مشاعر — مما يمنحك تحكمًا منفصلًا في النص ووصف المشاعر:

```python
text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"

# IndexTTS2
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)

# IndexTTS2.5
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, lang="ZH", output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
```

#### 7. التحكم في سرعة الكلام (`duration_factor`)

القيمة الأكبر من `1.0` تبطئ الكلام، والقيمة الأصغر من `1.0` تسرّعه.
الافتراضي: `1.0` (السرعة العادية). النطاق الصالح: `0.5 - 2.0`.

```python
text = "大家好，欢迎来到IndexTTS的语速控制演示。"

# IndexTTS2.5
# Slow down (1.2x duration)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="ZH", output_path="gen_slow.wav", duration_factor=1.2, verbose=True)

# Speed up (0.8x duration)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, lang="ZH", output_path="gen_fast.wav", duration_factor=0.8, verbose=True)
```

### 🗣️ التحكم في النطق

**IndexTTS2.5 — البينيين / أصوات CMU / الكانا اليابانية:**

يدعم IndexTTS2.5 هذه الاستبدالات النصية مع قدرة أفضل على اتباع التعليمات.
للقائمة الكاملة بالإدخالات الصالحة، راجع `checkpoints/pinyin.vocab` للبينيين
و[قاموس CMU](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b)
للأصوات الإنجليزية.

```
他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。

He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details of the contract.

彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
```

**IndexTTS2 — البينيين:**

يدعم IndexTTS2 النمذجة المختلطة للأحرف الصينية والبينيين. لتفعيل التحكم
في البينيين، قدّم نصًا مع تعليقات بينيين محددة. لاحظ أن التحكم في البينيين
لا يعمل مع كل تركيبة صامت–صائت ممكنة؛ فقط حالات البينيين الصيني الصالحة
مدعومة (راجع `checkpoints/pinyin.vocab`).

```
之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。
```

### 🕰️ IndexTTS-1.5 (الإصدار القديم)

يمكنك أيضًا استخدام نموذج IndexTTS1 السابق باستيراد وحدة مختلفة:

```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")
voice = "examples/voice_07.wav"
text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, 'gen.wav')
```

لمزيد من التفاصيل، راجع [README_INDEXTTS_1_5](../archive/README_INDEXTTS_1_5.md)،
أو زر مستودع IndexTTS1 على [index-tts:v1.5.0](https://github.com/index-tts/index-tts/tree/v1.5.0).

## 📊 التقييم

**الجدول 1: تحويل النص إلى كلام بتقنية الاستدلال الصفري على CV3-Eval** (تستخدم العربية مجموعة اختبار داخلية). †مقتبس من الورقة الأصلية.

<table>
<thead>
<tr>
<th rowspan="2">النموذج</th>
<th rowspan="2">المعاملات</th>
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

**الجدول 2: تحويل النص إلى كلام عبر اللغات على CV3-Eval** (مطالبة صينية ← اللغة المستهدفة، تستخدم العربية مجموعة اختبار داخلية).

<table>
<thead>
<tr>
<th rowspan="2">النموذج</th>
<th rowspan="2">المعاملات</th>
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

## 🤝 المجتمع والتواصل

- **QQ Groups:** 663272642 (No.4), 1013410623 (No.5)
- **Discord:** https://discord.gg/uT32E7KDmy
- **Email:** indexspeech@bilibili.com

نرحب بانضمامك إلى مجتمعنا! 🌏 欢迎大家来交流讨论！

> [!CAUTION]
> شكرًا لدعمكم مشروع bilibili IndexTTS!
> يرجى ملاحظة أن **القناة الرسمية الوحيدة** التي يديرها الفريق الأساسي هي: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts).
> ***أي مواقع أو خدمات أخرى ليست رسمية***، ولا يمكننا ضمان أمانها أو دقتها أو تحديثها في الوقت المناسب.
> للاطلاع على آخر التحديثات، يرجى دائمًا الرجوع إلى هذا المستودع الرسمي.

للاستخدام التجاري والتعاون، يرجى التواصل عبر <u>indexspeech@bilibili.com</u>.

## 📚 الاستشهاد

🌟 إذا وجدت عملنا مفيدًا، يرجى منحنا نجمة والاستشهاد بأوراقنا البحثية.

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

## 🙏 شكر وتقدير

1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)
6. [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
7. [seed-vc](https://github.com/Plachtaa/seed-vc)

## 📄 الترخيص

هذا المشروع صادر بموجب [اتفاقية ترخيص استخدام نماذج bilibili](../LICENSE).
يرجى أيضًا قراءة [إخلاء المسؤولية](../DISCLAIMER) قبل الاستخدام.
