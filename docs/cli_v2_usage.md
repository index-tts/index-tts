# IndexTTS2 CLI v2 使用文档

本文档面向用户级 `indextts2` 命令行入口。CLI v2 是 IndexTTS2-first 的专用入口, 可安装到用户环境后从任意目录运行, 不依赖仓库根目录作为当前工作目录。它不会替换已有 `indextts` 命令。`indextts` 仍保留现有 IndexTTS1 行为。

## 功能范围

当前 CLI v2 提供以下子命令:

- `indextts2 init`: 创建持久化配置和默认模型资源目录, 不下载模型资源。
- `indextts2 config`: 查看或修改持久化配置。
- `indextts2 download`: 显式下载 IndexTTS2 模型资源。
- `indextts2 check`: 检查本地模型资源, Python 包和设备可用性。
- `indextts2 synth`: 使用 IndexTTS2 合成单条文本音频。
- `indextts2 batch`: 读取 JSON Lines 批量清单, 串行合成多条独立音频, 或合成后拼接为一个 WAV 文件。
- `indextts2 concat`: 读取 JSON Lines 拼接清单, 将已有 WAV 片段拼接为一个 WAV 文件。

以后可能会提供，但暂不包括:

- WebUI 子命令, WebUI 仍使用 `uv run webui.py`。
- `batch` 的 `--continue-on-error`。
- `batch` 的并发执行。
- GPT 采样细节参数, 例如 `top_p`, `top_k`, `temperature` 等。
- IndexTTS1/IndexTTS2 引擎选择器。

## 安装与入口

项目在 `pyproject.toml` 中注册了两个命令:

```toml
indextts = "indextts.cli:main"
indextts2 = "indextts.cli_v2:main"
```

普通用户推荐使用用户级安装方式注册 CLI。当前仓库包如果尚未发布到 PyPI 或默认包 registry, 需要在仓库根目录从本地源码安装:

```bash
uv tool install --python 3.10 .
```

也可以使用 `pipx` 从本地源码安装:

```bash
pipx install .
```

在仓库开发和调试时, 可使用 editable 安装方式注册 CLI:

```bash
uv tool install --python 3.10 -e .
```

安装后可查看帮助:

```bash
indextts2 --help
indextts2 init --help
indextts2 config --help
indextts2 download --help
indextts2 check --help
indextts2 synth --help
indextts2 batch --help
indextts2 concat --help
```

未安装入口时, 也可以在仓库根目录用 Python 模块方式调用。该入口保留给开发和调试使用, 不作为普通用户主路径:

```bash
python -m indextts.cli_v2 --help
```

## 首次初始化

`indextts2 init` 会创建平台标准配置目录, `config.toml` 和默认模型资源目录。它只准备本地状态, 不会自动联网下载模型资源。

```bash
indextts2 init
```

指定模型资源目录并写入持久化配置（请将 `D:/models/IndexTTS-2` 替换为你的目标模型目录）:

```bash
indextts2 init --model-dir D:/models/IndexTTS-2
```

`check`, `synth` 和 `batch` 在运行前也会执行首次初始化, 因此首次运行这些命令会创建配置文件和默认模型资源目录, 但仍不会自动下载模型资源。`concat` 不需要模型资源, 不执行模型资源检查。

## 模型资源目录

CLI v2 使用“模型资源目录”保存 IndexTTS2 的模型文件和配套资源。

平台默认模型资源目录:

| 平台 | 默认模型资源目录 |
| --- | --- |
| Windows | `%LOCALAPPDATA%\IndexTTS\models\IndexTTS-2` |
| macOS | `~/Library/Application Support/IndexTTS/models/IndexTTS-2` |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/indextts/models/IndexTTS-2` |

持久化配置位置:

| 平台 | 持久化配置 |
| --- | --- |
| Windows | `%APPDATA%\IndexTTS\config.toml` |
| macOS | `~/Library/Application Support/IndexTTS/config.toml` |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/indextts/config.toml` |

模型资源目录解析优先级:

1. 命令行参数 `--model-dir PATH`。
2. 环境变量 `INDEXTTS2_MODEL_DIR`。
3. 持久化配置 `model_dir`。
4. 平台默认模型资源目录。

CLI 内部固定使用:

```text
{model-resource-directory}/config.yaml
```

模型资源目录必须包含以下关键资源:

```text
config.yaml
bpe.model
gpt.pth
s2mel.pth
wav2vec2bert_stats.pt
feat1.pt
feat2.pt
qwen0.6bemo4-merge
hf_cache/w2v-bert-2.0
hf_cache/semantic_codec_model.safetensors
hf_cache/campplus_cn_common.bin
hf_cache/bigvgan/config.json
hf_cache/bigvgan/bigvgan_generator.pt
```

## 配置管理

查看持久化配置文件位置:

```bash
indextts2 config path
```

查看当前持久化配置:

```bash
indextts2 config get
```

写入模型资源目录:

```bash
indextts2 config set model_dir D:/models/IndexTTS-2
```

可配置键:

| 键 | 用途 |
| --- | --- |
| `model_dir` | 持久化模型资源目录。 |
| `default_device` | 持久化默认运行设备, 例如 `cpu`, `cuda`, `cuda:0`, `mps`, `xpu`。 |
| `use_fp16` | 持久化是否启用 FP16, 值为 `true` 或 `false`。 |
| `use_deepspeed` | 持久化是否启用 DeepSpeed, 值为 `true` 或 `false`。 |
| `use_cuda_kernel` | 持久化是否启用 CUDA kernel, 值为 `true` 或 `false`。 |

普通 `check`, `synth` 和 `batch` 参数只覆盖本次运行, 不会写回持久化配置。配置写入只通过 `indextts2 init --model-dir PATH`, `indextts2 config set KEY VALUE`, 以及成功的 `indextts2 download --model-dir PATH` 完成。`download --no-save` 不写回。

## 模型下载

推荐通过 CLI 显式下载模型资源:

```bash
indextts2 download
```

默认下载源是 HuggingFace。选择下载源:

```bash
indextts2 download --source huggingface
indextts2 download --source modelscope
```

指定下载目标目录:

```bash
indextts2 download --source huggingface --model-dir D:/models/IndexTTS-2
```

`download --model-dir PATH` 下载成功并通过资源检查后, 默认会把 `PATH` 写入持久化配置的 `model_dir`, 后续 `check`, `synth` 和 `batch` 会默认使用该目录。临时下载或预热其他目录时使用 `--no-save`:

```bash
indextts2 download --source modelscope --model-dir D:/tmp/IndexTTS-2 --no-save
```

下载命令通过 Python API 下载资源, 不依赖外部 `hf` 或 `modelscope` 可执行文件在 `PATH` 中。目标目录已有文件时, CLI 不会清空目录, 下载器可以增量补齐。下载完成后, CLI 会复用模型资源检查判断目录是否可用。

## 环境检查

运行:

```bash
indextts2 check
```

指定设备检查:

```bash
indextts2 check --device cuda:0
```

`check` 会检查:

- 解析后的模型资源目录是否存在。
- 必需模型资源是否存在。
- Python 包是否可导入: `torch`, `torchaudio`, `indextts`。
- 设备可用性: `cuda`, `xpu`, `mps`, `cpu`。
- 当传入 `--device` 时, 该设备是否可用。

示例成功输出:

```text
Checking model directory: <resolved-model-resource-directory>
OK: model directory <resolved-model-resource-directory>
OK: required model files
OK: python packages
cuda: available
xpu: unavailable
mps: unavailable
cpu: available
```

## 单条文本合成

最小命令:

```bash
indextts2 synth --text "你好, IndexTTS2。" --voice examples/voice_01.wav --output outputs/hello.wav
```

从 UTF-8 文本文件读取:

```bash
indextts2 synth --text-file input.txt --voice examples/voice_01.wav --output outputs/hello.wav
```

从标准输入读取:

```bash
echo "从标准输入读取文本。" | indextts2 synth --stdin --voice examples/voice_01.wav --output outputs/stdin.wav
```

文本来源必须三选一:

- `--text TEXT`
- `--text-file PATH`
- `--stdin`

CLI 会去除文本首尾空白, 去除后不能为空。`--text-file` 按 UTF-8 读取。

## 输出文件

`--output` 必填。默认不覆盖已有文件:

```bash
indextts2 synth --text "测试覆盖。" --voice examples/voice_01.wav --output outputs/hello.wav
```

如需覆盖已有输出, 使用 `--force`:

```bash
indextts2 synth --text "测试覆盖。" --voice examples/voice_01.wav --output outputs/hello.wav --force
```

输出路径的父目录会自动创建。合成成功时, 标准输出会打印:

```text
Generated: outputs/hello.wav
```

## 音色与情感控制

`--voice` 必填, 指向音色参考音频:

```bash
indextts2 synth --text "保持音色。" --voice examples/voice_01.wav --output outputs/voice.wav
```

使用情感参考音频:

```bash
indextts2 synth --text "使用情感参考音频。" --voice examples/voice_01.wav --emotion-audio examples/emo_sad.wav --emotion-weight 0.75 --output outputs/emotion_audio.wav
```

使用情感描述文本:

```bash
indextts2 synth --text "使用情感描述文本。" --voice examples/voice_01.wav --emotion-text "warm and calm" --emotion-weight 0.6 --output outputs/emotion_text.wav
```

使用 8 维情感向量:

```bash
indextts2 synth --text "使用情感向量。" --voice examples/voice_01.wav --emotion-vector 0,0,0.8,0,0,0,0,0 --emotion-weight 1.0 --output outputs/emotion_vector.wav
```

8 维情感向量的顺序固定为: 高兴, 愤怒, 悲伤, 害怕, 厌恶, 忧郁, 惊讶, 平静。

`--emotion-vector` 接受两种格式:

- 逗号分隔格式: `0,0,0.8,0,0,0,0,0`
- 带方括号的列表格式: `[0, 0, 0.8, 0, 0, 0, 0, 0]`

规则:

- `--emotion-vector`, `--emotion-audio` 和 `--emotion-text` 三者互斥, 每次只能选择一种情感来源。
- `--emotion-text` 不能为空字符串。
- `--emotion-vector` 必须正好包含 8 个数字。
- `--emotion-vector` 每个值必须在 `0.0..1.0` 范围内。
- `--emotion-vector` 的总和必须小于等于 `0.8`。
- CLI 不会自动归一化, 裁剪, 重平衡或改写情感向量。
- `--emotion-weight` 必须是浮点数, 默认值为 `1.0`。
- `--emotion-weight` 会传给 IndexTTS2 的 `emo_alpha`。用于 `--emotion-vector` 时, 它会整体缩放 8 维情感向量的强度。

## 批量合成

`indextts2 batch` 是当前 CLI v2 的批量合成子命令。它读取一份 JSON Lines 批量清单, 先预检整份清单, 再只初始化一次模型。默认模式按清单顺序串行生成多份独立音频文件。传入 `--concat` 时, CLI 会先为每条任务生成临时 WAV 片段, 再按清单顺序拼接成一个最终 WAV 文件。

`batch` 是新增子命令, 不会改变 `indextts2 synth` 的单条合成契约。

### 基本命令

只校验批量清单和模型资源, 不加载模型, 不生成音频:

```bash
indextts2 batch --batch-file examples/batch/demo.jsonl --voice examples/voice_01.wav --dry-run
```

`--dry-run` 成功时输出:

```text
Batch file OK: 4 tasks
```

真实批量合成:

```bash
indextts2 batch --batch-file examples/batch/demo.jsonl --voice examples/voice_01.wav --device cuda:0 --fp16
```

合成成功时会逐条输出生成路径, 最后输出汇总:

```text
Generated: examples/batch/out/001.wav
Generated: examples/batch/out/002.wav
Generated: examples/batch/out/003.wav
Generated: examples/batch/out/004.wav
Batch complete: 4 tasks generated
```

### 输出模式

`batch` 支持三种输出模式。

逐行输出是默认模式。清单中每行必须提供 `output`:

```bash
indextts2 batch --batch-file examples/batch/demo.jsonl --voice examples/voice_01.wav
```

自动命名输出使用 `--output-dir`, 清单中不能提供 `output`。输出文件按任务顺序命名为 `0001.wav`, `0002.wav`:

```bash
indextts2 batch --batch-file examples/batch/auto-output.jsonl --voice examples/voice_01.wav --output-dir examples/batch/out/auto --dry-run
```

如需文件名前缀, 加 `--output-prefix`。前缀不能包含路径分隔符或文件扩展名:

```bash
indextts2 batch --batch-file examples/batch/auto-output.jsonl --voice examples/voice_01.wav --output-dir examples/batch/out/auto --output-prefix chapter --dry-run
```

上例真实执行时会生成:

```text
examples/batch/out/auto/chapter-0001.wav
examples/batch/out/auto/chapter-0002.wav
examples/batch/out/auto/chapter-0003.wav
```

拼接输出使用 `--concat --output <final.wav>`, 清单中不能提供逐行 `output`:

```bash
indextts2 batch --batch-file examples/batch/batch-concat.jsonl --voice examples/voice_01.wav --concat --output examples/batch/out/story.wav --dry-run
```

`--dry-run` 成功时输出:

```text
Batch concat OK: 3 tasks
```

真实执行时, CLI 会创建内部临时目录, 逐条合成 WAV 片段, 拼接为 `--output` 指定的最终 WAV, 然后默认删除临时目录:

```bash
indextts2 batch --batch-file examples/batch/batch-concat.jsonl --voice examples/voice_01.wav --concat --output examples/batch/out/story.wav
```

成功时输出:

```text
Generated: <resolved-output.wav>
```

如需保留内部临时 WAV 片段用于排查问题, 加 `--keep-temp`:

```bash
indextts2 batch --batch-file examples/batch/batch-concat.jsonl --voice examples/voice_01.wav --concat --output examples/batch/out/story.wav --keep-temp
```

成功时会额外输出临时目录:

```text
Generated: <resolved-output.wav>
Temp dir: <resolved-temp-dir>
```

### JSON Lines 批量清单

批量清单每个非空行是一条 JSON object。空行会被忽略, 不计入任务数。当前不支持注释行或行内注释。

示例 `examples/batch/demo.jsonl`:

```jsonl
{"text": "第一句直接写在批量清单里。", "output": "out/001.wav"}
{"text_file": "texts/second.txt", "emotion_audio": "../emo_sad.wav", "emotion_weight": 0.75, "output": "out/002.wav"}
{"text": "第三句使用 JSON array 格式的情感向量。", "emotion_vector": [0, 0, 0.6, 0, 0, 0, 0, 0], "output": "out/003.wav"}
{"text": "第四句使用 CLI 字符串格式的情感向量。", "emotion_vector": "0,0,0,0,0,0,0,0.5", "emotion_weight": "0.4", "output": "out/004.wav"}
```

每行支持的字段:

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `text` | 与 `text_file` 二选一 | 直接传入待合成文本, 去除首尾空白后不能为空。 |
| `text_file` | 与 `text` 二选一 | 从 UTF-8 文本文件读取待合成文本, 整个文件作为一条任务文本。 |
| `voice` | 有命令级 `--voice` 时可省略 | 音色参考音频路径。 |
| `output` | 默认逐行输出模式必填 | 输出音频路径。使用 `--output-dir` 或 `--concat` 时不能提供。 |
| `emotion_audio` | 否 | 情感参考音频路径。 |
| `emotion_text` | 否 | 情感描述文本。 |
| `emotion_vector` | 否 | 8 维情感向量, 支持 JSON array 或 CLI 风格字符串。 |
| `emotion_weight` | 否 | 情感权重, 必须可解析为浮点数。 |
| `silence_after_ms` | 否 | 拼接模式下, 在当前片段后追加的静音毫秒数。只能与 `--concat` 一起使用。 |

规则:

- 未知字段会报错, 避免字段拼写错误被静默忽略。
- `text` 和 `text_file` 必须二选一, 不能同时提供, 也不能都不提供。
- 批量清单不支持逐行 `stdin`。
- `voice` 可逐行提供, 也可用命令级 `--voice` 作为默认值。
- 默认逐行输出模式下, `output` 必须解析为每条任务的唯一输出路径。
- 使用 `--output-dir` 时, `output` 由 CLI 自动生成, 清单中不能出现 `output`。
- 使用 `--concat` 时, 最终输出来自命令级 `--output`, 清单中不能出现逐行 `output`。
- `silence_after_ms` 必须是非负整数, 默认值为 `0`。

### 路径解析

批量清单中的相对路径按批量清单文件所在目录解析, 包括 `text_file`, `voice`, 默认逐行输出模式下的 `output` 和 `emotion_audio`。

命令行参数路径仍按当前工作目录解析, 包括 `--batch-file`, 显式传入的 `--model-dir`, 命令级 `--voice`, 命令级 `--emotion-audio`, `--output-dir` 和拼接模式的 `--output`。

例如从仓库根目录运行:

```bash
indextts2 batch --batch-file examples/batch/demo.jsonl --voice examples/voice_01.wav --dry-run
```

如果 `examples/batch/demo.jsonl` 某行写入 `"output": "out/001.wav"`, 实际输出路径会解析为 `examples/batch/out/001.wav`。

如果使用:

```bash
indextts2 batch --batch-file examples/batch/auto-output.jsonl --voice examples/voice_01.wav --output-dir examples/batch/out/auto --dry-run
```

自动输出目录 `examples/batch/out/auto` 会按当前工作目录解析, 不会按 `auto-output.jsonl` 所在目录再次拼接。

### 覆盖, 失败和输出

默认不覆盖已有输出文件。如果任一任务的输出文件已存在, 且没有传 `--force`, 命令会在模型初始化前失败:

```bash
indextts2 batch --batch-file examples/batch/demo.jsonl --voice examples/voice_01.wav --force
```

覆盖规则:

- `--force` 只允许覆盖批量运行开始前已经存在的外部输出文件。
- 同一批量清单内部的重复 `output` 始终是输入错误, 即使传入 `--force` 也会失败。
- 使用 `--output-dir` 时, 自动生成的输出文件如果已存在, 仍需传入 `--force` 才能覆盖。
- 使用 `--concat` 时, 最终 `--output` 如果已存在, 仍需传入 `--force` 才能覆盖。
- 使用 `--output-dir` 或 `--concat` 时, CLI 会拒绝把生成输出写到批量清单, `text_file`, `voice` 或 `emotion_audio` 等受保护输入路径上。
- 批量合成当前采用失败即停止。没有 `--continue-on-error`。
- 如果某条任务在推理阶段失败, 已经生成的前序输出文件不会回滚删除。
- 批量拼接模式下, 推理失败或拼接失败会默认清理内部临时目录。传入 `--keep-temp` 时会保留临时目录并把路径写入输出。
- 成功任务会逐条打印 `Generated: <path>`。
- 全部任务成功后打印 `Batch complete: <n> tasks generated`。
- 批量拼接成功后只打印最终文件 `Generated: <path>`, 不逐条打印临时片段。
- 与具体行相关的错误会包含批量清单中的 1-based 行号。

### 批量情感字段

命令行可以提供公共默认值:

```bash
indextts2 batch --batch-file examples/batch/demo.jsonl --voice examples/voice_01.wav --emotion-text "warm and calm" --emotion-weight 0.6
```

批量清单每行可以用同名 snake_case 字段覆盖命令级默认值:

- 命令级 `--voice` 对应逐行 `voice`。
- 命令级 `--emotion-audio` 对应逐行 `emotion_audio`。
- 命令级 `--emotion-text` 对应逐行 `emotion_text`。
- 命令级 `--emotion-vector` 对应逐行 `emotion_vector`。
- 命令级 `--emotion-weight` 对应逐行 `emotion_weight`。

情感规则:

- `emotion_audio`, `emotion_text` 和 `emotion_vector` 是互斥的情感来源, 每行最多提供一种。
- 如果一行没有情感来源字段, 会继承命令级情感来源。
- 如果一行提供了情感来源字段, 该行会覆盖命令级情感来源。
- 如果一行没有 `emotion_weight`, 会继承命令级 `--emotion-weight`, 默认是 `1.0`。
- 如果一行只提供 `emotion_weight`, 它会继承命令级情感来源并只覆盖强度。
- 如果没有命令级情感来源, 单独提供逐行 `emotion_weight` 是输入错误。
- 如果没有任何情感来源, 当前行为与 `synth` 一致, 情感跟随 `voice` 参考音频。

批量清单中的 `emotion_vector` 接受两种格式:

```jsonl
{"text": "JSON array 格式。", "voice": "../voice_01.wav", "emotion_vector": [0, 0, 0.8, 0, 0, 0, 0, 0], "output": "out/vector-array.wav"}
{"text": "CLI 字符串格式。", "voice": "../voice_01.wav", "emotion_vector": "0,0,0.8,0,0,0,0,0", "output": "out/vector-string.wav"}
```

`emotion_vector` 的顺序和校验规则与 `synth --emotion-vector` 相同: 必须正好 8 个数字, 每个值在 `0.0..1.0` 范围内, 总和必须小于等于 `0.8`。

## 已有 WAV 拼接

`indextts2 concat` 用于拼接已经存在的 WAV 文件。它不会加载模型, 也不会执行 TTS 推理。它只读取拼接清单, 校验每个 WAV 片段, 再按清单顺序写出一个新的 WAV 文件。

### 基本命令

只校验拼接清单, 不创建输出文件:

```bash
indextts2 concat --concat-file examples/batch/concat-audio.jsonl --output examples/batch/out/emotion_joined.wav --dry-run
```

`--dry-run` 成功时输出:

```text
Concat file OK: 2 segments
```

真实拼接:

```bash
indextts2 concat --concat-file examples/batch/concat-audio.jsonl --output examples/batch/out/emotion_joined.wav
```

成功时输出:

```text
Generated: <resolved-output.wav>
```

默认不覆盖已有输出。如需覆盖, 加 `--force`:

```bash
indextts2 concat --concat-file examples/batch/concat-audio.jsonl --output examples/batch/out/emotion_joined.wav --force
```

### JSON Lines 拼接清单

拼接清单每个非空行是一条 JSON object。空行会被忽略, 不计入片段数。当前不支持注释行或行内注释。

示例 `examples/batch/concat-audio.jsonl`:

```jsonl
{"audio": "../emo_hate.wav", "silence_after_ms": 300}
{"audio": "../emo_sad.wav", "silence_after_ms": 500}
```

每行支持的字段:

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `audio` | 是 | 待拼接 WAV 文件路径。 |
| `silence_after_ms` | 否 | 当前片段后追加的静音毫秒数, 必须是非负整数, 默认 `0`。 |

规则:

- 未知字段会报错。
- `audio` 必须是非空字符串, 并且扩展名必须是 `.wav`, 大小写不敏感。
- `audio` 相对路径按拼接清单文件所在目录解析。
- `--concat-file` 和 `--output` 相对路径按当前工作目录解析。
- `--output` 必须是 `.wav` 文件, 不能与 `--concat-file` 相同, 也不能与任何 `audio` 输入相同。
- 清单必须至少包含一个片段。
- 每个输入 WAV 必须存在, 可读取, 且至少包含一个音频帧。
- 所有输入 WAV 的采样率, 声道数和采样宽度必须一致。
- 拼接时会保留清单顺序, 并在每个片段后追加 `silence_after_ms` 指定的静音。

## 运行参数

单条合成示例:

```bash
indextts2 synth --text "运行参数示例。" --voice examples/voice_01.wav --output outputs/runtime.wav --device cuda:0 --fp16 --deepspeed --cuda-kernel --verbose
```

批量合成示例:

```bash
indextts2 batch --batch-file examples/batch/demo.jsonl --voice examples/voice_01.wav --device cuda:0 --fp16 --deepspeed --cuda-kernel --verbose
```

批量合成后拼接示例:

```bash
indextts2 batch --batch-file examples/batch/batch-concat.jsonl --voice examples/voice_01.wav --concat --output examples/batch/out/story.wav --device cuda:0 --fp16
```

参数说明:

| 参数 | 说明 |
| --- | --- |
| `--model-dir PATH` | 本次运行使用的模型资源目录, 覆盖 `INDEXTTS2_MODEL_DIR`, 持久化配置和平台默认值。 |
| `--device DEVICE` | 本次运行设备, 例如 `cpu`, `cuda`, `cuda:0`, `mps`, `xpu`; 未传时可使用持久化配置 `default_device`。 |
| `--fp16` / `--no-fp16` | 本次运行是否启用 FP16 半精度推理; 未传时可使用持久化配置 `use_fp16`。 |
| `--deepspeed` / `--no-deepspeed` | 本次运行是否启用 DeepSpeed; 未传时可使用持久化配置 `use_deepspeed`。 |
| `--cuda-kernel` / `--no-cuda-kernel` | 本次运行是否启用 CUDA kernel 路径; 未传时可使用持久化配置 `use_cuda_kernel`。 |
| `--verbose` | 显示模型运行输出, 并向 `IndexTTS2.infer` 传入 `verbose=True`。 |

默认情况下, CLI 会隐藏模型初始化和推理过程中的普通标准输出。`synth` 成功后打印 `Generated: <path>`; `batch` 每条成功任务打印 `Generated: <path>`, 全部成功后再打印 `Batch complete: <n> tasks generated`。需要调试时使用 `--verbose`。

`concat` 不加载模型, 因此不支持 `--model-dir`, `--device`, `--fp16`, `--deepspeed`, `--cuda-kernel` 或 `--verbose`。

## 参数速查

### `indextts2 init`

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--model-dir PATH` | 否 | 平台默认模型资源目录 | 创建本地状态并把模型资源目录写入持久化配置。 |

### `indextts2 config`

| 命令 | 说明 |
| --- | --- |
| `indextts2 config path` | 输出持久化配置文件位置。 |
| `indextts2 config get` | 输出当前持久化配置。 |
| `indextts2 config set KEY VALUE` | 写入一个配置值。`KEY` 可为 `model_dir`, `default_device`, `use_fp16`, `use_deepspeed`, `use_cuda_kernel`。 |

### `indextts2 download`

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--source huggingface|modelscope` | 否 | `huggingface` | 模型下载源。 |
| `--model-dir PATH` | 否 | 解析后的模型资源目录 | 下载目标目录。成功后默认写入持久化配置。 |
| `--no-save` | 否 | `False` | 与 `--model-dir` 一起使用时, 禁止下载成功后写回 `model_dir`。 |

### `indextts2 check`

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--model-dir PATH` | 否 | 解析后的模型资源目录 | 本次检查使用的 IndexTTS2 模型资源目录。 |
| `--device DEVICE` | 否 | 无 | 要求可用的运行设备, 例如 `cuda:0`。 |

### `indextts2 synth`

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--text TEXT` | 三选一 | 无 | 直接传入待合成文本。 |
| `--text-file PATH` | 三选一 | 无 | 从 UTF-8 文本文件读取待合成文本。 |
| `--stdin` | 三选一 | `False` | 从标准输入读取待合成文本。 |
| `--voice PATH` | 是 | 无 | 音色参考音频路径。 |
| `--emotion-audio PATH` | 否 | 无 | 情感参考音频路径, 与 `--emotion-vector` 和 `--emotion-text` 互斥。 |
| `--emotion-text TEXT` | 否 | 无 | 情感描述文本, 与 `--emotion-vector` 和 `--emotion-audio` 互斥。 |
| `--emotion-vector VECTOR` | 否 | 无 | 8 维情感向量, 支持逗号分隔或带方括号的列表格式, 与其他情感来源互斥。 |
| `--emotion-weight FLOAT` | 否 | `1.0` | 情感权重, 映射到 `emo_alpha`, 可缩放情感向量强度。 |
| `--output PATH` | 是 | 无 | 输出音频路径。 |
| `--force` | 否 | `False` | 允许覆盖已有输出文件。 |
| `--model-dir PATH` | 否 | 解析后的模型资源目录 | 本次合成使用的 IndexTTS2 模型资源目录。 |
| `--device DEVICE` | 否 | 持久化配置或无 | 运行设备。 |
| `--fp16` / `--no-fp16` | 否 | 持久化配置或 `False` | 启用或禁用 FP16 推理。 |
| `--deepspeed` / `--no-deepspeed` | 否 | 持久化配置或 `False` | 启用或禁用 DeepSpeed。 |
| `--cuda-kernel` / `--no-cuda-kernel` | 否 | 持久化配置或 `False` | 启用或禁用 CUDA kernel。 |
| `--verbose` | 否 | `False` | 显示详细运行输出。 |

### `indextts2 batch`

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--batch-file PATH` | 是 | 无 | JSON Lines 批量清单路径。 |
| `--dry-run` | 否 | `False` | 只校验批量清单和模型资源, 不加载模型, 不生成音频。 |
| `--output-dir PATH` | 否 | 无 | 自动命名输出目录。传入后清单中不能提供逐行 `output`。 |
| `--output-prefix TEXT` | 否 | 无 | 自动命名输出文件前缀, 需与 `--output-dir` 一起使用。 |
| `--concat` | 否 | `False` | 合成后拼接为一个最终 WAV 文件。 |
| `--output PATH` | `--concat` 模式必填 | 无 | 拼接模式的最终 WAV 输出路径。只可与 `--concat` 一起使用。 |
| `--keep-temp` | 否 | `False` | 拼接模式下保留内部临时 WAV 片段目录。 |
| `--voice PATH` | 否 | 无 | 每条任务的默认音色参考音频, 可被逐行 `voice` 覆盖。 |
| `--emotion-audio PATH` | 否 | 无 | 默认情感参考音频, 与 `--emotion-vector` 和 `--emotion-text` 互斥。 |
| `--emotion-text TEXT` | 否 | 无 | 默认情感描述文本, 与 `--emotion-vector` 和 `--emotion-audio` 互斥。 |
| `--emotion-vector VECTOR` | 否 | 无 | 默认 8 维情感向量, 支持逗号分隔或带方括号的列表格式, 与其他情感来源互斥。 |
| `--emotion-weight FLOAT` | 否 | `1.0` | 默认情感权重, 可被逐行 `emotion_weight` 覆盖。 |
| `--force` | 否 | `False` | 允许覆盖批量运行开始前已经存在的输出文件。 |
| `--model-dir PATH` | 否 | 解析后的模型资源目录 | 本次批量合成使用的 IndexTTS2 模型资源目录。 |
| `--device DEVICE` | 否 | 持久化配置或无 | 运行设备。 |
| `--fp16` / `--no-fp16` | 否 | 持久化配置或 `False` | 启用或禁用 FP16 推理。 |
| `--deepspeed` / `--no-deepspeed` | 否 | 持久化配置或 `False` | 启用或禁用 DeepSpeed。 |
| `--cuda-kernel` / `--no-cuda-kernel` | 否 | 持久化配置或 `False` | 启用或禁用 CUDA kernel。 |
| `--verbose` | 否 | `False` | 显示详细运行输出。 |

### `indextts2 concat`

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--concat-file PATH` | 是 | 无 | JSON Lines 拼接清单路径。 |
| `--output PATH` | 是 | 无 | 拼接后的 WAV 输出路径。 |
| `--force` | 否 | `False` | 允许覆盖已有输出文件。 |
| `--dry-run` | 否 | `False` | 只校验拼接清单和输入 WAV, 不生成输出音频。 |

## 返回码

| 返回码 | 含义 | 常见场景 |
| --- | --- | --- |
| `0` | 成功 | 初始化完成, 配置更新, 下载检查通过, 环境检查通过或合成成功。 |
| `1` | 输入错误 | 文本来源不唯一, 文本为空, 缺少 `--output`, 输出已存在但未传 `--force`, 情感参数冲突, 批量清单字段错误, 拼接清单字段错误, WAV 格式不匹配。 |
| `2` | 本地资源缺失 | 模型资源目录不存在, 必需模型资源缺失, 文本文件不存在, 音色参考音频不存在, 情感参考音频不存在, 批量清单不存在, 拼接清单不存在, 拼接音频不存在。 |
| `3` | 运行环境不可用 | 必需 Python 包缺失, 下载源 Python 包缺失, 指定设备不可用, IndexTTS2 runtime 导入失败。 |
| `4` | 推理或拼接失败 | 模型初始化, `infer` 执行或 WAV 写入替换过程中抛出异常。批量合成会停在第一条失败任务。 |

错误信息写入 `stderr`, 成功信息写入 `stdout`。

## 常见错误

文本来源冲突:

```text
ERROR: provide exactly one text source: --text, --text-file or --stdin
```

处理方式: 只保留 `--text`, `--text-file`, `--stdin` 中的一个。

模型资源目录不存在:

```text
ERROR: model directory does not exist: <resolved-model-resource-directory>
Model directory: <resolved-model-resource-directory>
Missing resources: model directory does not exist
Download with HuggingFace:
  huggingface-cli download IndexTeam/IndexTTS-2 --local-dir "<resolved-model-resource-directory>"
Download with ModelScope:
  modelscope download --model IndexTeam/IndexTTS-2 --local_dir "<resolved-model-resource-directory>"
Persist a different model resource directory:
  indextts2 config set model_dir <resolved-model-resource-directory>
Hint: rerun indextts2 download or choose a different model resource directory.
```

处理方式: 按输出中的目标目录下载模型资源。推荐先使用 CLI 下载:

```bash
indextts2 download --model-dir <resolved-model-resource-directory>
```

如果模型资源已经在其他目录, 写入持久化配置:

```bash
indextts2 config set model_dir <existing-model-resource-directory>
```

模型资源缺失:

```text
ERROR: missing required model files: bpe.model, gpt.pth
Model directory: <resolved-model-resource-directory>
Missing resources: bpe.model, gpt.pth
Download with HuggingFace:
  huggingface-cli download IndexTeam/IndexTTS-2 --local-dir "<resolved-model-resource-directory>"
Download with ModelScope:
  modelscope download --model IndexTeam/IndexTTS-2 --local_dir "<resolved-model-resource-directory>"
Persist a different model resource directory:
  indextts2 config set model_dir <resolved-model-resource-directory>
Hint: rerun indextts2 download or choose a different model resource directory.
```

处理方式: 重新运行 `indextts2 download --model-dir <resolved-model-resource-directory>` 补齐资源, 或使用 `indextts2 config set model_dir <existing-model-resource-directory>` 指向已有完整模型资源目录, 然后重新运行 `indextts2 check`。

输出文件已存在:

```text
ERROR: output file already exists: outputs/hello.wav
```

处理方式: 更换 `--output` 路径, 或确认可覆盖后加 `--force`。

情感向量数量错误:

```text
ERROR: --emotion-vector must contain exactly 8 values; got 7
```

处理方式: 按高兴, 愤怒, 悲伤, 害怕, 厌恶, 忧郁, 惊讶, 平静的顺序传入 8 个数字。

批量清单字段错误:

```text
ERROR: batch file line 2 has unknown fields: prompt_audio
```

处理方式: 使用 `voice`, `emotion_audio`, `emotion_text`, `emotion_vector` 等 CLI v2 字段名, 不要使用底层 API 或旧样例字段名。

批量输出重复:

```text
ERROR: batch file line 3 has duplicate output path: examples/batch/out/001.wav
```

处理方式: 为同一批量清单中的每条任务设置不同的 `output`。`--force` 不允许批次内部互相覆盖。

批量拼接清单包含逐行输出:

```text
ERROR: batch file line 1 field 'output' is not allowed with --concat
```

处理方式: 使用 `--concat --output final.wav` 指定最终输出, 并从批量清单中删除逐行 `output`。

未开启批量拼接却提供静音字段:

```text
ERROR: batch file line 1 field 'silence_after_ms' is only valid with --concat
```

处理方式: 删除 `silence_after_ms`, 或在命令中加入 `--concat --output final.wav`。

拼接清单中的 WAV 格式不一致:

```text
ERROR: concat file line 2 WAV format does not match baseline line 1
```

处理方式: 确认所有输入 WAV 的采样率, 声道数和采样宽度一致后再拼接。

拼接输出已存在:

```text
ERROR: output file already exists: examples/batch/out/emotion_joined.wav
```

处理方式: 更换 `--output` 路径, 或确认可覆盖后加 `--force`。

设备不可用:

```text
ERROR: requested device is not available: cuda:0
```

处理方式: 使用 `indextts2 check` 查看当前设备可用性, 或改用 `--device cpu`。如果希望长期使用某个设备, 可写入持久化配置:

```bash
indextts2 config set default_device cuda:0
```

下载源 Python 包缺失:

```text
ERROR: runtime unavailable for huggingface download source: No module named 'huggingface_hub'
Install download support with: pip install huggingface_hub
```

处理方式: 为当前环境安装提示中的 Python 包, 或改用另一个下载源:

```bash
indextts2 download --source modelscope
```

## 推荐工作流

普通用户首次运行:

```bash
indextts2 init
indextts2 download
indextts2 check
indextts2 synth --text "你好, IndexTTS2。" --voice examples/voice_01.wav --output outputs/hello.wav
```

使用已有模型资源目录:

```bash
indextts2 config set model_dir D:/models/IndexTTS-2
indextts2 check
```

GPU 环境:

```bash
indextts2 config set default_device cuda:0
indextts2 config set use_fp16 true
indextts2 check
indextts2 synth --text "GPU 推理测试。" --voice examples/voice_01.wav --output outputs/gpu.wav
```

一次性覆盖模型资源目录或运行参数:

```bash
indextts2 check --model-dir D:/models/IndexTTS-2 --device cuda:0
indextts2 synth --text "GPU 推理测试。" --voice examples/voice_01.wav --output outputs/gpu.wav --device cuda:0 --fp16
```

调试模型输出:

```bash
indextts2 synth --text "调试输出。" --voice examples/voice_01.wav --output outputs/debug.wav --verbose
```

批量合成:

```bash
indextts2 check
indextts2 batch --batch-file examples/batch/demo.jsonl --voice examples/voice_01.wav --dry-run
indextts2 batch --batch-file examples/batch/demo.jsonl --voice examples/voice_01.wav
```

自动命名批量输出:

```bash
indextts2 batch --batch-file examples/batch/auto-output.jsonl --voice examples/voice_01.wav --output-dir examples/batch/out/auto --dry-run
indextts2 batch --batch-file examples/batch/auto-output.jsonl --voice examples/voice_01.wav --output-dir examples/batch/out/auto --output-prefix chapter
```

批量合成后拼接:

```bash
indextts2 batch --batch-file examples/batch/batch-concat.jsonl --voice examples/voice_01.wav --concat --output examples/batch/out/story.wav --dry-run
indextts2 batch --batch-file examples/batch/batch-concat.jsonl --voice examples/voice_01.wav --concat --output examples/batch/out/story.wav
```

拼接已有 WAV:

```bash
indextts2 concat --concat-file examples/batch/concat-audio.jsonl --output examples/batch/out/emotion_joined.wav --dry-run
indextts2 concat --concat-file examples/batch/concat-audio.jsonl --output examples/batch/out/emotion_joined.wav
```

开发和调试 CLI v2:

```bash
python -m indextts.cli_v2 --help
python -m indextts.cli_v2 check
```
