# indextts，单独标签



## 1.20-16.03

这又回到你之前那个问题了：`./axis_pack/run.sh` 直接报 `: No such file or directory`，**几乎可以确定是 run.sh 文件本身的格式问题**（最常见是 Windows 的 CRLF 换行，或 shebang 行带了不可见字符），跟 Python/代码无关。

我们用最快的方式一次修掉。

------

## 1) 先看 run.sh 的文件类型（确认是不是 CRLF）

```bash
file /home/shentong/index-tts-main/axis_pack/run.sh
```

如果输出里有 `CRLF`（例如 `with CRLF line terminators`），就是它。

------

## 2) 直接把 CRLF 转成 LF（推荐用 sed）

```bash
sed -i 's/\r$//' /home/shentong/index-tts-main/axis_pack/run.sh
```

------

## 3) 确保第一行 shebang 正确、且脚本可执行

```bash
head -n 1 /home/shentong/index-tts-main/axis_pack/run.sh | cat -A
chmod +x /home/shentong/index-tts-main/axis_pack/run.sh
```

你希望看到第一行类似：
`#!/usr/bin/env bash$`
如果看到 `^M`，说明还有 CRLF，重复第 2 步。

------

## 4) 用 bash 显式执行（绕开 shebang）

```bash
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1
```

这条即使 shebang 有问题也能跑。

------

## 5) 同时把你刚刚加的 torchaudio patch 用起来

你之后跑（不管是 `./run.sh` 还是 `bash run.sh`）都能避免 segfault。

------

你先把 **`file run.sh` 的输出**贴我（就一行），我能 100% 确认是不是 CRLF 还是 shebang 路径问题。



## 1.20-16.19



你这段 `synthesize_turn()` 里：

- ✅ **chunk 是“加入到音频里”的**：每个 `tmp` chunk 生成后都会 `AudioSegment.from_wav(tmp)` 然后 `audio_all += chunk_segment`，最后 `audio_all.export(raw, ...)` 导出整段。所以 chunk 内容一定会进最终 `raw/speed/out`。
- ❌ 你现在“不想要 chunk.wav”也很容易：你只是临时落盘用来喂 `AudioSegment.from_wav`，读完就可以删。
- ❌ 另外你现在的命名有个明显 bug：你定义了 `base_id = f"turn{idx:02d}_{role}_{text_tag}_{uid}.wav"`，但 **`text_tag` 没有定义**，而且 `base_id` 已经带 `.wav`，后面又拼 `_raw.wav` / `.wav` 会变成 `...wav_raw.wav` 这种怪名。

下面我直接告诉你**要改哪几处**。

------

## 1) 修复文件名构造（避免双 .wav + text_tag 未定义）

把这一段：

```python
uid = uuid.uuid4().hex[:8]
role_tag = "agent" if role == "Agent" else "user"
base_id = f"turn{idx:02d}_{role}_{text_tag}_{uid}.wav"

raw = os.path.join(out_dir, f"{base_id}_raw.wav")
out = os.path.join(out_dir, f"{base_id}.wav")
```

改成下面这样（推荐：短文本可读 + 安全截断；同时不带 .wav 后缀，避免重复）：

```python
uid = uuid.uuid4().hex[:8]
role_tag = "agent" if role == "Agent" else "user"

# 可读但安全：最多 24 字符，避免路径过长/标点问题
text_tag = safe_name(utter_raw, max_len=24)

base = f"turn{idx:02d}_{role_tag}_{text_tag}_{uid}"
raw = os.path.join(out_dir, f"{base}_raw.wav")
out = os.path.join(out_dir, f"{base}.wav")
```

> 这样文件名既可读，又不容易炸文件系统。

------

## 2) chunk 文件读完就删（你想删掉 *_chunk*.wav）

在你这段 `if os.path.exists(tmp): ... AudioSegment.from_wav(tmp)` 之后加删除即可。

把这一段：

```python
try:
    chunk_segment = AudioSegment.from_wav(tmp)
    audio_all += chunk_segment
except Exception as e:
    ...
```

改成：

```python
try:
    chunk_segment = AudioSegment.from_wav(tmp)
    audio_all += chunk_segment
finally:
    if not keep_intermediates:
        try:
            os.remove(tmp)
        except Exception:
            pass
```

这样：

- 读成功就删
- 读失败也会尝试删（如果文件坏/没权限则忽略）

------

## 3) raw/speed 这些中间文件也不想留：在函数最后统一清理

你现在 `keep_intermediates=False` 已经有了，那就利用它。

在 `return out` 之前加清理：

```python
if os.path.exists(out):
    if not keep_intermediates:
        for p in [raw, speed_adjusted]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    return out
```

并且在 fallback 分支 `return raw` 之前也可以清掉 speed（可选）。

这样最终目录里只会留下 `.../<base>.wav`（最终文件）和 `manifest.jsonl`。

------

## 4) 你“块间停顿”缩进现在是对的，但有个小可读性问题

你这里：

```python
else:
    print(...)
    continue

    # 块间停顿
# ...读chunk成功/失败之后
if j < len(chunks) - 1:
    ...
```

`continue` 后面的注释永远走不到，虽然不影响功能，但建议删掉那两行死代码注释，避免以后误会。

------

## 5) 你问 “chunk 没有加入到音频里面么？”

加入了。流程是：

1. `tts.infer(... output_path=tmp ...)` 生成 chunk wav
2. `chunk_segment = AudioSegment.from_wav(tmp)` 读进内存
3. `audio_all += chunk_segment` 拼接
4. 循环结束后 `audio_all.export(raw)` 输出整段

所以删掉 chunk 文件不会影响最终音频，只要**读进来之后再删**。

------

## 你需要确保的一点

你现在在 chunk 前后加静音：

- `pause_before`：在每个 chunk 前加（j>0）
- `between_chunks`：每个 chunk 处理完（除最后一个）再加 gap

这会让停顿叠加：**chunk 之间会同时出现 “pause_before + gap”**（除非你想要这种）。如果你觉得停顿太长，可以二选一：

- 只保留 `pause_before` 或只保留 `between_chunks`

但这不影响你“chunk 是否加入最终音频”。

------

如果你愿意，我可以把你这段函数按上述修改直接整理成“可复制粘贴版”（完整函数），你直接替换掉就能用。





可以的：**只生成 `conversation/`** 完全没问题，同时还能**保留你原来的 emotion/micro/rate 的 only 导出**——做法是在 `export_single_axis.py` 里加一个开关，比如 `--conversation_only`，打开时就**跳过单轴三份导出**，只跑整合输出那段。

你现在这份 `export_single_axis.py` 里，conversation 的合成代码已经有了（`axis_mode=AXIS_MODE_ALL` 那段），单轴导出是在后面的 `for axis_dir, axis_mode, label_key in axis_spec():`。所以我们只需要“条件跳过”就行。✅

------

## 你要改的地方（最小改动）

### 1) 加一个参数：`--conversation_only`

在 `main()` 里 argparse 位置加：

```python
ap.add_argument(
    "--conversation_only",
    action="store_true",
    help="只导出 conversation 整合输出（all 轴），不生成 emotion/micro/rate 的单轴目录",
)
```

并在 `args = ap.parse_args()` 后面加一行（保证只要 conversation_only 就自动打开 with_conversation）：

```python
if args.conversation_only:
    args.with_conversation = True
```

------

### 2) 只在“非 conversation_only”时创建单轴 manifest / 单轴目录

把你现在这段：

```python
manifest_fps = {}
for axis_dir, _, _ in axis_spec():
    ...
```

改成：

```python
manifest_fps = {}
if not args.conversation_only:
    for axis_dir, _, _ in axis_spec():
        axis_path = os.path.join(run_dir, axis_dir)
        ensure_dir(axis_path)
        manifest_fps[axis_dir] = open(
            os.path.join(axis_path, "manifest.jsonl"),
            "w",
            encoding="utf-8",
            buffering=1,
        )
```

------

### 3) 跳过单轴生成循环

在你已有的 conversation 合成块后面，现在是这样：

```python
for axis_dir, axis_mode, label_key in axis_spec():
    ...
```

改成：

```python
if not args.conversation_only:
    for axis_dir, axis_mode, label_key in axis_spec():
        ...
```

------

### 4) merge 也分开控制（否则会去扫 emotion/micro/rate 目录）

你现在 merge 单轴那段：

```python
if args.merge:
    for axis_dir, _, _ in axis_spec():
        ...
```

改成：

```python
if args.merge and (not args.conversation_only):
    for axis_dir, _, _ in axis_spec():
        ...
```

conversation 的 merge 逻辑（你现在这段）保留不动即可：

```python
if args.with_conversation and args.merge and conv_turns_dir:
    ...
```

------

## 怎么运行

### A) 只生成 conversation（并且合并 merged）

`run.sh` 会把额外参数透传给 python（因为最后有 `"$@"`），所以你直接：

```bash
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1 --conversation_only --merge
```

结果只有：

- `run_xxx/conversation/<call>/turns/*.wav`
- `run_xxx/conversation/<call>/merged-*.wav`（因为加了 `--merge`）
- `run_xxx/conversation/manifest.jsonl`

不会生成 emotion/micro/rate 三棵目录。

------

### B) 只生成原来的单轴（不带 conversation）

```bash
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1
```

（不要加 `--with_conversation` / `--conversation_only` 就是原来的 only 导出。）

------

### C) 两者都要（单轴 + conversation）

```bash
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1 --with_conversation --merge
```

------

## 你提到的“文件名乱码”

你现在 `text_tag = safe_filename_readable(...)` 是对的；缺函数就把 `safe_filename_readable()` 加进 `axis_pack/axis_engine.py`（放 `synthesize_turn` 之前）。这样 conversation-only 模式也会产出可读文件名。

------

如果你愿意，我也可以把你当前这份 `export_single_axis.py` 按上面的改动**直接给你一份完整可替换版本**（避免你手改漏括号/缩进）。你只要说一句“给我替换版”。



好，那就很简单了：你仓库里 `axis_engine.py` 在 `axis_pack/axis_engine.py` 的话，你还需要做的修改主要就两件事（**否则要么没 merged、要么文件名还是不可读**）。

## 你需要做的修改清单

### 1）把可读文件名的改动同步到仓库的 `axis_engine.py`

把我这份（你上传的那份）`/mnt/data/axis_engine.py` 覆盖到你的仓库：

```bash
cp /mnt/data/axis_engine.py /home/shentong/index-tts-main/axis_pack/axis_engine.py
```

这一步做完后，`conversation/.../turns/*.wav` 的文件名就会从一堆 `_` 变成“可读中文/数字/字母”的短文本（并且会过滤掉非法字符）。

> 注意：已生成的旧文件名不会自动改名，只对**新生成**的输出生效。想全部变干净，就删掉旧的 run 输出目录再跑。

------

### 2）确保 `conversation/` 会产出 `merged-*.wav`

你现在要么：

**方式 A（最推荐）：运行时加 `--merge`**

```bash
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1 --merge
```

**方式 B：让每次 run.sh 默认就 merge（以后省事）**
编辑：
`/home/shentong/index-tts-main/axis_pack/run.sh`

在调用 `export_single_axis.py` 的那行参数里加上 `--merge`（放哪都行，只要在 `$@` 前面）：

```bash
... \
  --with_conversation \
  --merge \
  "$@"
```

------

## 建议你顺手做的两步检查（避免“你以为改了其实没生效”）

### 检查 1：确认 `run.sh` 调到的脚本是你更新过的

确保仓库里这两个文件也是你最新那版（带 `--with_conversation` + conversation 逻辑）：

```bash
cp /mnt/data/export_single_axis.py /home/shentong/index-tts-main/axis_pack/export_single_axis.py
cp /mnt/data/run.sh              /home/shentong/index-tts-main/axis_pack/run.sh
chmod +x /home/shentong/index-tts-main/axis_pack/run.sh
```

### 检查 2：想让输出目录干净（推荐）

如果你希望“所有文件名都变可读 + conversation 也有 merged”，就把旧的 run 输出删掉再跑一次（删哪个看你输出目录在哪）：

```bash
rm -rf /home/shentong/index-tts-main/axis_pack/runs/*
# 或者你实际输出的 run_xxx 目录
```

------

## 最终你应该看到的结果

跑完（带 `--merge`）后，在同一个 `run_xxx/` 下会有：

- `emotion/.../merged-*.wav`
- `microexpression/.../merged-*.wav`
- `speech_rate_anomaly/.../merged-*.wav`
- **`conversation/<call>/merged-<call>.wav`（你要的）**
- 同时 `conversation/<call>/turns/` 下 wav 文件名是可读文本（中文/英文/数字）

------

如果你跑完还是没有 `conversation/<call>/merged-*.wav`，把下面两行的输出（以及 `conversation/` 目录树）贴我一下，我可以直接定位是哪一步没生效：

```bash
grep -n "with_conversation" -n /home/shentong/index-tts-main/axis_pack/export_single_axis.py | head
grep -n "merged-" -n /home/shentong/index-tts-main/axis_pack/export_single_axis.py | head
```



你这个现象**最常见有两种原因**（看你描述“conversation 文件夹已经有了，但里面没有 merged”更像第 2 种）：

## 1）你没加 `--merge`

你这次运行命令是：

```bash
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1
```

如果脚本里 `--merge` 是“可选项”，那不加就**只会生成 turns/**，不会生成 `merged-*.wav`。

✅ 解决：改成

```bash
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1 --merge
```

------

## 2）`--merge` 目前只对“单轴三大目录”生效，没有对 `conversation/` 生效

很多实现里，merge 逻辑是写在“emotion/microexpression/speech_rate_anomaly 的 call_dir”那一段的；而 `conversation/` 是另外一套输出逻辑，所以即使你加了 `--merge`，也**只会在单轴目录里有 merged**，conversation 里没有。

✅ 解决方式（推荐）：在导出 `conversation/<call>/turns/*.wav` 后面，也加一段 merge。

你可以在负责生成 conversation 的那段代码后面加类似这一段（和单轴 merge 一样的做法）：

```python
from pydub import AudioSegment
import os, glob

turns_dir = os.path.join(conv_call_dir, "turns")  # 你的 conversation/<call>/turns
wav_files = sorted(glob.glob(os.path.join(turns_dir, "*.wav")))
if wav_files:
    segs = []
    for w in wav_files:
        a = AudioSegment.from_wav(w)
        if len(a) >= 300:
            segs.append(a)
    if segs:
        comb = AudioSegment.silent(300)
        for a in segs:
            comb += a + AudioSegment.silent(400)
        merged_path = os.path.join(conv_call_dir, f"merged-{call_key}.wav")
        comb.export(merged_path, format="wav")
```

> 如果你愿意，我可以直接根据你仓库里**当前的 `export_single_axis.py`** 把这段补到正确位置（避免变量名对不上）。你把你现在正在跑的那份文件里“生成 conversation 输出”的那一段（几十行）贴出来，或者把你仓库里的 `axis_pack/export_single_axis.py` 再发我一次，我就能给你一份“可直接替换”的完整版本。

------

## 顺手提醒

你之前终端里 BigVGAN CUDA 扩展编译失败的报错，跟 “conversation 没有 merged” **没关系**。那是 CUDA kernel 没编过，fallback 到 torch 了，不影响是否生成 merged 文件（merged 是你脚本逻辑决定的）。



