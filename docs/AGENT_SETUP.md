# Agent Setup Guide

Instructions for a coding agent (Claude Code, Cursor, Codex, …) asked to get
IndexTTS running on a user's machine. A user may hand you only the URL of this
file; everything you need is here.

Your goal: a working inference run. Your constraint: **do not redo work the
machine has already done.** A full setup downloads ~10 GB of wheels and ~5 GB of
weights. Users upgrading from IndexTTS-2 usually have most of it already.

Never guess state — run the probe commands and read the output.

## 0. Get the code, then probe

The repo comes first: a clone is ~57 MB and takes seconds, against ~18 GB for the
install it informs. Nothing here is worth working around it for.

**Check for an existing checkout first** — cloning over one, or beside one, is the
single most destructive thing available here. Someone upgrading from IndexTTS-2 may
have local edits, a warm `.venv`, and gigabytes of weights already in place.

```bash
git rev-parse --show-toplevel 2>/dev/null && echo "checkout already here — do NOT clone"
```

If that prints a path, you are already inside the repo: skip the clone entirely and
let step 2 update it. Note the directory may be named anything, so a blind
`cd index-tts` afterwards would land you somewhere else or fail.

Only if there is no checkout:

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
```

Then probe:

```bash
# repo
git rev-parse --abbrev-ref HEAD; git log --oneline -1
git status --short

# toolchain — uv often lives where a non-login shell cannot see it
uv --version || ls -l ~/.local/bin/uv ~/.cargo/bin/uv 2>/dev/null
python3 -VV

# accelerator: try every vendor, not just NVIDIA
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
rocm-smi --showproductname 2>/dev/null                       # AMD
xpu-smi discovery 2>/dev/null                                # Intel
[ "$(uname -sm)" = "Darwin arm64" ] && sysctl -n machdep.cpu.brand_string

# CUDA toolkits: PATH and /usr/local/cuda are frequently different versions
find /usr/local -maxdepth 1 -name 'cuda*' 2>/dev/null   # not `ls -d /usr/local/cuda*`: zsh aborts the line on an unmatched glob
nvcc --version 2>/dev/null | grep release
/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep release

# existing environment
[ -d .venv ] && .venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
[ -d .venv ] && .venv/bin/python -c "import torch; p=torch.cuda.get_device_properties(0); \
  print(p.name, '%d.%d' % (p.major, p.minor), 'bf16:', torch.cuda.is_bf16_supported())" 2>/dev/null

# existing weights and caches
ls checkpoints checkpoints_2 checkpoints_25 2>/dev/null
du -sh ~/.cache/huggingface/hub 2>/dev/null
df -h . | tail -1
```

Read the exit statuses, not just the output. Every vendor probe above is silenced
with `2>/dev/null`, so "no output" means one of two different things: exit 127 is
*the tool is not installed* (expected on a machine of another vendor), while exit
0 with nothing printed would mean the tool ran and found no device. Same for the
capability probe — it raises on a non-CUDA box and the redirect eats it. And
`torch.version.cuda` printing `None` is correct on macOS, not a fault.

To answer "what already exists vs needs downloading" — which the confirmation
below requires — run the manifest check from step 4 now. It needs only `curl` and
`python3`, no venv, so it works before any install. `ls checkpoints` alone cannot
tell you whether a directory is complete.

Once dependencies exist, `uv run tools/gpu_check.py` enumerates every torch
backend and is the authoritative check.

### Then show the user what you found, and let them correct it

Summarise it back before acting. This costs one message and can save an
afternoon of downloading:

> Found: repo on `main`, one edited file (`webui.py`) plus the expected untracked
> `checkpoints/config.yaml`; uv 0.8.9; RTX 4090 24 GB;
> CUDA 11.5 on PATH but 12.8 at `/usr/local/cuda`; `.venv` with torch 2.8.0+cu128;
> `checkpoints/` complete for 2.5; old HF cache 3.2 GB; 166 GB free.
>
> Plan: reuse the `.venv`, migrate the cached auxiliary models rather than
> re-downloading, leave your uncommitted files alone. Anything wrong?

Ask about what a probe cannot see:

- **Is the whole GPU yours?** `memory.total` reports the card, not what is free. A
  shared box can show 24 GB installed with 6 GB available, which changes every
  decision in step 1.
- **Are there weights elsewhere?** On another volume, or under a name this guide
  does not check. Asking beats re-downloading 5 GB.
- **Which version do they want** — 2.5, 2.0, or both. Fetching the wrong one is the
  most expensive mistake available here.
- **Is the CUDA reading right?** If `PATH` and `/usr/local/cuda` disagree, say which
  you intend to use.

One `git status` entry is a false alarm, so don't hand it back as a user edit:
`.gitignore` ignores `/checkpoints*/*` but re-includes `*.yaml`, so a config file
inside a weights directory always shows up as `?? checkpoints/config.yaml`. It is
normal, and it is not something they changed.

Take corrections at face value: if they say the weights are at `/data/tts/ckpt`,
point `--model_dir` there or symlink it rather than re-fetching.

## 1. Requirements

Only disk is a hard precondition. The rest you either install yourself or work
around, so do not turn any of them into a reason to stop.

| Requirement | Really? |
| --- | --- |
| Disk, ~18 GB standing | yes — see the sizes below |
| Python `>=3.10,<3.12` | pinned in `pyproject.toml`, but `uv` downloads its own interpreter, so the system Python is irrelevant |
| `uv` | needed, and you install it: `pip install -U uv`, or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| CUDA Toolkit 12.8+ | only to *compile*; the torch wheels bundle their own CUDA runtime |
| A GPU at all | no — `indextts/infer_v2_5.py:94-114` falls through CUDA → XPU → MPS → CPU |
| ~6 GB VRAM | for a CUDA run, and only with half precision available — see Small GPUs |

**Disk, measured on a Linux CUDA box:** 9.5 GB for the venv with all extras
(8.5 GB with only `webui`), 5.1 GB for the 2.5 weights, 2.8 GB for the auxiliary
models that land in `checkpoints/hf_cache/`. That is ~18 GB standing. A cold uv
cache adds ~11 GB while installing, so budget ~29 GB of headroom when
`~/.cache/uv` shares the filesystem — `uv cache prune` reclaims that part.

The venv figure is dominated by CUDA wheels, so it does not transfer: the same
environment on Apple Silicon measured 1.7 GB, making the standing total ~10 GB
rather than ~18 GB. Off-CUDA, size the venv from what `uv sync --dry-run`
actually lists.

Those figures are for a **fresh** install. Count only what step 0 found missing:
an existing `.venv` or a complete `checkpoints/` means most of it is already
paid for, and a machine with 20 GB free can be entirely ready. Do not report a
disk blocker without subtracting what is already on disk.

If there is no usable CUDA device, say so *before* downloading 5 GB of weights
and let the user decide — CPU and MPS both work, just slowly. Don't decide for
them in either direction.

A box often has several toolkits, and `nvcc` on `PATH` is frequently the oldest
of them — `/usr/bin/nvcc` at 11.5 while `/usr/local/cuda` points at 12.8 is
normal. Do not conclude the machine is too old from the `PATH` one alone. What
matters: `torch.version.cuda` for wheel selection, and `CUDA_HOME` for anything
that compiles. Export `CUDA_HOME=/usr/local/cuda-12.8` (adjust to what step 0
found) rather than relying on `PATH`. On a 4090 with the 11.5 `nvcc` in front,
the BigVGAN kernel build fails with `nvcc fatal : Unsupported gpu architecture
'compute_89'`; that one is survivable — the code prints `Failed to load custom
CUDA kernel for BigVGAN. Falling back to torch.` and inference continues — but
it is the same root cause and it will bite something else later.

### Small GPUs

Every figure in this subsection is **dedicated NVIDIA VRAM**. They do not map
onto Apple Silicon's unified memory, which is shared with the OS and the display;
for MPS, see the non-NVIDIA subsection below instead of applying these numbers.

Step 0 already reports `memory.total`. Act on it — the defaults are tuned for a
big card, and the two settings that matter are precision and whether QwenEmotion
is loaded. Half precision without QwenEmotion peaks around 5.5 GB; the full
default configuration needs about 8 GB.

So a 6 GB card runs it, but only in half precision with QwenEmotion skipped. An
8 GB card fits the full configuration with nothing left over for the display
output and driver overhead it also has to fund. Below about 5.5 GB it does not
run at all.

`webui.py` applies this automatically below 10 GB — half precision, QwenEmotion
skipped, and emotion-control-from-text removed from the UI — and prints what it
decided. `--qwen_emo` forces the model back on, which does not fit in 6 GB.

Driving the Python API directly, make the same choice yourself — and guard the
precision flag rather than passing `True` blindly, because **bf16 needs Ampere or
newer**. A pre-Ampere card (GTX 1660, RTX 2060) falls back to fp32 and then needs
about 7 GB even without QwenEmotion. That is the trap: the cards most likely to be
short on memory are the ones that cannot have the setting that would save it. So a
6 GB Turing card cannot run this while a 6 GB Ampere card can — read the compute
capability, not just the memory size.

```python
import torch
from indextts.infer_v2_5 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    use_qwen_emo=False,   # skip the ~1.1 GB emotion model
)
```

On a non-CUDA backend that expression is False anyway, and `use_bf16` is ignored —
`indextts/infer_v2_5.py:108` overrides it for MPS regardless of what you pass.

`webui.py` prints `>> BF16 is not supported on this device, falling back to full
precision.` when that applies, so its startup output tells you which precision you
actually got.

### AMD, Intel, Apple Silicon and CPU

The inference code is not NVIDIA-only: `indextts/infer_v2_5.py:94-114` selects CUDA, then
Intel XPU, then Apple MPS, then CPU, and `tools/gpu_check.py` enumerates all of
them. **The dependency configuration is the narrow part**, not the model code —
the two need different fixes, so don't conflate them.

**On Linux, `uv sync` installs a CUDA build regardless of GPU vendor.**
`[tool.uv.sources]` pins torch to the `cu128` index for `sys_platform == 'linux'
or 'win32'` with no ROCm or XPU alternative (an attempt to add `--extra cu128/xpu`,
#732, was closed). On an AMD or Intel GPU the install therefore succeeds and then
`torch.cuda.is_available()` is False, and inference silently runs on CPU. Install a
vendor torch over the top afterwards and confirm with `uv run tools/gpu_check.py`
before concluding anything. `--extra accel` is useless there — flash-attn is
CUDA-only.

**On Apple Silicon, `uv sync --all-extras` fails outright:**

```
error: Distribution `nvidia-cuda-runtime-cu12==12.8.90` can't be installed
because it doesn't have a source distribution or wheel for the current platform
```

`accel` is the only extra that cannot resolve; `webui`, `torch_compile`,
`deepspeed` and `test` all do. Use `uv sync --extra webui --extra test` and never
`accel`. MPS then runs fp32 — `indextts/infer_v2_5.py:107-109` sets `use_bf16 = False`
there deliberately, so passing `use_bf16=True` has no effect.

It does work end to end on MPS, but throughput is far below CUDA and varies a lot
by chip, so measure on the actual machine before promising anything. Give the user
a number from their hardware, not a claim that "it works on Apple Silicon".

ROCm and XPU are code paths, not measured configurations — for those, "runs" means
the code selects the device. Say that rather than implying otherwise.

**Memory off CUDA gets no automatic protection.** `detect_vram_gb()` in `webui.py`
returns `None` whenever `torch.cuda.is_available()` is False, so `LOW_VRAM` is
False and the low-VRAM adaptation never engages on MPS, XPU or CPU — a 8 GB Mac
loads the full default configuration, fp32 with QwenEmotion, and nothing warns
you. The VRAM thresholds above are CUDA-only in implementation, not just in their
numbers. If a unified-memory machine runs out, pass `use_qwen_emo=False` yourself
through the Python API; there is no WebUI flag that turns it off.

## 2. Update the code

**Their edits are theirs. Never discard or overwrite one without showing the
user what it is and getting an answer first.** Someone upgrading from
IndexTTS-2 may have been running patched inference code for months.

```bash
git stash list                       # pre-existing stashes are not yours to pop
git status --short                   # any output here means: stop and read it
git diff                             # show the user what they changed
git log --oneline @{u}.. 2>/dev/null # unpushed commits; silent if the branch has no upstream
```

If the tree is clean, just update:

```bash
git pull --ff-only
```

If it is dirty, tell the user what they have modified and let them choose. Git
itself will not silently clobber a tracked edit — `pull --ff-only` aborts with
`error: Your local changes to the following files would be overwritten by
merge` and applies nothing. The danger is you "fixing" that abort. **Never run
`git reset --hard`, `git checkout -- .`, `git clean -fd`, or `git stash drop`
to get past it.** That is the one irreversible thing in this whole guide.

To keep the edits and still update, stash and restore in the same breath:

```bash
git stash push -u -m "pre-update, agent"
git pull --ff-only
git stash pop
```

`pop` can end in `CONFLICT (content): Merge conflict in <file>`, leaving
`<<<<<<<` markers in the file. That is recoverable — git keeps the stash entry
on conflict (`git stash list` still shows it) and untracked files come back
too — but it is the user's merge to make, not yours. Report the conflicted
paths and stop. Do not resolve their code by guessing which side to keep.

If they would rather not update at all, that is a valid answer: an older
checkout with a working `.venv` still runs. Skip to step 4.

Fresh machine:

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
pip install -U uv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

Git LFS is not required.

## 3. Dependencies

```bash
uv sync --all-extras
```

**Not on an NVIDIA box?** That command fails outright on Apple Silicon and gives
AMD/Intel users a CUDA-only torch — read the non-NVIDIA subsection in step 1
first and use the command it gives instead.

That creates `.venv`, picks the interpreter, and installs from `uv.lock`. It is
incremental: an existing `.venv` is reconciled, not rebuilt. Extras are
`webui`, `deepspeed`, `accel`, `torch_compile`, `test`; drop `--all-extras` and
name individual ones with `--extra <name>` if the user wants a smaller install.

**Name every extra you want in one command.** `uv sync` makes the environment
match exactly what you asked for, so `--extra webui` alone *uninstalls* whatever
the other extras had brought in — measured here as 3 packages removed, and up to
30 for other single-extra combinations. Running two syncs in sequence churns the
venv both ways and undoes the previous one.

Reusing an env from IndexTTS-2: run `uv sync` in the updated checkout and let it
converge. Do not `pip install` into `.venv` by hand, and do not activate a conda
env first — `uv` manages the environment itself and a pre-activated one causes
conflicts.

Verify, then stop touching dependencies:

```bash
# repeat the SAME extras you installed with — on Apple Silicon that is
# `uv sync --extra webui --extra test --dry-run`, never --all-extras
uv sync --all-extras --dry-run       # expect "Would make no changes"
uv run tools/gpu_check.py            # expect your accelerator listed
```

`uv lock --check` only compares `pyproject.toml` against `uv.lock`. It says
nothing about what is installed — use `uv sync --dry-run` for that.

### Slow networks (mainland China)

```bash
export UV_HTTP_TIMEOUT=900
uv sync --all-extras --default-index "https://mirrors.cloud.tencent.com/pypi/simple"
export HF_ENDPOINT=https://hf-mirror.com     # for model downloads in step 4
```

For the torch wheels specifically, edit the `url` inside the
`[[tool.uv.index]]` block in `pyproject.toml` (e.g. to
`https://mirror.nju.edu.cn/pytorch/whl/cu128`). Do **not** set that mirror via
`UV_INDEX="pytorch-cuda=..."` — that drops `explicit = true`, turns the torch
mirror into a general index, and its stale copies of unrelated packages will
fabricate resolution conflicts.

### If a build fails

Errors naming `flash-attn` or `deepspeed` are build-environment problems, not
resolution problems. Both are source distributions that need the project's
CUDA-enabled torch at build time; `pyproject.toml` handles this with
`[tool.uv.extra-build-dependencies]`. Two failure signatures:

- `ModuleNotFoundError: No module named 'setuptools'` — something switched these
  packages to `no-build-isolation`. That mode uses `.venv` as the build
  environment, and a fresh `.venv` has no `setuptools` yet. Keep
  `extra-build-dependencies` instead.
- a long `nvcc` compile for flash-attn — it should download a prebuilt wheel
  matching your torch version and C++ ABI. Compiling means the guessed wheel
  name missed. Check `CUDA_HOME` points at 12.8 and that GitHub releases are
  reachable.

`accel` (flash-attn) and `torch_compile` are optional speedups. If they will not
build, drop those extras and continue — the model runs without them.

## 4. Models — check before downloading

IndexTTS-2.5 is 22 files / ~5.1 GiB. Auxiliary models (w2v-bert-2.0, MaskGCT
codec, CAMPPlus, BigVGAN) are **not** in that repo; they land in
`{model_dir}/hf_cache/` on first run.

```bash
uv tool install "huggingface-hub"
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints
```

or

```bash
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints
```

Verify against the remote manifest instead of re-downloading blind:

```bash
curl -sf "https://huggingface.co/api/models/IndexTeam/IndexTTS-2.5/tree/main?recursive=true" \
  -o /tmp/tree.json || echo "API UNREACHABLE — see the mirror note below"
python3 -c 'import json,os
SKIP={".gitattributes","LICENSE","README.md"}
for f in json.load(open("/tmp/tree.json")):
    if f["type"]!="file" or f["path"] in SKIP: continue
    want=(f.get("lfs") or {}).get("size") or f.get("size") or 0
    p=os.path.join("checkpoints",f["path"])
    have=os.path.getsize(p) if os.path.exists(p) else -1
    if have!=want: print("REDOWNLOAD", f["path"], want, have)'
```

Two commands rather than a pipeline on purpose: a failed `curl` inside a pipe
leaves you reading the *python* exit status, and the portable way to recover the
first stage differs by shell (`$PIPESTATUS` in bash, `$pipestatus` in zsh). Here
`curl -sf` announces its own failure, so no `REDOWNLOAD` output genuinely means
the local copy matches — skip the download entirely. The `SKIP` set
matters: `hf download` fetches the repo's `.gitattributes`, `LICENSE` and
`README.md` too, and a directory assembled any other way will not have them. They
have nothing to do with inference, so without that filter a complete set of
weights reports three spurious `REDOWNLOAD` lines and the whole check reads as a
failure.

If that URL times out, swap `huggingface.co` for `hf-mirror.com` — the API host is
blocked on some networks even when downloads work.

### Reusing caches from an IndexTTS-2 install

**First, check whether this applies at all** — it needs no venv:

```bash
ls -l checkpoints/hf_cache/w2v-bert-2.0/model.safetensors \
      checkpoints/hf_cache/bigvgan/bigvgan_generator.pt 2>/dev/null
ls -d ~/.cache/huggingface/hub/models--facebook--w2v-bert-2.0 2>/dev/null
```

If the first command lists two multi-hundred-MB files, the auxiliary models are
already in place and this whole subsection is moot — skip it. If the second
command finds nothing, there is no old cache to reuse; skip it too.

Also confirm the old cache holds real weights rather than an empty shell. A
snapshot directory can contain only symlinks into a `blobs/` directory that was
never filled, in which case migrating it produces an empty directory and the
loader then treats that as complete:

```bash
du -shL ~/.cache/huggingface/hub/models--facebook--w2v-bert-2.0/snapshots/*/ 2>/dev/null
```

Kilobytes means hollow — download that component instead of migrating it.

The migration itself runs **after step 3**, because it imports from the project.
That is still "before any download": on first run the model code fetches these
same files itself, and this replaces that fetch.

`ensure_models_available()` in `indextts/utils/model_download.py` searches
`{model_dir}/hf_cache/` and then `$HF_HUB_CACHE` (default
`~/.cache/huggingface/hub`) for the old
`models--{owner}--{name}/snapshots/{hash}/` layout, and copies what it finds
instead of downloading.

Exporting `HF_HUB_CACHE` on its own is not enough, because
`indextts/infer_v2_5.py:4` overwrites it at import time:

    os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

That line runs before anything looks at the variable, and it applies to both
entrypoints — `webui.py` imports `infer_v2_5` too — so an exported value is
gone by the time the search happens, and the migration finds nothing.

Run the migration as its own step first, importing only the download helper:

```bash
HF_HUB_CACHE=~/.cache/huggingface/hub uv run python -c \
  "from indextts.utils.model_download import ensure_models_available; \
   ensure_models_available('checkpoints')"
```

Expect one `>> Migrating ...` line per component — the wording varies, two of the
three append `to <dir>...`, so match on `Migrating` rather than the whole string —
and a closing `>> All auxiliary models ready.`. Measured on a box with a populated old
cache: 2.8 GB migrated in about two seconds, against several minutes of
downloading. Afterwards `checkpoints/hf_cache/` is populated, so the later steps
find it there and never consult `$HF_HUB_CACHE` at all.

It **copies** rather than symlinks, so budget disk for a second copy of
w2v-bert-2.0 (~2.2 GB).

Keeping 2.0 alongside 2.5: 2.5 weights in `checkpoints/`, 2.0 weights in
`checkpoints_2/`. Nothing is shared between the two, and neither overwrites the
other. If disk is tight, one `checkpoints*` tree can be a symlink to weights
already on another volume.

## 5. Example audio

`examples/*.wav` is not tracked in git. It is fetched automatically the first
time the WebUI starts. The `infer_v2_5.py` entrypoint does **not** fetch it, and
its default `--prompt_wav` is `examples/voice_01.wav`, so a script run on a
fresh clone fails with `FileNotFoundError`. Fetch it first:

```bash
uv run python -c "from indextts.utils.examples_downloader import ensure_examples_available; ensure_examples_available()"
```

Idempotent — existing files are skipped.

## 6. Smoke test

```bash
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2_5.py \
  --cfg_path checkpoints/config.yaml \
  --model_dir checkpoints \
  --text "Hello world" \
  --lang EN
```

Success looks like `>> wav file saved to: gen.wav` plus an `RTF:` line. Confirm
`gen.wav` is non-trivial (tens of KB, not 0). On first run this also populates
`hf_cache/`, which takes several GB and several minutes.

`--lang` accepts `ZH`, `EN`, `JA`, `ES`, `AR` (case-insensitive) and is **not
validated**: `lang_to_token()` in `indextts/utils/tokenizer.py:173-177` silently
falls back to a generic token for anything it does not recognise. A typo produces
worse audio, not an error — so if output quality is off, check this value before
suspecting the weights.

Then the WebUI:

```bash
uv run webui.py            # 2.5, default
```

Serves on port 7860, and binds `0.0.0.0` by default (`webui.py:26`) — reachable
from the network, not just localhost — with no authentication. On a shared or
internet-reachable machine pass `--host 127.0.0.1`, or put it behind something
that authenticates. Report this rather than silently exposing it.

## Known traps

| Symptom | Cause | Action |
| --- | --- | --- |
| `FileNotFoundError: examples/voice_01.wav` | examples are WebUI-fetched only | step 5 |
| `ValueError: vocab_file checkpoints/bpe.model does not exist` | `indextts/infer_v2.py` hardcodes `checkpoints/` and is a benchmark loop, not a CLI | use the Python API for 2.0, or point `checkpoints/` at 2.0 weights |
| `HTTPError: <Response [404]>` mentioning `nvidia/bigvgan_*` | BigVGAN is absent from ModelScope | benign — the code falls back to hf-mirror and continues; check for a later `>> All auxiliary models ready.` |
| `ModuleNotFoundError: No module named 'setuptools'` while building | `no-build-isolation` on a fresh `.venv` | see step 3 |
| `torch.cuda.OutOfMemoryError` at load or first inference | card too small for the active configuration | half precision + `use_qwen_emo=False`, see step 1; check nothing else is holding the GPU (`nvidia-smi`) |
| `triton-windows ... only has wheels for win_amd64` | that package is Windows-only | it must carry `sys_platform == 'win32'`; Linux gets `triton` via torch |
| `does not have an extra named 'cli'` | modern `huggingface-hub` dropped it | install it plain: `huggingface-hub` |
| `uv pip check` reports `deepspeed requires nvidia-ml-py` | upstream declares it, the lockfile omits it | pre-existing and harmless for inference |
| `warning: The extra-build-dependencies option is experimental` on every uv command | it is experimental, and `pyproject.toml` uses it deliberately | benign, ignore it. Do **not** add `preview-features` to `[tool.uv]` to silence it — that is not a valid key, and an invalid key makes uv discard the whole `[tool.uv]` table |

## What to report when you finish

Lead with whether it works, then how to use it, then what the user needs to know
or decide. Six parts, in this order:

1. **Verdict** — did the smoke test produce audio? Give the file, its size and
   the `RTF:` figure. If it did not, say that instead of describing the setup as
   complete.
2. **Commands to run it again** — the WebUI line and the inference snippet, with
   the paths this machine actually uses.
3. **Configuration in effect** — model version, precision, extras installed, and
   whether QwenEmotion is loaded. This determines which features exist, so it is
   not optional detail.
4. **Reused vs installed vs downloaded** — the point of this whole guide. Name
   what you skipped and roughly what that saved.
5. **Needs the user's attention** — their uncommitted edits, a WebUI bound to
   `0.0.0.0`, a card near its VRAM limit, anything you could not verify.
6. **Skipped or unverified** — say so explicitly rather than leaving it implied.

A report that reads like this:

> **Working.** `gen.wav`, 75 KB, RTF 0.71 on an RTX 4090.
>
> ```bash
> uv run webui.py                      # http://127.0.0.1:7860
> PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2_5.py \
>   --cfg_path checkpoints/config.yaml --model_dir checkpoints \
>   --text "Hello world" --lang EN
> ```
>
> IndexTTS-2.5, bf16, extras `webui,accel`. QwenEmotion loaded, so
> emotion-from-text works.
>
> Reused your existing `.venv` (`uv sync` changed 3 packages) and migrated
> w2v-bert-2.0, MaskGCT, CAMPPlus and BigVGAN out of `~/.cache/huggingface/hub`
> — 2.8 GB not re-downloaded. Fetched only the 2.5 weights, 5.1 GiB.
>
> Two things for you: `webui.py` binds `0.0.0.0` with no authentication, so pass
> `--host 127.0.0.1` if this box is reachable from outside. And you have
> uncommitted changes in `indextts/infer_v2_5.py` — I left them alone and did
> not pull.
>
> Not done: never started the 2.0 path, so `checkpoints_2` is still absent.

Do not report a configuration you did not run. If you only checked that a
command exists, say that, not that it works.
