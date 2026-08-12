# Agent Setup Guide

Instructions for a coding agent (Claude Code, Cursor, Codex, …) asked to get
IndexTTS running on a user's machine. A user may hand you only the URL of this
file; everything you need is here.

Your goal: a working inference run. Your constraint: **do not redo work the
machine has already done.** A full setup downloads ~10 GB of wheels and ~5 GB of
weights. Users upgrading from IndexTTS-2 usually have most of it already.

Never guess state — run the probe commands and read the output.

## 0. Probe before you touch anything

```bash
# repo
git -C . rev-parse --abbrev-ref HEAD 2>/dev/null; git -C . log --oneline -1 2>/dev/null
git -C . status --short

# toolchain
uv --version                     # required; see step 2 if missing
python3 -VV
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
ls -d /usr/local/cuda* 2>/dev/null
nvcc --version 2>/dev/null | grep release              # the one on PATH
/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep release   # often a different one

# existing env
[ -d .venv ] && .venv/bin/python -c "import torch;print(torch.__version__, torch.version.cuda)" 2>/dev/null

# existing weights, anywhere on the box
ls checkpoints checkpoints_2 checkpoints_25 2>/dev/null
du -sh ~/.cache/huggingface/hub 2>/dev/null
ls ~/.cache/huggingface/hub 2>/dev/null | grep -iE "w2v-bert|maskgct|campplus|bigvgan"
df -h . | tail -1
```

Record: CUDA toolkit version, free disk, whether a `.venv` exists, and which
model files already exist. Decide the branch points in steps 3–5 from that.

## 1. Requirements that are not negotiable

| Requirement | Why |
| --- | --- |
| Python `>=3.10,<3.12` | `pyproject.toml` pins it; `uv` installs a matching interpreter itself |
| CUDA Toolkit **12.8+** | only to *compile* anything; the torch wheels bundle their own CUDA runtime |
| `uv` | the lockfile is the only supported dependency path |
| ~35 GB free disk | ~10 GB venv + ~5 GB weights (2.5) + build/cache overhead |
| ~6 GB VRAM | measured floor, in half precision with QwenEmotion skipped — see below |

CPU-only and Apple Silicon can install, but inference expects CUDA. Say so
early rather than after a 20-minute download.

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

Step 0 already reports `memory.total`. Act on it — the defaults are tuned for a
big card. Peak reserved memory for one short utterance, measured on a 4090:

| Configuration | Peak |
| --- | --- |
| full precision + QwenEmotion | 8.15 GB |
| half precision + QwenEmotion | 6.54 GB |
| half precision, QwenEmotion skipped | 5.48 GB |

Re-run under a hard per-process cap (`torch.cuda.set_per_process_memory_fraction`):

| Cap | full + Qwen | half + Qwen | half, no Qwen |
| --- | --- | --- | --- |
| 8 GB | ok, but peaks at exactly 8.00 GB | ok (6.56 GB) | ok (5.45 GB) |
| 6 GB | OOM | OOM | ok (5.44 GB) |
| 5 GB | — | — | OOM |

So an 8 GB card runs the full configuration with no headroom left for the
display output and driver overhead it also has to fund, and a 6 GB card runs
only the last column. The floor is between 5 and 6 GB.

`webui.py` applies this automatically below 10 GB — half precision, QwenEmotion
skipped, and emotion-control-from-text removed from the UI — and prints what it
decided. `--qwen_emo` forces the model back on, which will not fit in 6 GB.

Driving the Python API directly, make the same choice yourself:

```python
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_bf16=True,        # half precision
    use_qwen_emo=False,   # skip the ~1.1 GB emotion model
)
```

Those two measurement runs at a 6 GB cap happened on a box where only ~6.2 GB
was physically free, so the cap was not the only binding constraint. The
conclusion holds on arithmetic regardless: those configurations need 7.70 GB and
6.13 GB just to load.

## 2. Update the code

If a repo already exists, do not clone over it — the user may have local edits
and a warm `.venv` next to it.

**Their edits are theirs. Never discard or overwrite one without showing the
user what it is and getting an answer first.** Someone upgrading from
IndexTTS-2 may have been running patched inference code for months.

```bash
git stash list                       # pre-existing stashes are not yours to pop
git status --short                   # any output here means: stop and read it
git diff                             # show the user what they changed
git log --oneline @{u}..             # local commits not pushed anywhere
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

That creates `.venv`, picks the interpreter, and installs from `uv.lock`. It is
incremental: an existing `.venv` is reconciled, not rebuilt. Extras are
`webui`, `deepspeed`, `accel`, `torch_compile`, `test`; drop `--all-extras` and
name individual ones with `--extra <name>` if the user wants a smaller install.

Reusing an env from IndexTTS-2: run `uv sync` in the updated checkout and let it
converge. Do not `pip install` into `.venv` by hand, and do not activate a conda
env first — `uv` manages the environment itself and a pre-activated one causes
conflicts.

Verify, then stop touching dependencies:

```bash
uv sync --all-extras --dry-run       # expect "Would make no changes"
uv run tools/gpu_check.py            # expect your GPUs listed
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
curl -s "https://huggingface.co/api/models/IndexTeam/IndexTTS-2.5/tree/main?recursive=true" \
  | python3 -c 'import json,os,sys
for f in json.load(sys.stdin):
    if f["type"]!="file": continue
    want=(f.get("lfs") or {}).get("size") or f.get("size") or 0
    p=os.path.join("checkpoints",f["path"])
    have=os.path.getsize(p) if os.path.exists(p) else -1
    if have!=want: print("REDOWNLOAD", f["path"], want, have)'
```

Silence means the local copy matches; skip the download entirely. If that URL
times out, swap `huggingface.co` for `hf-mirror.com` — the API host is blocked
on some networks even when downloads work.

### Reusing caches from an IndexTTS-2 install

Do this before any auxiliary download — measured at 2.77 GiB migrated in 2.1s,
versus several minutes of downloading.

`ensure_models_available()` in `indextts/utils/model_download.py` searches
`{model_dir}/hf_cache/` and then `$HF_HUB_CACHE` (default
`~/.cache/huggingface/hub`) for the old
`models--{owner}--{name}/snapshots/{hash}/` layout, and copies what it finds
instead of downloading. So if the user has an old HuggingFace cache, just point
at it:

    export HF_HUB_CACHE=~/.cache/huggingface/hub

NOTE: `indextts/infer_v2_5.py` currently sets `HF_HUB_CACHE=./checkpoints/hf_cache` internally, so this affects `webui.py` but not the step 6 smoke test.

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

Then the WebUI:

```bash
uv run webui.py            # 2.5, default
```

Serves on `http://localhost:7860` (local access). Note it binds `0.0.0.0` by default and has no
authentication — on a shared or internet-reachable machine, pass
`--host 127.0.0.1` to restrict it to local-only, or put it behind something that authenticates.
Report this to the user rather than silently exposing it.

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
| `unknown field 'preview-features'` | not a valid `[tool.uv]` key | remove it; the whole `[tool.uv]` table is ignored while it is present |

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
