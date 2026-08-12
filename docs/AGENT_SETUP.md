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
ls -d .venv && .venv/bin/python -c "import torch;print(torch.__version__, torch.version.cuda)" 2>/dev/null

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
| CUDA Toolkit **12.8+** | wheels come from a `cu128` index |
| `uv` | the lockfile is the only supported dependency path |
| ~35 GB free disk | ~10 GB venv + ~5 GB weights (2.5) + build/cache overhead |

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

```bash
export HF_HUB_CACHE=~/.cache/huggingface/hub
```

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

Serves on `http://127.0.0.1:7860`. Note it binds `0.0.0.0` by default and has no
authentication — on a shared or internet-reachable machine, pass
`--host 127.0.0.1` or put it behind something that authenticates. Report this to
the user rather than silently exposing it.

## Known traps

| Symptom | Cause | Action |
| --- | --- | --- |
| `FileNotFoundError: examples/voice_01.wav` | examples are WebUI-fetched only | step 5 |
| `ValueError: vocab_file checkpoints/bpe.model does not exist` | `indextts/infer_v2.py` hardcodes `checkpoints/` and is a benchmark loop, not a CLI | use the Python API for 2.0, or point `checkpoints/` at 2.0 weights |
| `HTTPError: <Response [404]>` mentioning `nvidia/bigvgan_*` | BigVGAN is absent from ModelScope | benign — the code falls back to hf-mirror and continues; check for a later `>> All auxiliary models ready.` |
| `ModuleNotFoundError: No module named 'setuptools'` while building | `no-build-isolation` on a fresh `.venv` | see step 3 |
| `triton-windows ... only has wheels for win_amd64` | that package is Windows-only | it must carry `sys_platform == 'win32'`; Linux gets `triton` via torch |
| `does not have an extra named 'cli'` | modern `huggingface-hub` dropped it | install it plain: `huggingface-hub` |
| `uv pip check` reports `deepspeed requires nvidia-ml-py` | upstream declares it, the lockfile omits it | pre-existing and harmless for inference |
| `unknown field 'preview-features'` | not a valid `[tool.uv]` key | remove it; the whole `[tool.uv]` table is ignored while it is present |

## Reporting back

State plainly: what you reused vs installed, what you downloaded, which
verification commands you ran and their result, and anything you skipped. If the
smoke test did not produce audio, say that instead of describing the setup as
complete.
