"""
Model download utility that automatically switches between HuggingFace Hub and
ModelScope based on the detected network environment.

All auxiliary models are downloaded to ``{model_dir}/hf_cache/`` at startup
via ``ensure_models_available()``, so no downloads happen during inference.
"""

import logging
import os
import shutil

logger = logging.getLogger(__name__)

from indextts.utils.network_detection import need_proxy

_USING_MODELSCOPE: bool | None = None  # lazily computed on first use


def _get_using_modelscope() -> bool:
    global _USING_MODELSCOPE
    if _USING_MODELSCOPE is None:
        _USING_MODELSCOPE = need_proxy()
    return _USING_MODELSCOPE

# Mapping from HuggingFace repo_id to ModelScope model_id.
HF_TO_MODELSCOPE_REPO_MAP = {
    "funasr/campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
    "facebook/w2v-bert-2.0": "AI-ModelScope/w2v-bert-2.0",
}

# Default BigVGAN repo (also in config.yaml, but needed for pre-download)
_BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"


def _download_single_file(repo_id: str, filename: str, local_path: str) -> str:
    """Download a single file from a HF/ModelScope repo to a specific local path."""
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    if _get_using_modelscope():
        ms_model_id = HF_TO_MODELSCOPE_REPO_MAP.get(repo_id, repo_id)
        # Try ModelScope SDK first
        try:
            from modelscope.hub.file_download import model_file_download
            downloaded_path = model_file_download(
                model_id=ms_model_id, file_path=filename, local_dir=local_dir,
            )
            if not downloaded_path or not os.path.isfile(downloaded_path):
                downloaded_path = os.path.join(local_dir, filename)
            if os.path.abspath(downloaded_path) != os.path.abspath(local_path):
                shutil.copy2(downloaded_path, local_path)
            if not os.path.isfile(local_path):
                raise RuntimeError(f"Downloaded file not found at expected path: {local_path}")
            return local_path
        except Exception as e:
            logger.warning(
                f"ModelScope download failed for {ms_model_id}/{filename}: {e}. Falling back to hf-mirror.",
                exc_info=True,
            )
        # Fallback to hf-mirror.com (only path that needs manual download)
        from indextts.utils.examples_downloader import _download_file
        url = f"https://hf-mirror.com/{repo_id}/resolve/main/{filename}"
        logger.info(f"Downloading {repo_id}/{filename} from hf-mirror -> {local_path}")
        _download_file(url, local_path, timeout=300)
    else:
        # Use HuggingFace Hub SDK
        from huggingface_hub import hf_hub_download
        logger.info(f"Downloading {repo_id}/{filename} -> {local_path}")
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        if downloaded_path and os.path.abspath(downloaded_path) != os.path.abspath(local_path):
            shutil.copy2(downloaded_path, local_path)
        elif not os.path.isfile(local_path):
            fallback_path = os.path.join(local_dir, filename)
            if os.path.isfile(fallback_path):
                shutil.copy2(fallback_path, local_path)
        if not os.path.isfile(local_path):
            raise RuntimeError(f"Downloaded file not found at expected path: {local_path}")

    return local_path


def ensure_config_available(model_dir: str) -> None:
    """Download only ``config.yaml`` if it is missing from *model_dir*."""
    model_dir = model_dir or "."
    config_path = os.path.join(model_dir, "config.yaml")
    if os.path.isfile(config_path):
        return
    print(f">> config.yaml not found in {model_dir}, downloading...")
    _download_single_file("IndexTeam/IndexTTS-2", "config.yaml", config_path)
    print(">> config.yaml downloaded.")


def _find_hf_cache_snapshot(cache_dir: str, repo_id: str) -> str | None:
    """Locate *repo_id* in a HuggingFace Hub cache directory."""
    repo_key = repo_id.replace("/", "--")
    models_dir = os.path.join(cache_dir, f"models--{repo_key}")
    if not os.path.isdir(models_dir):
        return None
    snapshots_dir = os.path.join(models_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    # Prefer the commit hash recorded in refs/main
    refs_main = os.path.join(models_dir, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            commit_hash = f.read().strip()
        snapshot = os.path.join(snapshots_dir, commit_hash)
        if os.path.isdir(snapshot) and os.listdir(snapshot):
            return snapshot

    # Some caches may not have refs/main. Reuse only when there is no ambiguity.
    try:
        entries = [
            os.path.join(snapshots_dir, e)
            for e in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, e))
        ]
        if len(entries) == 1:
            return entries[0]
    except OSError:
        pass
    return None


def _locate_snapshot(repo_id: str, local_cache_dir: str) -> str | None:
    """Search local and default HuggingFace caches for *repo_id*."""
    default_hf_cache = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    search_dirs = [local_cache_dir]
    if os.path.abspath(default_hf_cache) != os.path.abspath(local_cache_dir):
        search_dirs.append(default_hf_cache)
    for d in search_dirs:
        snapshot = _find_hf_cache_snapshot(d, repo_id)
        if snapshot:
            return snapshot
    return None


def ensure_models_available(model_dir: str, bigvgan_repo: str = _BIGVGAN_REPO) -> dict:
    """
    Download all auxiliary models to ``{model_dir}/hf_cache/`` if missing.

    If files already exist in the old standard HuggingFace Hub cache layout
    (``models--{owner}--{name}/snapshots/{hash}/``), they are copied into the
    new flat layout instead of being re-downloaded.

    Call this once at startup before creating ``IndexTTS2``.

    Returns a dict of local paths:
        - ``w2v_bert``: directory containing w2v-bert-2.0 model
        - ``semantic_codec``: path to semantic_codec/model.safetensors
        - ``campplus``: path to campplus_cn_common.bin
        - ``bigvgan``: directory containing config.json + bigvgan_generator.pt
    """
    cache_dir = os.path.join(model_dir, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}

    # w2v-bert-2.0 is a full repo needed by SeamlessM4T and Wav2Vec2BertModel.
    w2v_dir = os.path.join(cache_dir, "w2v-bert-2.0")
    if not os.path.isdir(w2v_dir) or not os.listdir(w2v_dir):
        old_snapshot = _locate_snapshot("facebook/w2v-bert-2.0", cache_dir)
        if old_snapshot:
            print(f">> Migrating w2v-bert-2.0 from existing HF cache to {w2v_dir}...")
            shutil.copytree(old_snapshot, w2v_dir, dirs_exist_ok=True)
        else:
            print(f">> Downloading w2v-bert-2.0 to {w2v_dir}...")
            snapshot_download("facebook/w2v-bert-2.0", local_dir=w2v_dir)
    paths["w2v_bert"] = w2v_dir

    for key, repo_id, remote_file, local_file, label in (
        (
            "semantic_codec",
            "amphion/MaskGCT",
            "semantic_codec/model.safetensors",
            os.path.join(cache_dir, "semantic_codec_model.safetensors"),
            "MaskGCT semantic codec",
        ),
        (
            "campplus",
            "funasr/campplus",
            "campplus_cn_common.bin",
            os.path.join(cache_dir, "campplus_cn_common.bin"),
            "CAMPPlus",
        ),
    ):
        if not os.path.isfile(local_file):
            old_snapshot = _locate_snapshot(repo_id, cache_dir)
            old_file = os.path.join(old_snapshot, *remote_file.split("/")) if old_snapshot else None
            if old_file and os.path.isfile(old_file):
                print(f">> Migrating {label} from existing HF cache...")
                shutil.copy2(old_file, local_file)
            else:
                print(f">> Downloading {label} to {local_file}...")
                _download_single_file(repo_id, remote_file, local_file)
        paths[key] = local_file

    # BigVGAN vocoder (config + weights)
    bigvgan_dir = os.path.join(cache_dir, "bigvgan")
    bigvgan_files = ("config.json", "bigvgan_generator.pt")
    missing_bigvgan = [
        f for f in bigvgan_files
        if not os.path.isfile(os.path.join(bigvgan_dir, f))
    ]
    if missing_bigvgan:
        old_snapshot = _locate_snapshot(bigvgan_repo, cache_dir)
        if old_snapshot and all(os.path.isfile(os.path.join(old_snapshot, f)) for f in missing_bigvgan):
            print(f">> Migrating BigVGAN from existing HF cache to {bigvgan_dir}...")
            os.makedirs(bigvgan_dir, exist_ok=True)
            for fname in missing_bigvgan:
                src = os.path.join(old_snapshot, fname)
                dst = os.path.join(bigvgan_dir, fname)
                shutil.copy2(src, dst)
        else:
            print(f">> Downloading BigVGAN to {bigvgan_dir}...")
            os.makedirs(bigvgan_dir, exist_ok=True)
            for fname in missing_bigvgan:
                _download_single_file(bigvgan_repo, fname, os.path.join(bigvgan_dir, fname))
    paths["bigvgan"] = bigvgan_dir

    print(">> All auxiliary models ready.")
    return paths


def snapshot_download(repo_id: str, local_dir: str, revision=None, force_download=False, **kwargs) -> str:
    """Download an entire model repository (HuggingFace or ModelScope)."""
    if _get_using_modelscope():
        return _snapshot_from_modelscope(repo_id, local_dir, revision)
    else:
        from huggingface_hub import snapshot_download as _hf_snapshot
        logger.info(f"Downloading repo from HuggingFace: {repo_id}")
        return _hf_snapshot(
            repo_id=repo_id, local_dir=local_dir, revision=revision,
            force_download=force_download, **kwargs,
        )


def _snapshot_from_modelscope(model_id: str, local_dir: str, revision=None) -> str:
    """Download an entire model repository from ModelScope."""
    ms_model_id = HF_TO_MODELSCOPE_REPO_MAP.get(model_id, model_id)
    if ms_model_id != model_id:
        logger.info(f"ModelScope: mapped '{model_id}' -> '{ms_model_id}'")

    from modelscope.hub.snapshot_download import snapshot_download as _ms_snapshot
    logger.info(f"Downloading repo from ModelScope: {ms_model_id}")

    os.makedirs(local_dir, exist_ok=True)
    _ms_snapshot(model_id=ms_model_id, local_dir=local_dir, revision=revision)
    return local_dir
