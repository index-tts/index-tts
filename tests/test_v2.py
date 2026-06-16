"""
IndexTTS v2 tests.

Run with:
    uv run --extra test pytest tests/test_v2.py -v

CI only (no GPU):
    uv run --extra test pytest tests/test_v2.py -v -m "not gpu"
"""
import importlib
import sys
import types
from pathlib import Path

import pytest

CHECKPOINTS_DIR = Path("checkpoints")
CONFIG_PATH = CHECKPOINTS_DIR / "config.yaml"


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture(scope="module")
def tts_model():
    pytest.importorskip("torch")
    if not CONFIG_PATH.exists():
        pytest.skip(f"Checkpoints not found at {CHECKPOINTS_DIR}")

    from indextts.infer_v2 import IndexTTS2
    return IndexTTS2(
        cfg_path=str(CONFIG_PATH),
        model_dir=str(CHECKPOINTS_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
    )


@pytest.fixture(scope="module")
def prompt_wav():
    from indextts.utils.examples_downloader import ensure_test_sample_available
    return ensure_test_sample_available()


# -- Download URL checks (no GPU) ---------------------------------------------

# Each auxiliary model: (test_id, repo_id, probe_file)
# probe_file: a small file in the repo to verify download works end-to-end.
_MODEL_PROBES = [
    ("bigvgan", "nvidia/bigvgan_v2_22khz_80band_256x", "config.json"),
    ("w2v-bert-2.0", "facebook/w2v-bert-2.0", "config.json"),
    ("campplus", "funasr/campplus", "campplus_cn_common.bin"),
    ("MaskGCT", "amphion/MaskGCT", "README.md"),
]


@pytest.mark.parametrize("name,repo_id,filename", _MODEL_PROBES, ids=[m[0] for m in _MODEL_PROBES])
def test_model_download_reachable(name, repo_id, filename, tmp_path):
    """Each auxiliary model must be downloadable via the real download path."""
    from indextts.utils.examples_downloader import _download_file
    from indextts.utils.network_detection import need_proxy

    base_url = "https://hf-mirror.com" if need_proxy() else "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/main/{filename}"
    dest = tmp_path / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download_file(url, str(dest), max_bytes=8192)
    assert dest.exists() and dest.stat().st_size > 0


def test_example_download_reachable(tmp_path, monkeypatch):
    """Example audio must be downloadable via the real download path."""
    from indextts.utils import examples_downloader

    monkeypatch.setattr(examples_downloader, "_TESTS_DIR", str(tmp_path))
    path = examples_downloader.download_test_sample(force=True)
    assert Path(path).exists() and Path(path).stat().st_size > 0


# -- Model download logic (no GPU) --------------------------------------------

def test_legacy_cache_compatibility(tmp_path, monkeypatch):
    """ensure_models_available preserves existing cache and skips re-download."""
    from indextts.utils import model_download
    from indextts.utils.model_download import ensure_models_available

    download_calls = []
    original_download = model_download._download_single_file

    def mock_download(*args, **kwargs):
        download_calls.append(args)
        return original_download(*args, **kwargs)

    monkeypatch.setattr(model_download, "_download_single_file", mock_download)

    model_dir = tmp_path / "checkpoints"
    cache_dir = model_dir / "hf_cache"
    cache_dir.mkdir(parents=True)

    w2v_dir = cache_dir / "w2v-bert-2.0"
    w2v_dir.mkdir()
    (w2v_dir / "config.json").write_text('{"test": true}')

    bigvgan_dir = cache_dir / "bigvgan"
    bigvgan_dir.mkdir()
    (bigvgan_dir / "config.json").write_text('{"test": true}')
    (bigvgan_dir / "bigvgan_generator.pt").write_bytes(b"fake")

    campplus = cache_dir / "campplus_cn_common.bin"
    campplus.write_bytes(b"fake_campplus")

    semantic = cache_dir / "semantic_codec_model.safetensors"
    semantic.write_bytes(b"fake_semantic")

    paths = ensure_models_available(str(model_dir))

    # Files preserved
    assert (w2v_dir / "config.json").exists()
    assert campplus.exists()
    assert semantic.exists()
    assert (bigvgan_dir / "bigvgan_generator.pt").exists()

    # No download triggered
    assert len(download_calls) == 0, f"Unexpected downloads: {download_calls}"


def test_modelscope_single_file_download_matches_local_path(tmp_path, monkeypatch):
    """ModelScope single-file download must produce the exact requested local_path."""
    from indextts.utils import model_download

    local_path = tmp_path / "hf_cache" / "semantic_codec_model.safetensors"
    expected_bytes = b"fake_semantic"

    def fake_model_file_download(model_id, file_path, local_dir):
        downloaded = Path(local_dir) / file_path
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(expected_bytes)
        return str(downloaded)

    fake_modelscope = types.ModuleType("modelscope")
    fake_hub = types.ModuleType("modelscope.hub")
    fake_file_download = types.ModuleType("modelscope.hub.file_download")
    fake_file_download.model_file_download = fake_model_file_download
    fake_hub.file_download = fake_file_download
    fake_modelscope.hub = fake_hub

    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", fake_hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", fake_file_download)
    monkeypatch.setattr(model_download, "_get_using_modelscope", lambda: True)

    got = model_download._download_single_file(
        repo_id="amphion/MaskGCT",
        filename="semantic_codec/model.safetensors",
        local_path=str(local_path),
    )

    assert got == str(local_path)
    assert local_path.exists()
    assert local_path.read_bytes() == expected_bytes


# -- Inference (GPU required) --------------------------------------------------

INFER_TEXTS = [
    "大家好，这是一段测试语音。",
    "There is a vehicle arriving in dock number 7?",
    "Joseph Gordon-Levitt is an American actor.",
]


@pytest.mark.gpu
@pytest.mark.parametrize("text", INFER_TEXTS, ids=lambda t: t[:20])
def test_infer(tts_model, prompt_wav, text, tmp_path):
    out = tmp_path / "out.wav"
    tts_model.infer(spk_audio_prompt=prompt_wav, text=text, output_path=str(out))
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_infer_with_emotion_vector(tts_model, prompt_wav, tmp_path):
    """infer() with explicit emotion vector."""
    emo_vec = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    out = tmp_path / "emo.wav"
    tts_model.infer(
        spk_audio_prompt=prompt_wav,
        text="今天天气真好，心情特别愉快！",
        output_path=str(out),
        emo_vector=emo_vec,
    )
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_infer_with_emo_text(tts_model, prompt_wav, tmp_path):
    """infer() with use_emo_text auto-detection."""
    out = tmp_path / "emo_text.wav"
    tts_model.infer(
        spk_audio_prompt=prompt_wav,
        text="这件事让我非常生气！",
        output_path=str(out),
        use_emo_text=True,
    )
    assert out.exists() and out.stat().st_size > 1000


# -- Long text (GPU required) -------------------------------------------------

@pytest.mark.gpu
def test_infer_long_text(tts_model, prompt_wav, tmp_path):
    text = (
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗诺兰执导并编剧，"
        "莱昂纳多迪卡普里奥、玛丽昂歌迪亚、约瑟夫高登莱维特、艾利奥特佩吉、"
        "汤姆哈迪等联袂主演，2010年7月16日在美国上映。"
        "影片剧情游走于梦境与现实之间，讲述了由莱昂纳多扮演的造梦师，"
        "带领特工团队进入他人梦境，从他人的潜意识中盗取机密的故事。"
    )
    out = tmp_path / "long.wav"
    tts_model.infer(spk_audio_prompt=prompt_wav, text=text, output_path=str(out))
    assert out.exists() and out.stat().st_size > 5000
