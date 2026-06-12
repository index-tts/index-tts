"""
IndexTTS v1 regression tests -- require GPU and model checkpoints.

Run with:
    uv run --extra test pytest tests/test_v1.py -v
"""
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

CHECKPOINTS_DIR = Path("checkpoints")
CONFIG_PATH = CHECKPOINTS_DIR / "config.yaml"


@pytest.fixture(scope="module")
def tts_model():
    pytest.importorskip("torch")
    if not CONFIG_PATH.exists():
        pytest.skip(f"Checkpoints not found at {CHECKPOINTS_DIR}")

    from indextts.infer import IndexTTS
    return IndexTTS(
        cfg_path=str(CONFIG_PATH),
        model_dir=str(CHECKPOINTS_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
    )


@pytest.fixture(scope="module")
def prompt_wav():
    from indextts.utils.examples_downloader import ensure_test_sample_available
    return ensure_test_sample_available()


# -- infer (single segment) ---------------------------------------------------

INFER_TEXTS = [
    "晕 XUAN4 是 一 种 GAN3 觉",
    "大家好，我现在正在bilibili体验ai科技，说实话，来之前我绝对想不到！",
    "There is a vehicle arriving in dock number 7?",
    "Joseph Gordon-Levitt is an American actor",
    "约瑟夫·高登-莱维特是美国演员",
    "蒂莫西·唐纳德·库克，通称蒂姆·库克，现任苹果公司首席执行官。",
]


@pytest.mark.parametrize("text", INFER_TEXTS, ids=lambda t: t[:20])
def test_infer(tts_model, prompt_wav, text, tmp_path):
    out = tmp_path / "out.wav"
    tts_model.infer(audio_prompt=prompt_wav, text=text, output_path=str(out), verbose=True)
    assert out.exists() and out.stat().st_size > 1000


# -- infer_fast (parallel segments) --------------------------------------------

INFER_FAST_TEXTS = [
    "亲爱的伙伴们，大家好！每一次的努力都是为了更好的未来，让我们一起勇敢前行,迈向更加美好的明天！",
    "The weather is really nice today, perfect for studying at home. Thank you!",
]


@pytest.mark.parametrize("text", INFER_FAST_TEXTS, ids=lambda t: t[:20])
def test_infer_fast(tts_model, prompt_wav, text, tmp_path):
    out = tmp_path / "out.wav"
    tts_model.infer_fast(audio_prompt=prompt_wav, text=text, output_path=str(out), verbose=True)
    assert out.exists() and out.stat().st_size > 1000


# -- infer_fast long text ------------------------------------------------------

LONG_TEXTS = [
    (
        "叶远随口答应一声，一定帮忙云云。"
        "教授看叶远的样子也知道，这事情多半是黄了。"
        "谁得到这样的东西也不会轻易贡献出来，这是很大的一笔财富。"
        "叶远回来后，又自己做了几次试验，发现空间湖水对一些外伤也有很大的帮助。"
        "找来一只断了腿的兔子，喝下空间湖水，一天时间，兔子就完全好了。"
        "感谢您的收听，下期再见！"
    ),
    (
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，"
        "莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、"
        "汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映。"
        "影片剧情游走于梦境与现实之间，讲述了由莱昂纳多扮演的造梦师，"
        "带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。"
    ),
]


@pytest.mark.parametrize("text", LONG_TEXTS, ids=["novel", "movie"])
def test_infer_fast_long(tts_model, prompt_wav, text, tmp_path):
    out = tmp_path / "out.wav"
    tts_model.infer_fast(audio_prompt=prompt_wav, text=text, output_path=str(out), verbose=True)
    assert out.exists() and out.stat().st_size > 5000
