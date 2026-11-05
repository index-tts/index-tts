"""Pinyin detection must not fire on names such as v1, A380 or U2 (#531).

A false match is left out of number normalization, so A380 became A3八十 and an
English sentence containing U2 was read with Chinese numbers.

CPU only, no model weights. The normalizer tests use the real backend and skip if
it is missing:
    uv run --extra test pytest tests/test_pinyin_detection.py -v
"""
import importlib.util
import platform
import re
from pathlib import Path

import pytest

pytest.importorskip("torch")

from indextts.utils.front import TextNormalizer, TextTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
BPE_MODEL = CHECKPOINTS / "bpe.model"
FINALS = (
    "a e i o u ü v ai ei ao ou an en in un vn ün ia ie iu ua ue ui uo ve üe "
    "ang eng ing ong ian iao uai uan van üan iang iong uang vang üang ng er"
).split()


def find_pinyin(text):
    pattern = TextNormalizer.PINYIN_TONE_PATTERN
    return ["".join(match) for match in re.findall(pattern, text, re.IGNORECASE)]


def pinyin_vocab():
    return (CHECKPOINTS / "pinyin.vocab").read_text(encoding="utf-8").split()


@pytest.fixture(scope="module")
def normalizer():
    instance = TextNormalizer()
    try:
        if platform.system() == "Linux":
            # load() builds its own zh graph, which takes several GB of memory with
            # the locked WeTextProcessing. The graph shipped in its wheel is enough here.
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn

            instance.zh_normalizer = NormalizerZh()
            instance.en_normalizer = NormalizerEn()
        else:
            instance.load()
    except ImportError as exc:
        pytest.skip(f"text normalizer backend is not installed: {exc}")
    return instance


@pytest.mark.parametrize("text", [
    "v1", "V2", "v3导通", "U2", "u2 net", "i5处理器",
    "A380", "空客A380客机", "A4纸", "Tu154", "Su27",
    "E2E测试", "O2O", "A1B2", "Fe2O3",
])
def test_names_are_not_pinyin(text):
    assert find_pinyin(text) == []


@pytest.mark.parametrize(("text", "expected"), [
    ("xuan4", ["xuan4"]),
    ("XUAN4", ["XUAN4"]),
    ("lv3", ["lv3"]),
    ("jve2", ["jve2"]),
    ("a5", ["a5"]),
    ("er2", ["er2"]),
    ("tu1", ["tu1"]),
    ("晕XUAN4是一种GAN3觉", ["XUAN4", "GAN3"]),
    ("最zhong4要的是：不要chong2蹈覆辙", ["zhong4", "chong2"]),
    ("xing2 bu5 xing2 a5？", ["xing2", "bu5", "xing2", "a5"]),
    # Syllables written without a space must still work.
    ("ni3hao3世界", ["ni3", "hao3"]),
    ("zhong1guo2", ["zhong1", "guo2"]),
])
def test_real_pinyin_is_detected(text, expected):
    assert find_pinyin(text) == expected


def test_every_pinyin_vocab_entry_is_detected():
    assert [entry for entry in pinyin_vocab() if find_pinyin(entry) != [entry]] == []


def test_syllables_without_initial_are_limited_to_the_vocab():
    vocab = {entry.upper() for entry in pinyin_vocab()}
    detected = {
        form.upper()
        for form in (f"{final}{tone}" for final in FINALS for tone in "12345")
        if find_pinyin(form) == [form]
    }
    assert sorted(detected - vocab) == []


def test_trt_backend_uses_the_same_pattern():
    # Load the file directly: importing the backends.trt.pipeline package needs TensorRT.
    path = ROOT / "backends" / "trt" / "pipeline" / "text_processing.py"
    spec = importlib.util.spec_from_file_location("trt_text_processing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TextNormalizer.PINYIN_TONE_PATTERN == TextNormalizer.PINYIN_TONE_PATTERN


@pytest.mark.parametrize(("text", "expected"), [
    ("U2 released 15 albums in 2023.", False),
    ("The A380 flew 12 routes.", False),
    ("Version v1 has 15 bugs.", False),
    ("ni3 hao3 world", True),
])
def test_use_chinese_ignores_names(text, expected):
    assert TextNormalizer().use_chinese(text) is expected


@pytest.mark.parametrize(("text", "expected"), [
    ("v1、v2重新截止，v3导通", "v一,v二重新截止,v三导通"),
    ("空客A380客机", "空客A三百八十客机"),
    ("A4纸", "A四纸"),
    ("晕XUAN4是一种GAN3觉", "晕XVAN4是一种GAN3觉"),
    ("lv3色", "lv3色"),
    ("ni3hao3世界", "ni3hao3世界"),
])
def test_chinese_text_normalization(normalizer, text, expected):
    assert normalizer.normalize(text) == expected


def test_english_sentence_keeps_english_numbers(normalizer):
    result = normalizer.normalize("U2 released 15 albums in 2023.")
    assert "fifteen" in result
    assert re.search(r"[一-鿿]", result) is None


@pytest.mark.skipif(not BPE_MODEL.exists(), reason="checkpoints/bpe.model is not downloaded")
def test_real_tokenizer(normalizer):
    tokenizer = TextTokenizer(str(BPE_MODEL), normalizer)
    for text in ("v1、v2重新截止", "空客A380客机", "A4纸", "U2 released 15 albums in 2023."):
        assert tokenizer.unk_token_id not in tokenizer.encode(text, out_type=int), text
    assert "XVAN4" in tokenizer.tokenize("晕xuan4")
    assert "LV3" in tokenizer.tokenize("lv3色")
