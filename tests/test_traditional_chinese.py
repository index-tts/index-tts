"""Traditional Chinese must reach the tokenizer as Simplified on every platform.

WeTextProcessing (Linux) converts Traditional to Simplified and full-width to
half-width by default. wetext (Windows, macOS) needs both switched on, otherwise
the text reaches the BPE model unchanged and many characters become <unk>.

CPU only, no model weights. Uses the real normalizer and skips if it is missing:
    uv run --extra test pytest tests/test_traditional_chinese.py -v
"""
import importlib.util
import platform
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

from indextts.utils.front import TextNormalizer, TextTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BPE_MODEL = ROOT / "checkpoints" / "bpe.model"


def trt_text_normalizer():
    # Load the file directly: importing the backends.trt.pipeline package needs TensorRT.
    path = ROOT / "backends" / "trt" / "pipeline" / "text_processing.py"
    spec = importlib.util.spec_from_file_location("trt_text_processing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextNormalizer


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


@pytest.mark.parametrize(("text", "expected"), [
    ("飛行員", "飞行员"),
    ("客機", "客机"),
    ("這架客機的飛行員說話", "这架客机的飞行员说话"),
    # Digits send the text through the tagger, which wetext skips otherwise.
    ("飛行員有3架客機", "飞行员有三架客机"),
    ("ＡＢＣ", "ABC"),
    ("飞行员", "飞行员"),
])
def test_traditional_and_full_width_are_normalized(normalizer, text, expected):
    assert normalizer.normalize(text) == expected


@pytest.mark.skipif(not BPE_MODEL.exists(), reason="checkpoints/bpe.model is not downloaded")
def test_traditional_text_has_no_unknown_tokens(normalizer):
    tokenizer = TextTokenizer(str(BPE_MODEL), normalizer)
    assert tokenizer.unk_token_id not in tokenizer.encode("這架客機的飛行員說話", out_type=int)


@pytest.mark.parametrize("normalizer_class", [TextNormalizer, trt_text_normalizer()], ids=["front", "trt"])
def test_wetext_gets_the_same_options_as_wetextprocessing(monkeypatch, normalizer_class):
    # Uses a fake module so this also runs on Linux, where wetext is not installed.
    calls = []
    fake_wetext = types.ModuleType("wetext")
    fake_wetext.Normalizer = lambda **kwargs: calls.append(kwargs) or object()
    monkeypatch.setitem(sys.modules, "wetext", fake_wetext)
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    normalizer_class().load()

    zh_kwargs = next(kwargs for kwargs in calls if kwargs.get("lang") == "zh")
    assert zh_kwargs["traditional_to_simple"] is True
    assert zh_kwargs["full_to_half"] is True
