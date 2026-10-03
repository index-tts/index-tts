"""CPU regressions for IndexTTS's contraction preprocessing (issue #542)."""

from types import SimpleNamespace

import pytest

from indextts.utils.front import TextNormalizer


@pytest.fixture
def normalizer():
    # Exercise normalize() while isolating the platform-specific FST backends.
    instance = TextNormalizer()
    instance.zh_normalizer = SimpleNamespace(normalize=lambda text: text)
    instance.en_normalizer = SimpleNamespace(normalize=lambda text: text)
    return instance


@pytest.mark.parametrize("text", [
    "The rabbit's fur is white.",
    "The exhibit's title is new.",
    "The niche's design matters.",
    "Blythe's design is new.",
    "The RABBIT'S fur is white.",
    "The designer's choice is new.",
])
def test_possessives_are_not_expanded_as_contractions(normalizer, text):
    assert normalizer.normalize(text) == text


@pytest.mark.parametrize(("text", "expected"), [
    ("It's sunny.", "It is sunny."),
    ("She's ready.", "She is ready."),
    ("What's the time?", "What is the time?"),
    ("There's a rabbit.", "There is a rabbit."),
])
def test_standalone_contractions_still_expand(normalizer, text, expected):
    assert normalizer.normalize(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("这是rabbit's皮毛。", "这是rabbit's皮毛."),
    ("这是niche's设计。", "这是niche's设计."),
    ("今天it's晴天。", "今天it is晴天."),
    ("她说she's ready。", "她说she is ready."),
    ("it's_unchanged", "it's_unchanged"),
])
def test_contraction_boundaries_in_mixed_text(normalizer, text, expected):
    assert normalizer.normalize(text) == expected
