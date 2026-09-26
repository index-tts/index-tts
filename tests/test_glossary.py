import pytest

from indextts.utils.front import TextNormalizer


def test_delete_glossary_term_persists_other_terms(tmp_path):
    glossary_path = tmp_path / "glossary.yaml"
    normalizer = TextNormalizer()
    normalizer.term_glossary = {
        "M.2": {"en": "M dot two", "zh": "M 二"},
        "C++": {"en": "C plus plus"},
    }
    normalizer.save_glossary_to_yaml(glossary_path)

    assert normalizer.delete_glossary_term("M.2", glossary_path) is True

    reloaded = TextNormalizer()
    assert reloaded.load_glossary_from_yaml(glossary_path) is True
    assert reloaded.term_glossary == {"C++": {"en": "C plus plus"}}


def test_delete_last_glossary_term_reloads_empty_glossary(tmp_path):
    glossary_path = tmp_path / "glossary.yaml"
    normalizer = TextNormalizer()
    normalizer.term_glossary = {"M.2": {"en": "M dot two"}}
    normalizer.save_glossary_to_yaml(glossary_path)

    assert normalizer.delete_glossary_term("M.2", glossary_path) is True
    normalizer.term_glossary["stale"] = "stale"
    assert normalizer.load_glossary_from_yaml(glossary_path) is True
    assert normalizer.term_glossary == {}


def test_delete_missing_glossary_term_does_not_save(tmp_path, monkeypatch):
    normalizer = TextNormalizer()
    normalizer.term_glossary = {"M.2": "M dot two"}

    def unexpected_save(_path):
        pytest.fail("Missing terms must not trigger a save")

    monkeypatch.setattr(normalizer, "save_glossary_to_yaml", unexpected_save)

    assert normalizer.delete_glossary_term("unknown", tmp_path / "glossary.yaml") is False
    assert normalizer.term_glossary == {"M.2": "M dot two"}


def test_delete_glossary_term_restores_memory_on_save_failure(tmp_path, monkeypatch):
    glossary_path = tmp_path / "glossary.yaml"
    normalizer = TextNormalizer()
    normalizer.term_glossary = {"M.2": "M dot two", "C++": "C plus plus"}
    normalizer.save_glossary_to_yaml(glossary_path)
    saved_yaml = glossary_path.read_text(encoding="utf-8")

    def failing_save(_path):
        raise OSError("disk unavailable")

    monkeypatch.setattr(normalizer, "save_glossary_to_yaml", failing_save)

    with pytest.raises(OSError, match="disk unavailable"):
        normalizer.delete_glossary_term("M.2", glossary_path)

    assert normalizer.term_glossary == {"M.2": "M dot two", "C++": "C plus plus"}
    assert glossary_path.read_text(encoding="utf-8") == saved_yaml
