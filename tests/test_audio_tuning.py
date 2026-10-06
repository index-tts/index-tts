import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from indextts.utils.audio_tuning import (
    DEFAULT_PRESET,
    PRESET_FIELDS,
    _cli,
    preset_values,
    resolve_preset,
    tune_audio,
)


def test_preset_aliases_resolve_to_stable_ids():
    assert resolve_preset("score92") == "voice_clarity"
    assert resolve_preset("voice-clarity") == DEFAULT_PRESET
    assert resolve_preset("none") == "bypass"
    assert len(preset_values("voice_clarity")) == len(PRESET_FIELDS)


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown tuning preset"):
        resolve_preset("not-a-preset")


def test_cli_lists_presets(capsys):
    assert _cli(["--list-presets"]) == 0
    payload = capsys.readouterr().out
    assert '"voice_clarity"' in payload
    assert '"ok": true' in payload


def test_tune_audio_writes_a_new_file(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for audio tuning")

    source = tmp_path / "src.wav"
    dest = tmp_path / "out.wav"
    tone = (0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.4, 8820, endpoint=False))).astype(np.float32)
    sf.write(source, tone, 22050)

    output, summary = tune_audio(
        str(source),
        *preset_values("voice_clarity"),
        output_path=str(dest),
    )
    assert Path(output) == dest
    assert dest.is_file()
    assert source.stat().st_size != dest.stat().st_size or "处理完成" in summary
    info = sf.info(str(dest))
    assert info.samplerate == 22050
    assert info.duration > 0.2


def test_bypass_copies_without_eq(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for audio tuning")

    source = tmp_path / "src.wav"
    dest = tmp_path / "out.wav"
    tone = (0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.2, 4410, endpoint=False))).astype(np.float32)
    sf.write(source, tone, 22050)

    output, summary = tune_audio(
        str(source),
        *preset_values("bypass"),
        output_path=str(dest),
    )
    assert Path(output) == dest
    assert "未应用调音" in summary
    assert dest.is_file()
