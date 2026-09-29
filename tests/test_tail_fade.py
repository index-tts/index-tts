"""末尾截断咔哒声的回归测试 (index-tts/index-tts#247, #488, #523, #633)。

纯 CPU、不需要 checkpoint，所以能跑在现有的 `not gpu` CI job 里。

背景：生成波形的长度来自 token 数而不是声学——
``target_lengths = (code_lens * 1.72)``，其中 ``code_lens`` 是采样到的
``stop_mel_token`` 的下标。解码是随机采样，因此少数情况下停止符落得偏早，
mel 在话音尚未结束时截止，文件最后一个采样点离零很远，播放时就是一声咔哒。

运行方式：
    uv run --extra test pytest tests/test_tail_fade.py -v
"""
import pytest

torch = pytest.importorskip("torch")

from indextts.utils.common import PCM16_MAX, TAIL_FADE_MS, fade_out_tail  # noqa: E402

SAMPLE_RATE = 22050


def _tone(seconds=1.0, freq=220.0, amplitude=0.9):
    t = torch.arange(int(SAMPLE_RATE * seconds), dtype=torch.float32) / SAMPLE_RATE
    return (amplitude * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0)


def _truncated_utterance():
    """一段在波峰附近被切断的正弦：停止符偏早时产生的就是这种波形。"""
    wav = _tone()
    cut = int(SAMPLE_RATE * 0.5) + int(SAMPLE_RATE / 220.0 / 4)
    return torch.clamp(PCM16_MAX * wav[:, :cut], -PCM16_MAX, PCM16_MAX)


def _healthy_utterance(trailing_silence_ms=250.0):
    """正弦后面跟一段静音：另外约 93% 的生成是这样收尾的。"""
    wav = torch.clamp(PCM16_MAX * _tone(), -PCM16_MAX, PCM16_MAX)
    pad = torch.zeros(1, int(SAMPLE_RATE * trailing_silence_ms / 1000.0))
    return torch.cat([wav, pad], dim=1)


def test_fade_lands_the_last_sample_on_zero():
    wav = _truncated_utterance()
    assert wav[0, -1].abs() > 0.5 * PCM16_MAX, "构造的样本本身应该以一个大阶跃结束"

    faded = fade_out_tail(wav, SAMPLE_RATE)

    assert faded[0, -1].abs() < 1.0


def test_fade_is_bit_exact_on_a_silent_tail():
    """尾部本来就是数字静音，缩放零仍是零。

    这正是这个淡出可以无条件施加的原因：对一条正常收尾的生成，
    它连一个采样点都改不了。
    """
    wav = _healthy_utterance()

    faded = fade_out_tail(wav, SAMPLE_RATE)

    assert torch.equal(faded, wav)


def test_fade_only_touches_the_ramp_region():
    wav = _truncated_utterance()
    n = int(SAMPLE_RATE * TAIL_FADE_MS / 1000.0)

    faded = fade_out_tail(wav, SAMPLE_RATE)

    assert torch.equal(faded[:, :-n], wav[:, :-n])
    assert faded.shape == wav.shape


def test_fade_ramp_is_monotonically_decreasing():
    """非单调的窗会引入它自己的瞬态。"""
    n = 64
    wav = torch.ones(1, n)

    faded = fade_out_tail(wav, SAMPLE_RATE, fade_ms=1000.0 * n / SAMPLE_RATE)

    diffs = faded[0, 1:] - faded[0, :-1]
    assert (diffs <= 1e-6).all()
    assert faded[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_fade_does_not_mutate_the_caller_tensor():
    wav = _truncated_utterance()
    before = wav.clone()

    fade_out_tail(wav, SAMPLE_RATE)

    assert torch.equal(wav, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int16, torch.int32])
def test_fade_preserves_dtype(dtype):
    """v2.5 的低显存路径拼接的是 int16，其余路径是浮点。"""
    wav = _truncated_utterance().to(dtype)

    faded = fade_out_tail(wav, SAMPLE_RATE)

    assert faded.dtype == dtype
    assert faded[0, -1].abs() <= 1


@pytest.mark.parametrize("samples", [0, 1, 2])
def test_fade_tolerates_degenerate_input(samples):
    wav = torch.ones(1, samples)

    faded = fade_out_tail(wav, SAMPLE_RATE)

    assert faded.shape == wav.shape


def test_fade_is_a_no_op_when_disabled():
    wav = _truncated_utterance()

    assert torch.equal(fade_out_tail(wav, SAMPLE_RATE, fade_ms=0.0), wav)
