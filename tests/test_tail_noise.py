"""末尾高频气声裁剪的回归测试 (index-tts/index-tts#488, #523)。

纯 CPU、不需要 checkpoint，跑在现有的 `not gpu` job 里。

背景：模型偶尔会在停止符之前多吐 35~170 ms 的高频气声。这段噪声在送进
BigVGAN 的声学特征里就已经存在（不是 vocoder 的边界伪影），所以可以在
声学特征上按频谱形态识别并裁掉。

判据要求两个条件同时成立：
  1. 末尾连续若干帧「高频占优」
  2. 这段之前存在能量低谷

第 2 条是安全阀：正常的擦音收尾（「四」「次」「西」）能量是连着元音下来的，
不会先掉下去再冒起来。下面 test_...sibilant... 就是守这条的。

运行方式：
    uv run --extra test pytest tests/test_tail_noise.py -v
"""
import pytest

torch = pytest.importorskip("torch")

from indextts.utils.common import (  # noqa: E402
    TAIL_NOISE_MAX_FRAMES,
    trim_tail_noise,
)

N_MELS = 80


def _speech(frames=200, level=-2.0):
    """低频占优的一段「语音」声学特征。"""
    mel = torch.full((1, N_MELS, frames), level)
    mel[:, : int(N_MELS * 0.4)] += 1.5      # 低频更强
    return mel


def _with_trailing_burst(gap_frames=5, burst_frames=12):
    """语音 → 一段静音低谷 → 一段高频噪声：模型多吐气声时的形态。"""
    mel = _speech()
    mel[:, :, -(gap_frames + burst_frames) : -burst_frames] -= 4.0     # 低谷
    mel[:, :, -burst_frames:] += 1.5                                   # 能量回升
    mel[:, int(N_MELS * 0.6) :, -burst_frames:] += 5.0                 # 高频占优
    return mel


def _sibilant_ending(frames_of_fricative=12):
    """以擦音收尾的正常语音：高频占优，但总能量是从元音一路衰下来的。

    这是和「多吐的气声」在物理上的区别，也是判据赖以区分两者的依据：
    /s/ /tsʰ/ 这类音紧接元音发出，能量单调下降；而模型多吐的那一段
    出现在一小段静音之后，能量是先掉下去再冒起来。
    """
    mel = _speech()
    tail = mel[:, :, -frames_of_fricative:]
    tail -= 3.0                                          # 总能量比元音低
    tail[:, int(N_MELS * 0.6) :] += 5.0                  # 但高频相对占优
    tail[:, : int(N_MELS * 0.4)] -= 1.5
    return mel


def test_trims_a_trailing_high_frequency_burst():
    mel = _with_trailing_burst(burst_frames=12)

    out = trim_tail_noise(mel)

    assert out.shape[-1] < mel.shape[-1]
    assert mel.shape[-1] - out.shape[-1] >= 12


def test_leaves_ordinary_speech_untouched():
    mel = _speech()

    assert trim_tail_noise(mel).shape == mel.shape


def test_leaves_a_sibilant_ending_untouched():
    """最重要的一条：擦音收尾高频占优，但能量连续，不该被当成噪声裁掉。"""
    mel = _sibilant_ending()

    assert trim_tail_noise(mel).shape == mel.shape


def test_ignores_a_burst_shorter_than_the_minimum():
    mel = _with_trailing_burst(burst_frames=2)

    assert trim_tail_noise(mel).shape == mel.shape


def test_never_cuts_more_than_the_safety_cap():
    mel = _with_trailing_burst(gap_frames=3, burst_frames=35)

    out = trim_tail_noise(mel)

    assert mel.shape[-1] - out.shape[-1] <= TAIL_NOISE_MAX_FRAMES


def test_does_not_mutate_the_caller_tensor():
    mel = _with_trailing_burst()
    before = mel.clone()

    trim_tail_noise(mel)

    assert torch.equal(mel, before)


@pytest.mark.parametrize("frames", [0, 1, 10, 39])
def test_tolerates_input_shorter_than_the_look_window(frames):
    mel = torch.zeros(1, N_MELS, frames)

    assert trim_tail_noise(mel).shape == mel.shape


def test_preserves_dtype():
    mel = _with_trailing_burst().to(torch.float64)

    assert trim_tail_noise(mel).dtype == torch.float64
