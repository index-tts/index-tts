import pytest

from indextts.utils.duration_control import (
    allocate_target_frames,
    fit_waveform_length,
    normalize_target_duration,
)


class FakeWaveform:
    """Small tensor-shaped test double for the framework-independent helper."""

    def __init__(self, samples):
        self.samples = list(samples)

    @property
    def shape(self):
        return (1, len(self.samples))

    def __getitem__(self, key):
        assert key[0] is Ellipsis
        return FakeWaveform(self.samples[key[1]])

    def __setitem__(self, key, value):
        assert key[0] is Ellipsis
        self.samples[key[1]] = value.samples

    def new_zeros(self, shape):
        assert shape[0] == 1
        return FakeWaveform([0] * shape[1])


def test_normalize_target_duration_supports_automatic_and_seconds():
    assert normalize_target_duration(None) is None
    assert normalize_target_duration(5) == 5.0
    assert normalize_target_duration("2.5") == 2.5


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "bad"])
def test_normalize_target_duration_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive number"):
        normalize_target_duration(value)


def test_allocate_target_frames_accounts_for_segment_silence():
    frames, target_samples = allocate_target_frames(
        7.3,
        [1, 2, 3],
        sampling_rate=22050,
        hop_length=256,
        interval_silence_ms=200,
    )

    silence_samples = int(22050 * 0.2) * 2
    expected_frames = round((target_samples - silence_samples) / 256)
    assert target_samples == round(7.3 * 22050)
    assert sum(frames) == expected_frames
    assert frames[0] < frames[1] < frames[2]
    assert all(frame_count >= 1 for frame_count in frames)


def test_allocate_target_frames_rejects_duration_shorter_than_pauses():
    with pytest.raises(ValueError, match="too short"):
        allocate_target_frames(
            0.1,
            [1, 1, 1],
            sampling_rate=22050,
            hop_length=256,
            interval_silence_ms=200,
        )


def test_fit_waveform_length_trims_and_pads():
    wav = FakeWaveform([0, 1, 2, 3, 4])

    assert fit_waveform_length(wav, 3).samples == [0, 1, 2]
    assert fit_waveform_length(wav, 7).samples == [0, 1, 2, 3, 4, 0, 0]
    assert fit_waveform_length(wav, None) is wav
