import pytest

from indextts.utils.duration_control import (
    allocate_target_frames,
    fit_final_segment_length,
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


def test_capacity_is_per_segment_and_accounts_for_pauses():
    # Total speech can exceed one DiT budget if each segment fits individually.
    assert allocate_target_frames(
        22, [1, 1], 10, 10, 2000, max_segment_frames=10
    ) == ([10, 10], 220)
    with pytest.raises(ValueError, match=r"segment 1 needs 11.*only 10"):
        allocate_target_frames(23, [1, 1], 10, 10, 2000, max_segment_frames=10)


def test_reference_can_exhaust_the_entire_position_budget():
    with pytest.raises(ValueError, match="DiT position capacity"):
        allocate_target_frames(1, [1], 10, 10, max_segment_frames=0)


def test_final_segment_fitting_accounts_for_previous_speech_and_gaps():
    wav = FakeWaveform([1] * 5)
    previous = [FakeWaveform([2] * 3), FakeWaveform([3] * 4)]
    assert fit_final_segment_length(wav, previous, 16, 10, 200).samples == [1] * 5
    assert fit_final_segment_length(wav, previous, 14, 10, 200).samples == [1] * 3
    assert fit_final_segment_length(wav, previous, 18, 10, 200).samples == [1] * 5 + [0] * 2
    assert fit_final_segment_length(wav, previous, None, 10, 200) is wav
    with pytest.raises(ValueError, match="too short"):
        fit_final_segment_length(wav, previous, 11, 10, 200)
