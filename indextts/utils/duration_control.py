import math


def normalize_target_duration(target_duration):
    """Return a validated duration in seconds, or ``None`` for automatic mode."""
    if target_duration is None or target_duration == "":
        return None
    if isinstance(target_duration, bool):
        raise ValueError("target_duration must be a positive number of seconds")

    try:
        duration = float(target_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_duration must be a positive number of seconds") from exc

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("target_duration must be a positive number of seconds")
    return duration


def allocate_target_frames(
    target_duration,
    segment_weights,
    sampling_rate,
    hop_length,
    interval_silence_ms=0,
):
    """Allocate a requested total duration across synthesis segments.

    The requested duration includes the silence inserted between segments.  The
    returned frame counts always add up to the nearest achievable mel-frame
    duration, with at least one frame assigned to every segment.
    """
    duration = normalize_target_duration(target_duration)
    if duration is None:
        return None, None
    if sampling_rate <= 0 or hop_length <= 0:
        raise ValueError("sampling_rate and hop_length must be positive")

    weights = [float(weight) for weight in segment_weights]
    if not weights:
        raise ValueError("target_duration requires at least one text segment")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("segment weights must be finite non-negative numbers")
    if not any(weights):
        weights = [1.0] * len(weights)

    target_samples = round(duration * sampling_rate)
    silence_per_gap = max(0, int(sampling_rate * interval_silence_ms / 1000.0))
    silence_samples = silence_per_gap * (len(weights) - 1)
    speech_samples = target_samples - silence_samples
    total_frames = round(speech_samples / hop_length)
    if total_frames < len(weights):
        minimum_seconds = (
            silence_samples + len(weights) * hop_length
        ) / sampling_rate
        raise ValueError(
            "target_duration is too short for the generated segments and pauses; "
            f"use at least {minimum_seconds:.3f} seconds"
        )

    # Reserve one frame per segment, then distribute the remaining frames using
    # the largest-remainder method so rounding never changes the requested sum.
    remaining = total_frames - len(weights)
    weight_sum = sum(weights)
    quotas = [remaining * weight / weight_sum for weight in weights]
    extra_frames = [math.floor(quota) for quota in quotas]
    unassigned = remaining - sum(extra_frames)
    remainder_order = sorted(
        range(len(weights)),
        key=lambda index: (quotas[index] - extra_frames[index], weights[index]),
        reverse=True,
    )
    for index in remainder_order[:unassigned]:
        extra_frames[index] += 1

    return [frames + 1 for frames in extra_frames], target_samples


def fit_waveform_length(wav, target_samples):
    """Trim or right-pad a ``[..., samples]`` tensor to an exact sample count."""
    if target_samples is None:
        return wav
    current_samples = wav.shape[-1]
    if current_samples > target_samples:
        return wav[..., :target_samples]
    if current_samples < target_samples:
        padded = wav.new_zeros((*wav.shape[:-1], target_samples))
        padded[..., :current_samples] = wav
        return padded
    return wav
