from __future__ import annotations

import os
from typing import Sequence

from indextts.reference_conditioning import WeightedReference


def normalize_weight_list(weights, label):
    normalized = [float(weight) for weight in weights]
    if any(weight < 0 for weight in normalized):
        raise ValueError(f"{label} weights must be non-negative")
    weight_sum = sum(normalized)
    if weight_sum <= 0:
        raise ValueError(f"{label} weights must sum to a value greater than 0")
    return [weight / weight_sum for weight in normalized]


def normalize_emotion_input_rows(emotion_references, default_text):
    if emotion_references is None:
        return None

    if isinstance(emotion_references, dict):
        raw_rows = [emotion_references]
    elif isinstance(emotion_references, (list, tuple)):
        raw_rows = list(emotion_references)
    else:
        raise TypeError("emotion_references must be a dict or a sequence of dicts")

    normalized_rows = []
    raw_weights = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise TypeError(f"emotion_references[{index}] must be a dict")

        row_type = str(raw_row.get("type", "audio")).strip().lower()
        weight = float(raw_row.get("weight", 1.0))
        if row_type == "speaker":
            normalized_row = {"type": "speaker"}
            speaker_index = raw_row.get("speaker_index")
            if speaker_index not in (None, ""):
                try:
                    normalized_row["speaker_index"] = int(speaker_index)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"emotion_references[{index}] speaker rows require an integer speaker_index"
                    ) from exc
            normalized_rows.append(normalized_row)
        elif row_type == "audio":
            path = raw_row.get("path") or raw_row.get("audio_path")
            if path is None:
                raise ValueError(f"emotion_references[{index}] audio rows require a path")
            normalized_rows.append({"type": "audio", "path": os.path.abspath(os.fspath(path))})
        elif row_type == "vector":
            vector = raw_row.get("vector")
            if not isinstance(vector, (list, tuple)) or len(vector) != 8:
                raise ValueError(f"emotion_references[{index}] vector rows require an 8-d vector")
            normalized_rows.append({"type": "vector", "vector": [float(value) for value in vector]})
        elif row_type == "text":
            text_value = raw_row.get("text")
            if text_value is None:
                text_value = default_text
            text_value = str(text_value).strip()
            if not text_value:
                raise ValueError(f"emotion_references[{index}] text rows require non-empty text")
            normalized_rows.append({"type": "text", "text": text_value})
        else:
            raise ValueError(f"Unsupported emotion reference type: {row_type}")

        raw_weights.append(weight)

    normalized_weights = normalize_weight_list(raw_weights, "emotion reference")
    for row, weight in zip(normalized_rows, normalized_weights):
        row["weight"] = weight
    return normalized_rows


def resolve_speaker_reference_index(row, speaker_reference_count):
    if speaker_reference_count <= 0:
        raise ValueError("Speaker-linked emotion rows require at least one active speaker reference")

    speaker_index = row.get("speaker_index")
    if speaker_index is None:
        if speaker_reference_count == 1:
            return 0
        raise ValueError(
            "Speaker-linked emotion rows require speaker_index when multiple speaker references are active"
        )

    try:
        speaker_index = int(speaker_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("Speaker-linked emotion rows require an integer speaker_index") from exc

    if speaker_index < 0 or speaker_index >= speaker_reference_count:
        raise ValueError(
            f"speaker_index {speaker_index} is out of range for {speaker_reference_count} active speaker references"
        )
    return speaker_index


def expand_audio_like_emotion_rows(
    emotion_rows,
    speaker_references: Sequence[WeightedReference],
):
    expanded_references = []

    for row in emotion_rows:
        if row["type"] == "speaker":
            speaker_index = resolve_speaker_reference_index(row, len(speaker_references))
            expanded_references.append(
                WeightedReference(
                    path=speaker_references[speaker_index].path,
                    weight=row["weight"],
                )
            )
        elif row["type"] == "audio":
            expanded_references.append(WeightedReference(path=row["path"], weight=row["weight"]))

    return expanded_references or None
