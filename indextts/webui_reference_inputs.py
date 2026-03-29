from __future__ import annotations

import os


def normalize_uploaded_path(value):
    if value is None:
        return None
    if isinstance(value, str):
        path = value.strip()
        return path or None
    if hasattr(value, "name"):
        path = str(value.name).strip()
        return path or None
    return None


def collect_speaker_references(values):
    speaker_refs = []
    weights = []
    for row in values:
        if not row["active"]:
            continue
        audio_path = normalize_uploaded_path(row["audio"])
        if audio_path is None:
            continue
        speaker_refs.append(audio_path)
        weights.append(clamp_weight(row["weight"]))

    if not speaker_refs:
        raise ValueError("At least one speaker reference audio is required.")
    return speaker_refs, weights


def list_speaker_reference_targets(values):
    targets = []
    for row in values:
        if not row["active"]:
            continue
        audio_path = normalize_uploaded_path(row["audio"])
        if audio_path is None:
            continue
        targets.append((len(targets), audio_path))
    return targets


def collect_emotion_references(values, synthesis_text, normalize_emo_vector, emotion_type_speaker, emotion_type_audio, emotion_type_vector, emotion_type_text):
    emotion_refs = []
    for row in values:
        if not row["active"]:
            continue

        row_type = row["type"] or emotion_type_speaker
        weight = clamp_weight(row["weight"])
        if row_type == emotion_type_speaker:
            emotion_row = {"type": emotion_type_speaker, "weight": weight}
            speaker_index = row.get("speaker_index")
            if speaker_index not in (None, ""):
                try:
                    emotion_row["speaker_index"] = int(speaker_index)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Speaker-linked emotion rows require a valid speaker selection.") from exc
            emotion_refs.append(emotion_row)
            continue

        if row_type == emotion_type_audio:
            audio_path = normalize_uploaded_path(row["audio"])
            if audio_path is None:
                continue
            emotion_refs.append({"type": emotion_type_audio, "path": audio_path, "weight": weight})
            continue

        if row_type == emotion_type_vector:
            emotion_refs.append(
                {
                    "type": emotion_type_vector,
                    "vector": normalize_emo_vector(row["vector"], apply_bias=True),
                    "weight": weight,
                }
            )
            continue

        text_value = (row["text"] or "").strip()
        if not text_value:
            text_value = synthesis_text
        if not text_value:
            raise ValueError("Emotion text row is empty and synthesis text is also empty.")
        emotion_refs.append({"type": emotion_type_text, "text": text_value, "weight": weight})

    if not emotion_refs:
        return [{"type": emotion_type_speaker, "weight": 1.0}]
    return emotion_refs


def clamp_weight(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
