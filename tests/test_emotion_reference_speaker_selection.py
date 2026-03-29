import unittest

from indextts.emotion_reference_selection import (
    expand_audio_like_emotion_rows,
    normalize_emotion_input_rows,
    resolve_speaker_reference_index,
)
from indextts.reference_conditioning import WeightedReference
from indextts.webui_reference_inputs import (
    collect_emotion_references,
    list_speaker_reference_targets,
)


class EmotionReferenceSpeakerSelectionTests(unittest.TestCase):
    def test_normalize_emotion_input_rows_keeps_speaker_index(self):
        rows = normalize_emotion_input_rows(
            [
                {"type": "speaker", "speaker_index": "1", "weight": 3},
                {"type": "audio", "path": "emotion.wav", "weight": 1},
            ],
            "hello",
        )

        self.assertEqual(rows[0]["speaker_index"], 1)
        self.assertEqual([row["weight"] for row in rows], [0.75, 0.25])

    def test_resolve_speaker_reference_index_requires_multi_speaker_selection(self):
        with self.assertRaises(ValueError):
            resolve_speaker_reference_index({"type": "speaker"}, 2)

    def test_resolve_speaker_reference_index_defaults_single_speaker(self):
        self.assertEqual(resolve_speaker_reference_index({"type": "speaker"}, 1), 0)

    def test_resolve_speaker_reference_index_rejects_out_of_range_value(self):
        with self.assertRaises(ValueError):
            resolve_speaker_reference_index({"type": "speaker", "speaker_index": 2}, 2)

    def test_expand_audio_like_emotion_rows_uses_only_selected_speaker_reference(self):
        speaker_references = [
            WeightedReference(path="speaker_a.wav", weight=0.25),
            WeightedReference(path="speaker_b.wav", weight=0.75),
        ]

        expanded = expand_audio_like_emotion_rows(
            [
                {"type": "speaker", "speaker_index": 1, "weight": 0.6},
                {"type": "audio", "path": "emotion.wav", "weight": 0.4},
            ],
            speaker_references,
        )

        self.assertEqual(
            [(reference.path, reference.weight) for reference in expanded],
            [("speaker_b.wav", 0.6), ("emotion.wav", 0.4)],
        )

    def test_expand_audio_like_emotion_rows_supports_multiple_targeted_speaker_rows(self):
        speaker_references = [
            WeightedReference(path="speaker_a.wav", weight=0.5),
            WeightedReference(path="speaker_b.wav", weight=0.5),
        ]

        expanded = expand_audio_like_emotion_rows(
            [
                {"type": "speaker", "speaker_index": 0, "weight": 0.3},
                {"type": "speaker", "speaker_index": 1, "weight": 0.7},
            ],
            speaker_references,
        )

        self.assertEqual(
            [(reference.path, reference.weight) for reference in expanded],
            [("speaker_a.wav", 0.3), ("speaker_b.wav", 0.7)],
        )

    def test_list_speaker_reference_targets_skips_empty_or_inactive_rows(self):
        targets = list_speaker_reference_targets(
            [
                {"active": True, "audio": "speaker_a.wav"},
                {"active": True, "audio": None},
                {"active": False, "audio": "speaker_unused.wav"},
                {"active": True, "audio": " speaker_b.wav "},
            ]
        )

        self.assertEqual([index for index, _ in targets], [0, 1])
        self.assertTrue(targets[0][1].endswith("speaker_a.wav"))
        self.assertTrue(targets[1][1].endswith("speaker_b.wav"))

    def test_collect_emotion_references_preserves_selected_speaker_index(self):
        refs = collect_emotion_references(
            [
                {
                    "active": True,
                    "type": "speaker",
                    "speaker_index": "1",
                    "weight": 0.6,
                    "audio": None,
                    "text": "",
                    "vector": [0.0] * 8,
                },
                {
                    "active": True,
                    "type": "text",
                    "speaker_index": None,
                    "weight": 0.4,
                    "audio": None,
                    "text": "calm",
                    "vector": [0.0] * 8,
                },
            ],
            "hello",
            lambda values, apply_bias=True: values,
            "speaker",
            "audio",
            "vector",
            "text",
        )

        self.assertEqual(refs[0], {"type": "speaker", "speaker_index": 1, "weight": 0.6})
        self.assertEqual(refs[1]["type"], "text")

    def test_collect_emotion_references_rejects_invalid_speaker_selection(self):
        with self.assertRaises(ValueError):
            collect_emotion_references(
                [
                    {
                        "active": True,
                        "type": "speaker",
                        "speaker_index": "bad",
                        "weight": 1.0,
                        "audio": None,
                        "text": "",
                        "vector": [0.0] * 8,
                    }
                ],
                "hello",
                lambda values, apply_bias=True: values,
                "speaker",
                "audio",
                "vector",
                "text",
            )


if __name__ == "__main__":
    unittest.main()
