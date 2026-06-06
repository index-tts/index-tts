import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REQUIRED_MODEL_FILES = [
    "config.yaml",
    "bpe.model",
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
]


def make_model_dir(base_dir):
    model_dir = base_dir / "checkpoints"
    model_dir.mkdir()
    for filename in REQUIRED_MODEL_FILES:
        (model_dir / filename).write_text("placeholder", encoding="utf-8")
    return model_dir


class BatchCommandDryRunTests(unittest.TestCase):
    def run_batch(self, args, tts_factory=None):
        from indextts.cli_v2 import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(args, tts_factory=tts_factory)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_batch_dry_run_validates_manifest_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_dir = temp_path / "batch"
            batch_dir.mkdir()
            voice_path = batch_dir / "voice.wav"
            batch_file = batch_dir / "batch.jsonl"
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                '\n{"text": "hello", "voice": "voice.wav", "output": "out.wav"}\n\n',
                encoding="utf-8",
            )

            def fail_if_called(**_kwargs):
                raise AssertionError("tts factory must not be called during dry-run")

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--dry-run",
                ],
                tts_factory=fail_if_called,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "Batch file OK: 1 tasks\n")
        self.assertEqual(stderr, "")

    def test_batch_dry_run_rejects_non_object_json_with_1_based_line_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_file = temp_path / "batch.jsonl"
            batch_file.write_text('\n["not", "an", "object"]\n', encoding="utf-8")

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("line 2", stderr)
        self.assertIn("JSON object", stderr)

    def test_batch_dry_run_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            voice_path = temp_path / "voice.wav"
            batch_file = temp_path / "batch.jsonl"
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                '{"text": "hello", "voice": "voice.wav", "output": "out.wav", "bogus": true}\n',
                encoding="utf-8",
            )

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("line 1", stderr)
        self.assertIn("unknown fields", stderr)
        self.assertIn("bogus", stderr)

    def test_batch_dry_run_rejects_conflicting_text_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            voice_path = temp_path / "voice.wav"
            text_path = temp_path / "input.txt"
            batch_file = temp_path / "batch.jsonl"
            voice_path.write_bytes(b"voice")
            text_path.write_text("hello from file", encoding="utf-8")
            batch_file.write_text(
                '{"text": "hello", "text_file": "input.txt", "voice": "voice.wav", "output": "out.wav"}\n',
                encoding="utf-8",
            )

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("line 1", stderr)
        self.assertIn("exactly one text source", stderr)

    def test_batch_dry_run_rejects_missing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            voice_path = temp_path / "voice.wav"
            batch_file = temp_path / "batch.jsonl"
            voice_path.write_bytes(b"voice")
            batch_file.write_text('{"text": "hello", "voice": "voice.wav"}\n', encoding="utf-8")

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("line 1", stderr)
        self.assertIn("missing required field: output", stderr)

    def test_batch_dry_run_rejects_duplicate_output_paths_with_line_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            voice_path = temp_path / "voice.wav"
            batch_file = temp_path / "batch.jsonl"
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                "\n".join(
                    [
                        '{"text": "hello", "voice": "voice.wav", "output": "out.wav"}',
                        '{"text": "world", "voice": "voice.wav", "output": "out.wav"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("line 2", stderr)
        self.assertIn("duplicate output", stderr)

    def test_batch_dry_run_resolves_text_file_and_voice_relative_to_batch_file_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_dir = temp_path / "batch"
            assets_dir = batch_dir / "assets"
            batch_dir.mkdir()
            assets_dir.mkdir()
            voice_path = assets_dir / "voice.wav"
            text_path = assets_dir / "input.txt"
            batch_file = batch_dir / "batch.jsonl"
            voice_path.write_bytes(b"voice")
            text_path.write_text("hello from file", encoding="utf-8")
            batch_file.write_text(
                '{"text_file": "assets/input.txt", "voice": "assets/voice.wav", "output": "out.wav"}\n',
                encoding="utf-8",
            )

            def fail_if_called(**_kwargs):
                raise AssertionError("tts factory must not be called during dry-run")

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--dry-run",
                ],
                tts_factory=fail_if_called,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "Batch file OK: 1 tasks\n")
        self.assertEqual(stderr, "")

    def test_batch_dry_run_checks_model_files_without_importing_runtime_packages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_file = temp_path / "batch.jsonl"
            voice_path = temp_path / "voice.wav"
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                '{"text": "hello", "voice": "voice.wav", "output": "out.wav"}\n',
                encoding="utf-8",
            )

            with mock.patch("indextts.cli_v2._import_required_packages", side_effect=AssertionError("must not import")):
                exit_code, stdout, stderr = self.run_batch(
                    [
                        "batch",
                        "--batch-file",
                        str(batch_file),
                        "--model-dir",
                        str(model_dir),
                        "--dry-run",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "Batch file OK: 1 tasks\n")
        self.assertEqual(stderr, "")

    def test_batch_dry_run_with_force_still_rejects_duplicate_output_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            voice_path = temp_path / "voice.wav"
            batch_file = temp_path / "batch.jsonl"
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                "\n".join(
                    [
                        '{"text": "hello", "voice": "voice.wav", "output": "out.wav"}',
                        '{"text": "world", "voice": "voice.wav", "output": "out.wav"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--dry-run",
                    "--force",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("line 2", stderr)
        self.assertIn("duplicate output", stderr)


if __name__ == "__main__":
    unittest.main()
