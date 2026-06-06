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


class BatchCommandExecutionTests(unittest.TestCase):
    def run_batch(self, args, tts_factory=None):
        from indextts.cli_v2 import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(args, tts_factory=tts_factory)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_batch_executes_tasks_in_order_with_one_model_initialization_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_dir = temp_path / "batch"
            batch_dir.mkdir()
            voice_path = batch_dir / "voice.wav"
            first_output = batch_dir / "first.wav"
            second_output = batch_dir / "second.wav"
            batch_file = batch_dir / "batch.jsonl"
            calls = []
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                "\n".join(
                    [
                        '{"text": "first", "voice": "voice.wav", "output": "first.wav"}',
                        '{"text": "second", "voice": "voice.wav", "output": "second.wav"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            class FakeIndexTTS2:
                def __init__(self, **kwargs):
                    calls.append(("init", kwargs))

                def infer(self, **kwargs):
                    calls.append(("infer", kwargs))
                    Path(kwargs["output_path"]).write_bytes(kwargs["text"].encode("utf-8"))

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                ],
                tts_factory=FakeIndexTTS2,
            )
            first_output_bytes = first_output.read_bytes()
            second_output_bytes = second_output.read_bytes()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout,
            f"Generated: {first_output}\nGenerated: {second_output}\nBatch complete: 2 tasks generated\n",
        )
        self.assertEqual(stderr, "")
        self.assertEqual([call[0] for call in calls], ["init", "infer", "infer"])
        self.assertEqual(calls[1][1]["text"], "first")
        self.assertEqual(calls[2][1]["text"], "second")
        self.assertEqual(calls[1][1]["spk_audio_prompt"], str(voice_path))
        self.assertEqual(calls[2][1]["spk_audio_prompt"], str(voice_path))
        self.assertEqual(first_output_bytes, b"first")
        self.assertEqual(second_output_bytes, b"second")

    def test_batch_maps_command_runtime_options_to_indextts2_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_file = temp_path / "batch.jsonl"
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            calls = []
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                '{"text": "hello", "voice": "voice.wav", "output": "out.wav"}\n',
                encoding="utf-8",
            )

            class FakeIndexTTS2:
                def __init__(self, **kwargs):
                    calls.append(("init", kwargs))

                def infer(self, **kwargs):
                    calls.append(("infer", kwargs))
                    Path(kwargs["output_path"]).write_bytes(b"audio")

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--device",
                    "cuda:0",
                    "--fp16",
                    "--deepspeed",
                    "--cuda-kernel",
                    "--verbose",
                ],
                tts_factory=FakeIndexTTS2,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertIn(f"Generated: {output_path}\n", stdout)
        self.assertEqual(
            calls[0][1],
            {
                "cfg_path": str(model_dir / "config.yaml"),
                "model_dir": str(model_dir),
                "use_fp16": True,
                "device": "cuda:0",
                "use_cuda_kernel": True,
                "use_deepspeed": True,
            },
        )
        self.assertTrue(calls[1][1]["verbose"])

    def test_batch_stops_on_first_inference_failure_and_keeps_prior_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_file = temp_path / "batch.jsonl"
            voice_path = temp_path / "voice.wav"
            first_output = temp_path / "first.wav"
            second_output = temp_path / "second.wav"
            third_output = temp_path / "third.wav"
            calls = []
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                "\n".join(
                    [
                        '{"text": "first", "voice": "voice.wav", "output": "first.wav"}',
                        '{"text": "second", "voice": "voice.wav", "output": "second.wav"}',
                        '{"text": "third", "voice": "voice.wav", "output": "third.wav"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            class FakeIndexTTS2:
                def __init__(self, **kwargs):
                    calls.append(("init", kwargs))

                def infer(self, **kwargs):
                    calls.append(("infer", kwargs))
                    if kwargs["text"] == "second":
                        raise RuntimeError("boom")
                    Path(kwargs["output_path"]).write_bytes(kwargs["text"].encode("utf-8"))

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                ],
                tts_factory=FakeIndexTTS2,
            )
            first_output_bytes = first_output.read_bytes()
            second_output_exists = second_output.exists()
            third_output_exists = third_output.exists()

        self.assertEqual(exit_code, 4)
        self.assertEqual(stdout, f"Generated: {first_output}\n")
        self.assertIn("ERROR: batch file line 2 inference failed: boom", stderr)
        self.assertEqual([call[0] for call in calls], ["init", "infer", "infer"])
        self.assertEqual(first_output_bytes, b"first")
        self.assertFalse(second_output_exists)
        self.assertFalse(third_output_exists)
        self.assertNotIn("Batch complete", stdout)

    def test_batch_rejects_existing_external_output_without_force_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_file = temp_path / "batch.jsonl"
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            output_path.write_bytes(b"existing")
            batch_file.write_text(
                '{"text": "hello", "voice": "voice.wav", "output": "out.wav"}\n',
                encoding="utf-8",
            )

            def fail_if_called(**_kwargs):
                raise AssertionError("tts factory must not be called when output precheck fails")

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                ],
                tts_factory=fail_if_called,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: batch file line 1 output file already exists", stderr)
        self.assertIn(str(output_path), stderr)

    def test_batch_force_allows_existing_external_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_file = temp_path / "batch.jsonl"
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            output_path.write_bytes(b"existing")
            batch_file.write_text(
                '{"text": "hello", "voice": "voice.wav", "output": "out.wav"}\n',
                encoding="utf-8",
            )

            class FakeIndexTTS2:
                def __init__(self, **_kwargs):
                    pass

                def infer(self, **kwargs):
                    Path(kwargs["output_path"]).write_bytes(b"new audio")

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(model_dir),
                    "--force",
                ],
                tts_factory=FakeIndexTTS2,
            )
            output_bytes = output_path.read_bytes()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\nBatch complete: 1 tasks generated\n")
        self.assertEqual(stderr, "")
        self.assertEqual(output_bytes, b"new audio")

    def test_batch_rejects_runtime_options_inside_batch_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            batch_file = temp_path / "batch.jsonl"
            voice_path = temp_path / "voice.wav"
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                '{"text": "hello", "voice": "voice.wav", "output": "out.wav", "device": "cpu"}\n',
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
        self.assertIn("device", stderr)

    def test_batch_returns_resource_error_when_model_directory_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            batch_file = temp_path / "batch.jsonl"
            voice_path = temp_path / "voice.wav"
            missing_model_dir = temp_path / "missing-models"
            voice_path.write_bytes(b"voice")
            batch_file.write_text(
                '{"text": "hello", "voice": "voice.wav", "output": "out.wav"}\n',
                encoding="utf-8",
            )

            exit_code, stdout, stderr = self.run_batch(
                [
                    "batch",
                    "--batch-file",
                    str(batch_file),
                    "--model-dir",
                    str(missing_model_dir),
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: model directory does not exist", stderr)
        self.assertIn(str(missing_model_dir), stderr)

    def test_batch_returns_runtime_error_when_indextts2_import_fails(self):
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

            with mock.patch("indextts.cli_v2._load_indextts2", side_effect=ImportError("torch")):
                exit_code, stdout, stderr = self.run_batch(
                    [
                        "batch",
                        "--batch-file",
                        str(batch_file),
                        "--model-dir",
                        str(model_dir),
                    ]
                )

        self.assertEqual(exit_code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: runtime unavailable: torch", stderr)


if __name__ == "__main__":
    unittest.main()
