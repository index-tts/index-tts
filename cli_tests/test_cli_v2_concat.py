import contextlib
import io
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock


def write_wav(path, channels=1, sample_width=2, frame_rate=24000, frames=8):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(b"\0" * channels * sample_width * frames)


def write_empty_wav(path, channels=1, sample_width=2, frame_rate=24000):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)


def write_wav_frames(path, frames, channels=1, sample_width=1, frame_rate=1000):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(frames)


def read_wav(path):
    with wave.open(str(path), "rb") as wav_file:
        return {
            "channels": wav_file.getnchannels(),
            "sample_width": wav_file.getsampwidth(),
            "frame_rate": wav_file.getframerate(),
            "frames": wav_file.readframes(wav_file.getnframes()),
        }


class working_directory:
    def __init__(self, path):
        self.path = path
        self.previous = None

    def __enter__(self):
        self.previous = Path.cwd()
        os.chdir(self.path)

    def __exit__(self, _exc_type, _exc, _tb):
        os.chdir(self.previous)


class ConcatCommandDryRunTests(unittest.TestCase):
    def run_concat(self, args):
        from indextts.cli_v2 import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(args)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_concat_dry_run_validates_manifest_without_creating_output_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            concat_dir = temp_path / "concat"
            concat_dir.mkdir()
            audio_path = concat_dir / "clip.wav"
            concat_file = concat_dir / "manifest.jsonl"
            output_path = temp_path / "new-parent" / "out.wav"
            write_wav(audio_path)
            concat_file.write_text(
                '\n{"audio": "clip.wav", "silence_after_ms": 125}\n\n',
                encoding="utf-8",
            )

            def fail_if_runtime_imports_are_checked():
                raise AssertionError("concat dry-run must not check runtime packages")

            import indextts.cli_v2 as cli_v2

            original_import_check = cli_v2._import_required_packages
            cli_v2._import_required_packages = fail_if_runtime_imports_are_checked
            try:
                exit_code, stdout, stderr = self.run_concat(
                    [
                        "concat",
                        "--concat-file",
                        str(concat_file),
                        "--output",
                        str(output_path),
                        "--dry-run",
                    ]
                )
            finally:
                cli_v2._import_required_packages = original_import_check

            output_parent_exists = output_path.parent.exists()
            output_exists = output_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "Concat file OK: 1 segments\n")
        self.assertEqual(stderr, "")
        self.assertFalse(output_parent_exists)
        self.assertFalse(output_exists)

    def test_concat_dry_run_rejects_non_object_json_with_1_based_line_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            concat_file = temp_path / "manifest.jsonl"
            concat_file.write_text('\n["not", "an", "object"]\n', encoding="utf-8")

            exit_code, stdout, stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(temp_path / "out.wav"),
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("line 2", stderr)
        self.assertIn("JSON object", stderr)

    def test_concat_dry_run_rejects_unknown_fields_and_comment_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "clip.wav"
            concat_file = temp_path / "manifest.jsonl"
            write_wav(audio_path)
            concat_file.write_text(
                '# {"audio": "clip.wav"}\n{"audio": "clip.wav", "text": "ignored"}\n',
                encoding="utf-8",
            )

            first_exit_code, first_stdout, first_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(temp_path / "out.wav"),
                    "--dry-run",
                ]
            )

            concat_file.write_text('{"audio": "clip.wav", "text": "ignored"}\n', encoding="utf-8")
            second_exit_code, second_stdout, second_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(temp_path / "out.wav"),
                    "--dry-run",
                ]
            )

        self.assertEqual(first_exit_code, 1)
        self.assertEqual(first_stdout, "")
        self.assertIn("line 1", first_stderr)
        self.assertIn("not valid JSON", first_stderr)
        self.assertEqual(second_exit_code, 1)
        self.assertEqual(second_stdout, "")
        self.assertIn("line 1", second_stderr)
        self.assertIn("unknown fields", second_stderr)
        self.assertIn("text", second_stderr)

    def test_concat_dry_run_rejects_invalid_segment_fields_with_line_number(self):
        cases = [
            ('{"silence_after_ms": 0}\n', "missing required field: audio"),
            ('{"audio": ""}\n', "must not be empty"),
            ('{"audio": 123}\n', "must be a string"),
            ('{"audio": "clip.wav", "silence_after_ms": -1}\n', "non-negative integer"),
            ('{"audio": "clip.wav", "silence_after_ms": 1.5}\n', "non-negative integer"),
            ('{"audio": "clip.wav", "silence_after_ms": true}\n', "non-negative integer"),
        ]
        for manifest, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    write_wav(temp_path / "clip.wav")
                    concat_file = temp_path / "manifest.jsonl"
                    concat_file.write_text("\n" + manifest, encoding="utf-8")

                    exit_code, stdout, stderr = self.run_concat(
                        [
                            "concat",
                            "--concat-file",
                            str(concat_file),
                            "--output",
                            str(temp_path / "out.wav"),
                            "--dry-run",
                        ]
                    )

                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "")
                self.assertIn("line 2", stderr)
                self.assertIn(expected_message, stderr)

    def test_concat_dry_run_resolves_command_paths_from_cwd_and_audio_from_concat_file_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cwd_path = temp_path / "cwd"
            concat_dir = temp_path / "manifests"
            assets_dir = concat_dir / "assets"
            cwd_path.mkdir()
            concat_dir.mkdir()
            assets_dir.mkdir()
            write_wav(assets_dir / "CLIP.WAV")
            concat_file = concat_dir / "manifest.jsonl"
            output_path = cwd_path / "OUT.WAV"
            concat_file.write_text('{"audio": "assets/CLIP.WAV"}\n', encoding="utf-8")

            with working_directory(cwd_path):
                exit_code, stdout, stderr = self.run_concat(
                    [
                        "concat",
                        "--concat-file",
                        str(Path("..") / "manifests" / "manifest.jsonl"),
                        "--output",
                        "OUT.WAV",
                        "--dry-run",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "Concat file OK: 1 segments\n")
        self.assertEqual(stderr, "")
        self.assertFalse(output_path.exists())

    def test_concat_dry_run_rejects_non_wav_output_and_audio_extensions_case_insensitively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "clip.mp3"
            concat_file = temp_path / "manifest.jsonl"
            audio_path.write_bytes(b"not checked after extension failure")
            concat_file.write_text('{"audio": "clip.mp3"}\n', encoding="utf-8")

            output_exit_code, output_stdout, output_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(temp_path / "out.mp3"),
                    "--dry-run",
                ]
            )
            audio_exit_code, audio_stdout, audio_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(temp_path / "out.wav"),
                    "--dry-run",
                ]
            )

        self.assertEqual(output_exit_code, 1)
        self.assertEqual(output_stdout, "")
        self.assertIn("--output must be a .wav file", output_stderr)
        self.assertEqual(audio_exit_code, 1)
        self.assertEqual(audio_stdout, "")
        self.assertIn("line 1", audio_stderr)
        self.assertIn("field 'audio' must be a .wav file", audio_stderr)

    def test_concat_dry_run_checks_wav_existence_readability_non_empty_and_format_match(self):
        cases = [
            ("missing.wav", None, 2, "audio file does not exist"),
            ("broken.wav", b"not a wav", 1, "not a readable WAV"),
            ("empty.wav", "empty", 1, "audio file is empty"),
        ]
        for audio_name, audio_content, expected_exit_code, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    concat_file = temp_path / "manifest.jsonl"
                    if audio_content == "empty":
                        write_empty_wav(temp_path / audio_name)
                    elif audio_content is not None:
                        (temp_path / audio_name).write_bytes(audio_content)
                    concat_file.write_text(f'{{"audio": "{audio_name}"}}\n', encoding="utf-8")

                    exit_code, stdout, stderr = self.run_concat(
                        [
                            "concat",
                            "--concat-file",
                            str(concat_file),
                            "--output",
                            str(temp_path / "out.wav"),
                            "--dry-run",
                        ]
                    )

                self.assertEqual(exit_code, expected_exit_code)
                self.assertEqual(stdout, "")
                self.assertIn("line 1", stderr)
                self.assertIn(expected_message, stderr)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            concat_file = temp_path / "manifest.jsonl"
            write_wav(temp_path / "first.wav", frame_rate=24000)
            write_wav(temp_path / "second.wav", frame_rate=22050)
            concat_file.write_text(
                '{"audio": "first.wav"}\n{"audio": "second.wav"}\n',
                encoding="utf-8",
            )

            exit_code, stdout, stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(temp_path / "out.wav"),
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("line 2", stderr)
        self.assertIn("baseline line 1", stderr)
        self.assertIn("WAV format does not match", stderr)

    def test_concat_dry_run_rejects_empty_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            concat_file = temp_path / "manifest.jsonl"
            concat_file.write_text("\n \n", encoding="utf-8")

            exit_code, stdout, stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(temp_path / "out.wav"),
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("at least one segment", stderr)

    def test_concat_dry_run_rejects_output_path_conflicts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "clip.wav"
            concat_file = temp_path / "manifest.jsonl"
            blocked_parent = temp_path / "blocked"
            write_wav(audio_path)
            blocked_parent.write_text("file blocks directory creation", encoding="utf-8")
            concat_file.write_text('{"audio": "clip.wav"}\n', encoding="utf-8")

            same_concat_exit_code, same_concat_stdout, same_concat_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(concat_file),
                    "--dry-run",
                ]
            )
            same_audio_exit_code, same_audio_stdout, same_audio_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(audio_path),
                    "--dry-run",
                ]
            )
            blocked_parent_exit_code, blocked_parent_stdout, blocked_parent_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(blocked_parent / "out.wav"),
                    "--dry-run",
                ]
            )

        self.assertEqual(same_concat_exit_code, 1)
        self.assertEqual(same_concat_stdout, "")
        self.assertIn("--output must not be the same path as --concat-file", same_concat_stderr)
        self.assertEqual(same_audio_exit_code, 1)
        self.assertEqual(same_audio_stdout, "")
        self.assertIn("line 1", same_audio_stderr)
        self.assertIn("conflicts with --output", same_audio_stderr)
        self.assertEqual(blocked_parent_exit_code, 1)
        self.assertEqual(blocked_parent_stdout, "")
        self.assertIn("output parent path cannot be created", blocked_parent_stderr)

    def test_concat_dry_run_rejects_existing_output_unless_force_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "clip.wav"
            concat_file = temp_path / "manifest.jsonl"
            output_path = temp_path / "out.wav"
            write_wav(audio_path)
            output_path.write_bytes(b"existing output")
            concat_file.write_text('{"audio": "clip.wav"}\n', encoding="utf-8")

            reject_exit_code, reject_stdout, reject_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(output_path),
                    "--dry-run",
                ]
            )
            force_exit_code, force_stdout, force_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(output_path),
                    "--dry-run",
                    "--force",
                ]
            )
            output_bytes = output_path.read_bytes()

        self.assertEqual(reject_exit_code, 1)
        self.assertEqual(reject_stdout, "")
        self.assertIn("output file already exists", reject_stderr)
        self.assertEqual(force_exit_code, 0)
        self.assertEqual(force_stdout, "Concat file OK: 1 segments\n")
        self.assertEqual(force_stderr, "")
        self.assertEqual(output_bytes, b"existing output")

    def test_concat_generates_wav_with_manifest_order_and_silence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_path = temp_path / "first.wav"
            second_path = temp_path / "second.wav"
            concat_file = temp_path / "manifest.jsonl"
            output_path = temp_path / "out.wav"
            write_wav_frames(first_path, b"\x01\x02")
            write_wav_frames(second_path, b"\x03")
            concat_file.write_text(
                '{"audio": "first.wav", "silence_after_ms": 2}\n'
                '{"audio": "second.wav", "silence_after_ms": 1}\n',
                encoding="utf-8",
            )

            exit_code, stdout, stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(output_path),
                ]
            )
            output_wav = read_wav(output_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(output_wav["channels"], 1)
        self.assertEqual(output_wav["sample_width"], 1)
        self.assertEqual(output_wav["frame_rate"], 1000)
        self.assertEqual(output_wav["frames"], b"\x01\x02\x00\x00\x03\x00")

    def test_concat_execution_does_not_initialize_or_check_model_resources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "clip.wav"
            concat_file = temp_path / "manifest.jsonl"
            output_path = temp_path / "out.wav"
            write_wav_frames(audio_path, b"\x04\x05")
            concat_file.write_text('{"audio": "clip.wav"}\n', encoding="utf-8")

            with mock.patch(
                "indextts.cli_v2._ensure_user_state",
                side_effect=AssertionError("concat must not initialize user state"),
            ), mock.patch(
                "indextts.cli_v2._missing_model_files",
                side_effect=AssertionError("concat must not check model resources"),
            ):
                exit_code, stdout, stderr = self.run_concat(
                    [
                        "concat",
                        "--concat-file",
                        str(concat_file),
                        "--output",
                        str(output_path),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")

    def test_concat_force_overwrites_existing_output_during_real_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "clip.wav"
            concat_file = temp_path / "manifest.jsonl"
            output_path = temp_path / "out.wav"
            write_wav_frames(audio_path, b"\x04\x05")
            write_wav_frames(output_path, b"\x09")
            concat_file.write_text('{"audio": "clip.wav"}\n', encoding="utf-8")

            reject_exit_code, reject_stdout, reject_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(output_path),
                ]
            )
            force_exit_code, force_stdout, force_stderr = self.run_concat(
                [
                    "concat",
                    "--concat-file",
                    str(concat_file),
                    "--output",
                    str(output_path),
                    "--force",
                ]
            )
            output_wav = read_wav(output_path)

        self.assertEqual(reject_exit_code, 1)
        self.assertEqual(reject_stdout, "")
        self.assertIn("output file already exists", reject_stderr)
        self.assertEqual(force_exit_code, 0)
        self.assertEqual(force_stdout, f"Generated: {output_path}\n")
        self.assertEqual(force_stderr, "")
        self.assertEqual(output_wav["frames"], b"\x04\x05")

    def test_concat_execution_failure_returns_code_4_and_removes_temporary_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "clip.wav"
            concat_file = temp_path / "manifest.jsonl"
            output_path = temp_path / "out.wav"
            write_wav_frames(audio_path, b"\x06")
            concat_file.write_text('{"audio": "clip.wav"}\n', encoding="utf-8")

            import indextts.cli_v2 as cli_v2

            original_replace = cli_v2.os.replace

            def fail_replace(_source, _target):
                raise OSError("replace failed")

            cli_v2.os.replace = fail_replace
            try:
                exit_code, stdout, stderr = self.run_concat(
                    [
                        "concat",
                        "--concat-file",
                        str(concat_file),
                        "--output",
                        str(output_path),
                    ]
                )
            finally:
                cli_v2.os.replace = original_replace
            temporary_wavs = list(temp_path.glob(f".{output_path.name}.*.wav"))

        self.assertEqual(exit_code, 4)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: concat failed: replace failed", stderr)
        self.assertNotIn("WARNING: cleanup failed", stderr)
        self.assertFalse(output_path.exists())
        self.assertEqual(temporary_wavs, [])

    def test_concat_cleanup_failure_is_appended_without_overriding_primary_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "clip.wav"
            concat_file = temp_path / "manifest.jsonl"
            output_path = temp_path / "out.wav"
            write_wav_frames(audio_path, b"\x07")
            concat_file.write_text('{"audio": "clip.wav"}\n', encoding="utf-8")

            import indextts.cli_v2 as cli_v2

            original_replace = cli_v2.os.replace
            original_cleanup = cli_v2._cleanup_concat_temp_file

            def fail_replace(_source, _target):
                raise OSError("replace failed")

            def fail_cleanup(_temp_path):
                return OSError("cannot remove temp")

            cli_v2.os.replace = fail_replace
            cli_v2._cleanup_concat_temp_file = fail_cleanup
            try:
                exit_code, stdout, stderr = self.run_concat(
                    [
                        "concat",
                        "--concat-file",
                        str(concat_file),
                        "--output",
                        str(output_path),
                    ]
                )
            finally:
                cli_v2.os.replace = original_replace
                cli_v2._cleanup_concat_temp_file = original_cleanup
                for temp_wav in temp_path.glob(f".{output_path.name}.*.wav"):
                    temp_wav.unlink()

        self.assertEqual(exit_code, 4)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: concat failed: replace failed", stderr)
        self.assertIn("WARNING: cleanup failed: cannot remove temp", stderr)
        self.assertLess(
            stderr.index("ERROR: concat failed: replace failed"),
            stderr.index("WARNING: cleanup failed: cannot remove temp"),
        )
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
