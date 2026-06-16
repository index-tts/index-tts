import contextlib
import importlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REQUIRED_MODEL_FILES = [
    "config.yaml",
    "bpe.model",
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
]
REQUIRED_MODEL_DIRS = [
    "qwen0.6bemo4-merge",
]
AUX_MODEL_FILES = [
    "hf_cache/semantic_codec_model.safetensors",
    "hf_cache/campplus_cn_common.bin",
    "hf_cache/bigvgan/config.json",
    "hf_cache/bigvgan/bigvgan_generator.pt",
]
AUX_MODEL_DIRS = [
    "hf_cache/w2v-bert-2.0",
]


def make_model_dir(base_dir, include_aux=True):
    model_dir = base_dir / "checkpoints"
    model_dir.mkdir()
    for filename in REQUIRED_MODEL_FILES:
        (model_dir / filename).write_text("placeholder", encoding="utf-8")
    for dirname in REQUIRED_MODEL_DIRS:
        (model_dir / dirname).mkdir()
    if include_aux:
        make_aux_model_cache(model_dir)
    return model_dir


def make_aux_model_cache(model_dir):
    for filename in AUX_MODEL_FILES:
        target = model_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder", encoding="utf-8")
    for dirname in AUX_MODEL_DIRS:
        target = model_dir / dirname
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("placeholder", encoding="utf-8")


def assert_model_resource_help(test_case, stderr, model_dir):
    test_case.assertIn(f"Model directory: {model_dir}", stderr)
    test_case.assertIn("Missing resources:", stderr)
    test_case.assertIn("huggingface-cli download IndexTeam/IndexTTS-2", stderr)
    test_case.assertIn("modelscope download --model IndexTeam/IndexTTS-2", stderr)
    test_case.assertIn(f"indextts2 config set model_dir {model_dir}", stderr)


def user_state_env(temp_path):
    if sys.platform == "win32":
        return {
            "APPDATA": str(temp_path / "roaming"),
            "LOCALAPPDATA": str(temp_path / "local"),
        }
    if sys.platform == "darwin":
        return {"HOME": str(temp_path)}
    return {
        "XDG_CONFIG_HOME": str(temp_path / "config"),
        "XDG_DATA_HOME": str(temp_path / "data"),
    }


def fake_torch(cuda=False, xpu=False, mps=False, cuda_device_count=0, xpu_device_count=0):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: cuda_device_count),
        xpu=SimpleNamespace(is_available=lambda: xpu, device_count=lambda: xpu_device_count),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


def patched_imports(torch_module):
    real_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "torch":
            return torch_module
        if name in {"torchaudio", "indextts"}:
            return SimpleNamespace(__name__=name)
        return real_import_module(name, package)

    return mock.patch("importlib.import_module", side_effect=import_module)


def patched_missing_import(missing_package, torch_module):
    real_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == missing_package:
            raise ImportError(name)
        if name == "torch":
            return torch_module
        if name in {"torchaudio", "indextts"}:
            return SimpleNamespace(__name__=name)
        return real_import_module(name, package)

    return mock.patch("importlib.import_module", side_effect=import_module)


class CheckCommandTests(unittest.TestCase):
    def setUp(self):
        self.user_state = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(os.environ, user_state_env(Path(self.user_state.name)), clear=False)
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.user_state.cleanup()

    def test_pyproject_registers_indextts2_without_replacing_existing_indextts_command(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('indextts = "indextts.cli:main"', pyproject)
        self.assertIn('indextts2 = "indextts.cli_v2:main"', pyproject)

    def test_check_returns_success_when_resources_packages_and_requested_device_are_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = make_model_dir(Path(temp_dir))

            with patched_imports(fake_torch(cuda=True)):
                from indextts.cli_v2 import main

                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exit_code = main(["check", "--model-dir", str(model_dir), "--device", "cuda"])

            self.assertEqual(exit_code, 0)
            self.assertIn(f"Checking model directory: {model_dir}", stdout.getvalue())
            self.assertIn("OK: model directory", stdout.getvalue())
            self.assertIn("OK: required model files", stdout.getvalue())
            self.assertIn("OK: python packages", stdout.getvalue())
            self.assertIn("cuda: available", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_check_returns_resource_error_when_model_directory_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_model_dir = Path(temp_dir) / "missing"

            from indextts.cli_v2 import main

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(["check", "--model-dir", str(missing_model_dir)])

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("ERROR: model directory does not exist", stderr.getvalue())
            self.assertIn(str(missing_model_dir), stderr.getvalue())

    def test_check_returns_resource_error_when_required_model_files_are_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "checkpoints"
            model_dir.mkdir()
            (model_dir / "config.yaml").write_text("placeholder", encoding="utf-8")

            from indextts.cli_v2 import main

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(["check", "--model-dir", str(model_dir)])

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("ERROR: missing required model files", stderr.getvalue())
            self.assertIn("bpe.model", stderr.getvalue())
            self.assertIn("gpt.pth", stderr.getvalue())
            assert_model_resource_help(self, stderr.getvalue(), model_dir)

    def test_check_requires_the_full_key_model_resource_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "checkpoints"
            model_dir.mkdir()
            for filename in [
                "config.yaml",
                "bpe.model",
                "gpt.pth",
                "s2mel.pth",
                "wav2vec2bert_stats.pt",
            ]:
                (model_dir / filename).write_text("placeholder", encoding="utf-8")

            from indextts.cli_v2 import main

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(["check", "--model-dir", str(model_dir)])

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("feat1.pt", stderr.getvalue())
            self.assertIn("feat2.pt", stderr.getvalue())
            self.assertIn("qwen0.6bemo4-merge", stderr.getvalue())

    def test_check_requires_the_auxiliary_model_cache_resources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = make_model_dir(Path(temp_dir), include_aux=False)

            from indextts.cli_v2 import main

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(["check", "--model-dir", str(model_dir)])

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("ERROR: missing required model files", stderr.getvalue())
            self.assertIn("hf_cache/w2v-bert-2.0", stderr.getvalue())
            self.assertIn("hf_cache/semantic_codec_model.safetensors", stderr.getvalue())
            self.assertIn("hf_cache/campplus_cn_common.bin", stderr.getvalue())
            self.assertIn("hf_cache/bigvgan/config.json", stderr.getvalue())
            self.assertIn("hf_cache/bigvgan/bigvgan_generator.pt", stderr.getvalue())

    def test_check_requires_file_resources_and_directory_resources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "checkpoints"
            model_dir.mkdir()
            for filename in REQUIRED_MODEL_FILES:
                if filename == "gpt.pth":
                    (model_dir / filename).mkdir()
                else:
                    (model_dir / filename).write_text("placeholder", encoding="utf-8")
            (model_dir / "qwen0.6bemo4-merge").write_text("placeholder", encoding="utf-8")

            from indextts.cli_v2 import main

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(["check", "--model-dir", str(model_dir)])

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("gpt.pth", stderr.getvalue())
            self.assertIn("qwen0.6bemo4-merge", stderr.getvalue())

    def test_check_returns_runtime_error_when_required_python_package_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = make_model_dir(Path(temp_dir))

            with patched_missing_import("torchaudio", fake_torch(cuda=True)):
                from indextts.cli_v2 import main

                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exit_code = main(["check", "--model-dir", str(model_dir)])

            self.assertEqual(exit_code, 3)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("ERROR: missing required Python packages", stderr.getvalue())
            self.assertIn("torchaudio", stderr.getvalue())

    def test_check_returns_runtime_error_when_requested_device_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = make_model_dir(Path(temp_dir))

            with patched_imports(fake_torch(cuda=False)):
                from indextts.cli_v2 import main

                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exit_code = main(["check", "--model-dir", str(model_dir), "--device", "cuda:0"])

            self.assertEqual(exit_code, 3)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("ERROR: requested device is not available: cuda:0", stderr.getvalue())

    def test_check_returns_runtime_error_when_requested_cuda_index_does_not_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = make_model_dir(Path(temp_dir))

            with patched_imports(fake_torch(cuda=True, cuda_device_count=1)):
                from indextts.cli_v2 import main

                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exit_code = main(["check", "--model-dir", str(model_dir), "--device", "cuda:1"])

            self.assertEqual(exit_code, 3)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("ERROR: requested device is not available: cuda:1", stderr.getvalue())

    def test_check_returns_runtime_error_when_requested_xpu_index_does_not_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = make_model_dir(Path(temp_dir))

            with patched_imports(fake_torch(xpu=True, xpu_device_count=1)):
                from indextts.cli_v2 import main

                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exit_code = main(["check", "--model-dir", str(model_dir), "--device", "xpu:1"])

            self.assertEqual(exit_code, 3)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("ERROR: requested device is not available: xpu:1", stderr.getvalue())


class SynthCommandTests(unittest.TestCase):
    def setUp(self):
        self.user_state = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(os.environ, user_state_env(Path(self.user_state.name)), clear=False)
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.user_state.cleanup()

    def run_synth(
        self,
        temp_path,
        args,
        stdin=None,
        fail_init=False,
        fail_infer=False,
        noisy=False,
        add_model_dir=True,
    ):
        calls = []
        if add_model_dir and "--model-dir" not in args:
            model_dir = make_model_dir(temp_path)
            args = [*args, "--model-dir", str(model_dir)]

        class FakeIndexTTS2:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))
                if noisy:
                    print("model init noise")
                if fail_init:
                    raise RuntimeError("load boom")

            def infer(self, **kwargs):
                calls.append(("infer", kwargs))
                if noisy:
                    print("model infer noise")
                if fail_infer:
                    raise RuntimeError("boom")

        from indextts.cli_v2 import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(args, tts_factory=FakeIndexTTS2, stdin=stdin)
        return exit_code, stdout.getvalue(), stderr.getvalue(), calls

    def test_synth_generates_audio_from_inline_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "  hello  ",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(
            calls,
            [
                (
                    "init",
                    {
                        "cfg_path": str(temp_path / "checkpoints" / "config.yaml"),
                        "model_dir": str(temp_path / "checkpoints"),
                        "use_fp16": False,
                        "device": None,
                        "use_cuda_kernel": False,
                        "use_deepspeed": False,
                    },
                ),
                (
                    "infer",
                    {
                        "spk_audio_prompt": str(voice_path),
                        "text": "hello",
                        "output_path": str(output_path),
                        "verbose": False,
                    },
                ),
            ],
        )

    def test_synth_generates_audio_from_utf8_text_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            text_path = temp_path / "input.txt"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            text_path.write_text("  你好, IndexTTS2  ", encoding="utf-8")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text-file",
                    str(text_path),
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][1]["text"], "你好, IndexTTS2")

    def test_synth_generates_audio_from_stdin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--stdin",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
                stdin=io.StringIO("  stdin text  "),
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][1]["text"], "stdin text")

    def test_synth_uses_emotion_audio_and_weight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            emotion_path = temp_path / "emotion.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            emotion_path.write_bytes(b"emotion")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-audio",
                    str(emotion_path),
                    "--emotion-weight",
                    "0.75",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][1]["emo_audio_prompt"], str(emotion_path))
        self.assertEqual(calls[1][1]["emo_alpha"], 0.75)
        self.assertNotIn("use_emo_text", calls[1][1])
        self.assertNotIn("emo_text", calls[1][1])

    def test_synth_uses_emotion_text_and_weight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-text",
                    "warm and calm",
                    "--emotion-weight",
                    "0.6",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertNotIn("emo_audio_prompt", calls[1][1])
        self.assertEqual(calls[1][1]["use_emo_text"], True)
        self.assertEqual(calls[1][1]["emo_text"], "warm and calm")
        self.assertEqual(calls[1][1]["emo_alpha"], 0.6)

    def test_synth_uses_emotion_vector_and_weight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-vector",
                    "0,0,0.8,0,0,0,0,0",
                    "--emotion-weight",
                    "0.7",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][1]["emo_vector"], [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(calls[1][1]["emo_alpha"], 0.7)
        self.assertNotIn("emo_audio_prompt", calls[1][1])
        self.assertNotIn("use_emo_text", calls[1][1])
        self.assertNotIn("emo_text", calls[1][1])

    def test_synth_accepts_python_list_style_emotion_vector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-vector",
                    "[0, 0, 0.8, 0, 0, 0, 0, 0]",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][1]["emo_vector"], [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(calls[1][1]["emo_alpha"], 1.0)

    def test_synth_does_not_rewrite_valid_emotion_vector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-vector",
                    "0.12,0.03,0.25,0.04,0,0.11,0.07,0.02",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][1]["emo_vector"], [0.12, 0.03, 0.25, 0.04, 0.0, 0.11, 0.07, 0.02])

    def test_synth_returns_input_error_when_emotion_vector_is_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-vector",
                    "",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: --emotion-vector must not be empty", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_emotion_vector_is_invalid(self):
        cases = [
            ("0,0,nope,0,0,0,0,0", "entries must be numeric"),
            ("0,0,0.8,0,0,0,0", "exactly 8 values"),
            ("0,0,0.8,0,0,0,0,0,0", "exactly 8 values"),
            ("0,0,-0.1,0,0,0,0,0", "between 0.0 and 1.0"),
            ("0,0,1.1,0,0,0,0,0", "between 0.0 and 1.0"),
            ("0.2,0.2,0.2,0.2,0.1,0,0,0", "sum must be <= 0.8"),
        ]

        for vector, expected_error in cases:
            with self.subTest(vector=vector):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    voice_path = temp_path / "voice.wav"
                    output_path = temp_path / "out.wav"
                    voice_path.write_bytes(b"voice")

                    exit_code, stdout, stderr, calls = self.run_synth(
                        temp_path,
                        [
                            "synth",
                            "--text",
                            "hello",
                            "--voice",
                            str(voice_path),
                            "--emotion-vector",
                            vector,
                            "--output",
                            str(output_path),
                        ],
                    )

                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "")
                self.assertIn("ERROR: --emotion-vector", stderr)
                self.assertIn(expected_error, stderr)
                self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_emotion_vector_conflicts_with_other_emotion_sources(self):
        for other_emotion_args in (["--emotion-audio"], ["--emotion-text", "warm and calm"]):
            with self.subTest(other_emotion_args=other_emotion_args):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    voice_path = temp_path / "voice.wav"
                    emotion_path = temp_path / "emotion.wav"
                    output_path = temp_path / "out.wav"
                    voice_path.write_bytes(b"voice")
                    emotion_path.write_bytes(b"emotion")
                    args = [
                        "synth",
                        "--text",
                        "hello",
                        "--voice",
                        str(voice_path),
                        "--emotion-vector",
                        "0,0,0.8,0,0,0,0,0",
                        "--output",
                        str(output_path),
                    ]
                    if other_emotion_args == ["--emotion-audio"]:
                        args.extend(["--emotion-audio", str(emotion_path)])
                    else:
                        args.extend(other_emotion_args)

                    exit_code, stdout, stderr, calls = self.run_synth(temp_path, args)

                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "")
                self.assertIn("--emotion-vector, --emotion-audio and --emotion-text are mutually exclusive", stderr)
                self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_emotion_text_is_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-text",
                    "",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: --emotion-text must not be empty", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_emotion_sources_conflict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            emotion_path = temp_path / "emotion.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            emotion_path.write_bytes(b"emotion")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-audio",
                    str(emotion_path),
                    "--emotion-text",
                    "warm and calm",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: --emotion-audio and --emotion-text are mutually exclusive", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_empty_emotion_audio_conflicts_with_emotion_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-audio",
                    "",
                    "--emotion-text",
                    "warm and calm",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: --emotion-audio and --emotion-text are mutually exclusive", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_resource_error_when_emotion_audio_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            emotion_path = temp_path / "missing-emotion.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-audio",
                    str(emotion_path),
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: emotion reference audio does not exist", stderr)
        self.assertIn(str(emotion_path), stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_emotion_weight_is_not_a_float(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            emotion_path = temp_path / "emotion.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            emotion_path.write_bytes(b"emotion")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--emotion-audio",
                    str(emotion_path),
                    "--emotion-weight",
                    "strong",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: --emotion-weight must be a float", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_text_source_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: provide exactly one text source", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_text_sources_conflict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            text_path = temp_path / "input.txt"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            text_path.write_text("file text", encoding="utf-8")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "inline",
                    "--text-file",
                    str(text_path),
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: provide exactly one text source", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_empty_text_source_conflicts_with_stdin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "",
                    "--stdin",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
                stdin=io.StringIO("stdin text"),
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: provide exactly one text source", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_text_is_empty_after_trimming(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    " \t\r\n ",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: text is empty", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_resource_error_when_text_file_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            text_path = temp_path / "missing.txt"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text-file",
                    str(text_path),
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: text file does not exist", stderr)
        self.assertIn(str(text_path), stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_resource_error_when_voice_file_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "missing.wav"
            output_path = temp_path / "out.wav"

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: voice reference audio does not exist", stderr)
        self.assertIn(str(voice_path), stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_voice_argument_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "out.wav"

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: --voice is required", stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_output_exists_without_force(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            output_path.write_bytes(b"existing")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: output file already exists", stderr)
        self.assertIn(str(output_path), stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_input_error_when_output_argument_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                ],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: --output is required", stderr)
        self.assertEqual(calls, [])

    def test_synth_allows_existing_output_when_force_is_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")
            output_path.write_bytes(b"existing")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                    "--force",
                ],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][1]["output_path"], str(output_path))

    def test_synth_creates_output_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "nested" / "audio" / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
            )
            output_parent_exists = output_path.parent.is_dir()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_parent_exists)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][1]["output_path"], str(output_path))

    def test_synth_maps_runtime_options_to_indextts2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                    "--model-dir",
                    str(model_dir),
                    "--device",
                    "cuda:0",
                    "--fp16",
                    "--deepspeed",
                    "--cuda-kernel",
                    "--verbose",
                ],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
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

    def test_synth_returns_inference_error_when_indextts2_infer_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
                fail_infer=True,
            )

        self.assertEqual(exit_code, 4)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: inference failed: boom", stderr)
        self.assertEqual(calls[0][0], "init")
        self.assertEqual(calls[1][0], "infer")

    def test_synth_returns_inference_error_when_indextts2_initialization_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
                fail_init=True,
            )

        self.assertEqual(exit_code, 4)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: inference failed: load boom", stderr)
        self.assertEqual(calls[0][0], "init")

    def test_synth_returns_resource_error_when_model_directory_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            missing_model_dir = temp_path / "missing-models"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                    "--model-dir",
                    str(missing_model_dir),
                ],
                add_model_dir=False,
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: model directory does not exist", stderr)
        self.assertIn(str(missing_model_dir), stderr)
        self.assertEqual(calls, [])

    def test_synth_returns_resource_error_when_model_file_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = temp_path / "models"
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            model_dir.mkdir()
            (model_dir / "config.yaml").write_text("placeholder", encoding="utf-8")
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                    "--model-dir",
                    str(model_dir),
                ],
                add_model_dir=False,
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: missing required model files", stderr)
        self.assertIn("bpe.model", stderr)
        assert_model_resource_help(self, stderr, model_dir)
        self.assertEqual(calls, [])

    def test_synth_returns_runtime_error_when_indextts2_import_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = make_model_dir(temp_path)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            from indextts.cli_v2 import main

            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch("indextts.cli_v2._load_indextts2", side_effect=ImportError("torch")):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exit_code = main(
                        [
                            "synth",
                            "--text",
                            "hello",
                            "--voice",
                            str(voice_path),
                            "--output",
                            str(output_path),
                            "--model-dir",
                            str(model_dir),
                        ]
                    )

        self.assertEqual(exit_code, 3)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("ERROR: runtime unavailable: torch", stderr.getvalue())

    def test_load_indextts2_points_huggingface_cache_at_model_resource_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"

            class FakeIndexTTS2:
                pass

            from indextts.cli_v2 import _load_indextts2

            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": "legacy-cache"}, clear=False):
                with mock.patch.dict(
                    sys.modules,
                    {"indextts.infer_v2": SimpleNamespace(IndexTTS2=FakeIndexTTS2)},
                    clear=False,
                ):
                    loaded = _load_indextts2(model_dir)
                    hf_hub_cache = os.environ["HF_HUB_CACHE"]

            self.assertIs(loaded, FakeIndexTTS2)
            self.assertEqual(hf_hub_cache, str(model_dir / "hf_cache"))

    def test_synth_suppresses_model_stdout_when_not_verbose(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                ],
                noisy=True,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertNotIn("model init noise", stdout)
        self.assertNotIn("model infer noise", stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][0], "infer")

    def test_synth_allows_model_stdout_when_verbose(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            voice_path.write_bytes(b"voice")

            exit_code, stdout, stderr, calls = self.run_synth(
                temp_path,
                [
                    "synth",
                    "--text",
                    "hello",
                    "--voice",
                    str(voice_path),
                    "--output",
                    str(output_path),
                    "--verbose",
                ],
                noisy=True,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("model init noise", stdout)
        self.assertIn("model infer noise", stdout)
        self.assertIn(f"Generated: {output_path}\n", stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(calls[1][0], "infer")


if __name__ == "__main__":
    unittest.main()
