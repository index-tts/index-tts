import contextlib
import io
import importlib
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


def make_model_dir(path):
    path.mkdir(parents=True)
    for filename in REQUIRED_MODEL_FILES:
        (path / filename).write_text("placeholder", encoding="utf-8")
    for dirname in REQUIRED_MODEL_DIRS:
        (path / dirname).mkdir()
    for filename in AUX_MODEL_FILES:
        target = path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder", encoding="utf-8")
    for dirname in AUX_MODEL_DIRS:
        target = path / dirname
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("placeholder", encoding="utf-8")


def fake_torch():
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        xpu=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )


def patched_imports():
    real_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "torch":
            return fake_torch()
        if name in {"torchaudio", "indextts"}:
            return SimpleNamespace(__name__=name)
        return real_import_module(name, package)

    return mock.patch("importlib.import_module", side_effect=import_module)


def user_state_paths(temp_path):
    if sys.platform == "win32":
        return {
            "env": {
                "APPDATA": str(temp_path / "roaming"),
                "LOCALAPPDATA": str(temp_path / "local"),
            },
            "config_path": temp_path / "roaming" / "IndexTTS" / "config.toml",
            "model_dir": temp_path / "local" / "IndexTTS" / "models" / "IndexTTS-2",
        }
    if sys.platform == "darwin":
        app_support = temp_path / "Library" / "Application Support" / "IndexTTS"
        return {
            "env": {"HOME": str(temp_path)},
            "config_path": app_support / "config.toml",
            "model_dir": app_support / "models" / "IndexTTS-2",
        }
    return {
        "env": {
            "XDG_CONFIG_HOME": str(temp_path / "config"),
            "XDG_DATA_HOME": str(temp_path / "data"),
        },
        "config_path": temp_path / "config" / "indextts" / "config.toml",
        "model_dir": temp_path / "data" / "indextts" / "models" / "IndexTTS-2",
    }


class ConfigCommandTests(unittest.TestCase):
    def run_cli(self, args, **kwargs):
        from indextts.cli_v2 import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(args, **kwargs)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_init_creates_persistent_config_and_default_model_directory_without_model_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = user_state_paths(Path(temp_dir))

            with mock.patch.dict(os.environ, state["env"], clear=False):
                exit_code, stdout, stderr = self.run_cli(["init"])

            config_text = state["config_path"].read_text(encoding="utf-8")
            model_dir_files = list(state["model_dir"].iterdir())

        self.assertEqual(exit_code, 0)
        self.assertIn(f"Config: {state['config_path']}", stdout)
        self.assertIn(f"Model directory: {state['model_dir']}", stdout)
        self.assertEqual(stderr, "")
        self.assertIn(f'model_dir = "{state["model_dir"].as_posix()}"', config_text)
        self.assertEqual(model_dir_files, [])

    def test_init_with_model_dir_persists_the_requested_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            model_dir = temp_path / "custom-models"

            with mock.patch.dict(os.environ, state["env"], clear=False):
                exit_code, stdout, stderr = self.run_cli(["init", "--model-dir", str(model_dir)])

            config_text = state["config_path"].read_text(encoding="utf-8")
            model_dir_exists = model_dir.is_dir()

        self.assertEqual(exit_code, 0)
        self.assertIn(f"Model directory: {model_dir}", stdout)
        self.assertEqual(stderr, "")
        self.assertTrue(model_dir_exists)
        self.assertIn(f'model_dir = "{model_dir.as_posix()}"', config_text)

    def test_config_path_prints_the_persistent_config_file_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = user_state_paths(Path(temp_dir))

            with mock.patch.dict(os.environ, state["env"], clear=False):
                exit_code, stdout, stderr = self.run_cli(["config", "path"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"{state['config_path']}\n")
        self.assertEqual(stderr, "")

    def test_config_set_model_dir_persists_the_model_resource_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            model_dir = temp_path / "persisted-models"

            with mock.patch.dict(os.environ, state["env"], clear=False):
                exit_code, stdout, stderr = self.run_cli(["config", "set", "model_dir", str(model_dir)])

            config_text = state["config_path"].read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"model_dir = {model_dir}\n")
        self.assertEqual(stderr, "")
        self.assertIn(f'model_dir = "{model_dir.as_posix()}"', config_text)

    def test_config_set_runtime_preferences_persists_device_and_boolean_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = user_state_paths(Path(temp_dir))

            with mock.patch.dict(os.environ, state["env"], clear=False):
                first = self.run_cli(["config", "set", "default_device", "cuda:0"])
                second = self.run_cli(["config", "set", "use_fp16", "true"])
                third = self.run_cli(["config", "set", "use_deepspeed", "false"])
                fourth = self.run_cli(["config", "set", "use_cuda_kernel", "true"])

            config_text = state["config_path"].read_text(encoding="utf-8")

        self.assertEqual(first, (0, "default_device = cuda:0\n", ""))
        self.assertEqual(second, (0, "use_fp16 = true\n", ""))
        self.assertEqual(third, (0, "use_deepspeed = false\n", ""))
        self.assertEqual(fourth, (0, "use_cuda_kernel = true\n", ""))
        self.assertIn('default_device = "cuda:0"', config_text)
        self.assertIn("use_fp16 = true", config_text)
        self.assertIn("use_deepspeed = false", config_text)
        self.assertIn("use_cuda_kernel = true", config_text)

    def test_config_set_boolean_preference_rejects_non_boolean_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = user_state_paths(Path(temp_dir))

            with mock.patch.dict(os.environ, state["env"], clear=False):
                exit_code, stdout, stderr = self.run_cli(["config", "set", "use_fp16", "yes"])
                config_exists = state["config_path"].exists()

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: use_fp16 must be true or false", stderr)
        self.assertFalse(config_exists)

    def test_config_get_prints_the_current_persistent_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            model_dir = temp_path / "models"

            with mock.patch.dict(os.environ, state["env"], clear=False):
                self.run_cli(["config", "set", "model_dir", str(model_dir)])
                self.run_cli(["config", "set", "default_device", "cpu"])
                exit_code, stdout, stderr = self.run_cli(["config", "get"])

        self.assertEqual(exit_code, 0)
        self.assertIn(f'model_dir = "{model_dir.as_posix()}"', stdout)
        self.assertIn('default_device = "cpu"', stdout)
        self.assertEqual(stderr, "")

    def test_check_uses_persisted_model_dir_when_command_line_and_environment_do_not_override_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            model_dir = temp_path / "persisted-models"
            make_model_dir(model_dir)

            with mock.patch.dict(os.environ, state["env"], clear=False):
                self.run_cli(["config", "set", "model_dir", str(model_dir)])
                with patched_imports():
                    exit_code, stdout, stderr = self.run_cli(["check"])

        self.assertEqual(exit_code, 0)
        self.assertIn(f"OK: model directory {model_dir}", stdout)
        self.assertEqual(stderr, "")

    def test_check_model_dir_resolution_prioritizes_command_line_then_environment_then_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            cli_model_dir = temp_path / "cli-models"
            env_model_dir = temp_path / "env-models"
            persisted_model_dir = temp_path / "persisted-models"
            for model_dir in [cli_model_dir, env_model_dir, persisted_model_dir]:
                make_model_dir(model_dir)

            with mock.patch.dict(os.environ, state["env"], clear=False):
                self.run_cli(["config", "set", "model_dir", str(persisted_model_dir)])
                with patched_imports():
                    cli_result = self.run_cli(
                        ["check", "--model-dir", str(cli_model_dir)],
                    )
                with mock.patch.dict(os.environ, {"INDEXTTS2_MODEL_DIR": str(env_model_dir)}, clear=False):
                    with patched_imports():
                        env_result = self.run_cli(["check"])
                with patched_imports():
                    config_result = self.run_cli(["check"])

        self.assertIn(f"OK: model directory {cli_model_dir}", cli_result[1])
        self.assertIn(f"OK: model directory {env_model_dir}", env_result[1])
        self.assertIn(f"OK: model directory {persisted_model_dir}", config_result[1])

    def test_check_initializes_default_state_and_checks_the_platform_default_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = user_state_paths(Path(temp_dir))

            with mock.patch.dict(os.environ, state["env"], clear=False):
                with mock.patch.dict(os.environ, {"INDEXTTS2_MODEL_DIR": ""}, clear=False):
                    exit_code, stdout, stderr = self.run_cli(["check"])

            config_text = state["config_path"].read_text(encoding="utf-8")
            model_dir_exists = state["model_dir"].is_dir()

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: missing required model files", stderr)
        self.assertIn(state["model_dir"].as_posix(), config_text)
        self.assertTrue(model_dir_exists)

    def test_check_with_command_model_dir_still_initializes_default_state_without_persisting_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            cli_model_dir = temp_path / "cli-models"
            make_model_dir(cli_model_dir)

            with mock.patch.dict(os.environ, state["env"], clear=False):
                with patched_imports():
                    exit_code, stdout, stderr = self.run_cli(["check", "--model-dir", str(cli_model_dir)])

            config_text = state["config_path"].read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn(f"OK: model directory {cli_model_dir}", stdout)
        self.assertEqual(stderr, "")
        self.assertIn(f'model_dir = "{state["model_dir"].as_posix()}"', config_text)
        self.assertNotIn(cli_model_dir.as_posix(), config_text)

    def test_synth_uses_persisted_model_dir_and_runtime_preferences_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            model_dir = temp_path / "models"
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            calls = []
            make_model_dir(model_dir)
            voice_path.write_bytes(b"voice")

            class FakeIndexTTS2:
                def __init__(self, **kwargs):
                    calls.append(("init", kwargs))

                def infer(self, **kwargs):
                    calls.append(("infer", kwargs))

            with mock.patch.dict(os.environ, state["env"], clear=False):
                self.run_cli(["config", "set", "model_dir", str(model_dir)])
                self.run_cli(["config", "set", "default_device", "cpu"])
                self.run_cli(["config", "set", "use_fp16", "true"])
                self.run_cli(["config", "set", "use_deepspeed", "true"])
                self.run_cli(["config", "set", "use_cuda_kernel", "true"])
                exit_code, stdout, stderr = self.run_cli(
                    [
                        "synth",
                        "--text",
                        "hello",
                        "--voice",
                        str(voice_path),
                        "--output",
                        str(output_path),
                    ],
                    tts_factory=FakeIndexTTS2,
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
                "device": "cpu",
                "use_cuda_kernel": True,
                "use_deepspeed": True,
            },
        )

    def test_batch_uses_persisted_model_dir_and_runtime_preferences_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            model_dir = temp_path / "models"
            voice_path = temp_path / "voice.wav"
            batch_file = temp_path / "batch.jsonl"
            output_path = temp_path / "out.wav"
            calls = []
            make_model_dir(model_dir)
            voice_path.write_bytes(b"voice")
            batch_file.write_text('{"text": "hello", "voice": "voice.wav", "output": "out.wav"}\n', encoding="utf-8")

            class FakeIndexTTS2:
                def __init__(self, **kwargs):
                    calls.append(("init", kwargs))

                def infer(self, **kwargs):
                    calls.append(("infer", kwargs))
                    Path(kwargs["output_path"]).write_bytes(b"audio")

            with mock.patch.dict(os.environ, state["env"], clear=False):
                self.run_cli(["config", "set", "model_dir", str(model_dir)])
                self.run_cli(["config", "set", "default_device", "cpu"])
                self.run_cli(["config", "set", "use_fp16", "true"])
                self.run_cli(["config", "set", "use_deepspeed", "true"])
                self.run_cli(["config", "set", "use_cuda_kernel", "true"])
                exit_code, stdout, stderr = self.run_cli(
                    ["batch", "--batch-file", str(batch_file)],
                    tts_factory=FakeIndexTTS2,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\nBatch complete: 1 tasks generated\n")
        self.assertEqual(stderr, "")
        self.assertEqual(
            calls[0][1],
            {
                "cfg_path": str(model_dir / "config.yaml"),
                "model_dir": str(model_dir),
                "use_fp16": True,
                "device": "cpu",
                "use_cuda_kernel": True,
                "use_deepspeed": True,
            },
        )

    def test_batch_command_line_can_disable_persisted_boolean_runtime_preferences_for_one_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            model_dir = temp_path / "models"
            voice_path = temp_path / "voice.wav"
            batch_file = temp_path / "batch.jsonl"
            calls = []
            make_model_dir(model_dir)
            voice_path.write_bytes(b"voice")
            batch_file.write_text('{"text": "hello", "voice": "voice.wav", "output": "out.wav"}\n', encoding="utf-8")

            class FakeIndexTTS2:
                def __init__(self, **kwargs):
                    calls.append(("init", kwargs))

                def infer(self, **kwargs):
                    calls.append(("infer", kwargs))
                    Path(kwargs["output_path"]).write_bytes(b"audio")

            with mock.patch.dict(os.environ, state["env"], clear=False):
                self.run_cli(["config", "set", "model_dir", str(model_dir)])
                self.run_cli(["config", "set", "use_fp16", "true"])
                self.run_cli(["config", "set", "use_deepspeed", "true"])
                self.run_cli(["config", "set", "use_cuda_kernel", "true"])
                before_config = state["config_path"].read_text(encoding="utf-8")
                exit_code, stdout, stderr = self.run_cli(
                    [
                        "batch",
                        "--batch-file",
                        str(batch_file),
                        "--no-fp16",
                        "--no-deepspeed",
                        "--no-cuda-kernel",
                    ],
                    tts_factory=FakeIndexTTS2,
                )
                after_config = state["config_path"].read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("Batch complete: 1 tasks generated", stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(before_config, after_config)
        self.assertEqual(calls[0][1]["use_fp16"], False)
        self.assertEqual(calls[0][1]["use_deepspeed"], False)
        self.assertEqual(calls[0][1]["use_cuda_kernel"], False)

    def test_synth_command_line_overrides_do_not_rewrite_persistent_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            persisted_model_dir = temp_path / "persisted-models"
            cli_model_dir = temp_path / "cli-models"
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            calls = []
            make_model_dir(persisted_model_dir)
            make_model_dir(cli_model_dir)
            voice_path.write_bytes(b"voice")

            class FakeIndexTTS2:
                def __init__(self, **kwargs):
                    calls.append(("init", kwargs))

                def infer(self, **kwargs):
                    calls.append(("infer", kwargs))

            with mock.patch.dict(os.environ, state["env"], clear=False):
                self.run_cli(["config", "set", "model_dir", str(persisted_model_dir)])
                self.run_cli(["config", "set", "default_device", "cpu"])
                self.run_cli(["config", "set", "use_fp16", "false"])
                before_config = state["config_path"].read_text(encoding="utf-8")
                exit_code, stdout, stderr = self.run_cli(
                    [
                        "synth",
                        "--text",
                        "hello",
                        "--voice",
                        str(voice_path),
                        "--output",
                        str(output_path),
                        "--model-dir",
                        str(cli_model_dir),
                        "--device",
                        "cuda:0",
                        "--fp16",
                        "--deepspeed",
                        "--cuda-kernel",
                    ],
                    tts_factory=FakeIndexTTS2,
                )
                after_config = state["config_path"].read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(before_config, after_config)
        self.assertEqual(calls[0][1]["model_dir"], str(cli_model_dir))
        self.assertEqual(calls[0][1]["device"], "cuda:0")
        self.assertEqual(calls[0][1]["use_fp16"], True)
        self.assertEqual(calls[0][1]["use_deepspeed"], True)
        self.assertEqual(calls[0][1]["use_cuda_kernel"], True)

    def test_synth_command_line_can_disable_persisted_boolean_runtime_preferences_for_one_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state = user_state_paths(temp_path)
            model_dir = temp_path / "models"
            voice_path = temp_path / "voice.wav"
            output_path = temp_path / "out.wav"
            calls = []
            make_model_dir(model_dir)
            voice_path.write_bytes(b"voice")

            class FakeIndexTTS2:
                def __init__(self, **kwargs):
                    calls.append(("init", kwargs))

                def infer(self, **kwargs):
                    calls.append(("infer", kwargs))

            with mock.patch.dict(os.environ, state["env"], clear=False):
                self.run_cli(["config", "set", "model_dir", str(model_dir)])
                self.run_cli(["config", "set", "use_fp16", "true"])
                self.run_cli(["config", "set", "use_deepspeed", "true"])
                self.run_cli(["config", "set", "use_cuda_kernel", "true"])
                before_config = state["config_path"].read_text(encoding="utf-8")
                exit_code, stdout, stderr = self.run_cli(
                    [
                        "synth",
                        "--text",
                        "hello",
                        "--voice",
                        str(voice_path),
                        "--output",
                        str(output_path),
                        "--no-fp16",
                        "--no-deepspeed",
                        "--no-cuda-kernel",
                    ],
                    tts_factory=FakeIndexTTS2,
                )
                after_config = state["config_path"].read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, f"Generated: {output_path}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(before_config, after_config)
        self.assertEqual(calls[0][1]["use_fp16"], False)
        self.assertEqual(calls[0][1]["use_deepspeed"], False)
        self.assertEqual(calls[0][1]["use_cuda_kernel"], False)


if __name__ == "__main__":
    unittest.main()
