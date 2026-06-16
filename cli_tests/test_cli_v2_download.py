import contextlib
import io
import os
import sys
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


def make_model_dir(path, include_aux=True):
    path.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_MODEL_FILES:
        (path / filename).write_text("placeholder", encoding="utf-8")
    for dirname in REQUIRED_MODEL_DIRS:
        (path / dirname).mkdir(exist_ok=True)
    if include_aux:
        make_aux_model_cache(path)


def make_aux_model_cache(path):
    for filename in AUX_MODEL_FILES:
        target = path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder", encoding="utf-8")
    for dirname in AUX_MODEL_DIRS:
        target = path / dirname
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("placeholder", encoding="utf-8")


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


class DownloadCommandTests(unittest.TestCase):
    def run_cli(self, args):
        from indextts.cli_v2 import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(args)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_download_defaults_to_huggingface_source_and_checks_downloaded_resources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = user_state_paths(Path(temp_dir).resolve())
            calls = []
            aux_calls = []

            def fake_snapshot_download(repo_id, local_dir, **kwargs):
                calls.append((repo_id, Path(local_dir)))
                make_model_dir(Path(local_dir), include_aux=False)
                return str(local_dir)

            def fake_ensure_models_available(model_dir):
                aux_calls.append(Path(model_dir))
                make_aux_model_cache(Path(model_dir))

            with mock.patch.dict(os.environ, state["env"], clear=False):
                with mock.patch("indextts.utils.model_download.snapshot_download", side_effect=fake_snapshot_download):
                    with mock.patch(
                        "indextts.utils.model_download.ensure_models_available",
                        side_effect=fake_ensure_models_available,
                    ):
                        exit_code, stdout, stderr = self.run_cli(["download"])
                config_exists = state["config_path"].exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [("IndexTeam/IndexTTS-2", state["model_dir"])])
        self.assertEqual(aux_calls, [state["model_dir"]])
        self.assertIn(f"Downloaded model resources to: {state['model_dir']}", stdout)
        self.assertEqual(stderr, "")
        self.assertFalse(config_exists)

    def test_download_from_modelscope_to_model_dir_persists_successful_target_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir).resolve()
            state = user_state_paths(temp_path)
            model_dir = temp_path / "custom-models"
            calls = []

            def fake_snapshot(model_id, local_dir, **kwargs):
                calls.append((model_id, Path(local_dir)))
                make_model_dir(Path(local_dir))
                return str(local_dir)

            with mock.patch.dict(os.environ, state["env"], clear=False):
                with mock.patch("indextts.utils.model_download._snapshot_from_modelscope", side_effect=fake_snapshot):
                    exit_code, stdout, stderr = self.run_cli(
                        ["download", "--source", "modelscope", "--model-dir", str(model_dir)]
                    )
                config_text = state["config_path"].read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [("IndexTeam/IndexTTS-2", model_dir)])
        self.assertIn(f"Downloaded model resources to: {model_dir}", stdout)
        self.assertEqual(stderr, "")
        self.assertIn(f'model_dir = "{model_dir.as_posix()}"', config_text)

    def test_download_from_huggingface_preserves_existing_files_in_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir).resolve()
            state = user_state_paths(temp_path)
            model_dir = temp_path / "models"
            sentinel = model_dir / "keep.txt"
            calls = []
            model_dir.mkdir()
            sentinel.write_text("keep", encoding="utf-8")

            def fake_snapshot_download(*, repo_id, local_dir):
                target = Path(local_dir)
                calls.append((repo_id, target, sentinel.exists()))
                make_model_dir(target)

            with mock.patch.dict(os.environ, state["env"], clear=False):
                with mock.patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download):
                    exit_code, stdout, stderr = self.run_cli(
                        ["download", "--source", "huggingface", "--model-dir", str(model_dir)]
                    )
                sentinel_text = sentinel.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [("IndexTeam/IndexTTS-2", model_dir, True)])
        self.assertEqual(sentinel_text, "keep")
        self.assertIn(f"Downloaded model resources to: {model_dir}", stdout)
        self.assertEqual(stderr, "")

    def test_download_with_no_save_does_not_persist_model_dir_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir).resolve()
            state = user_state_paths(temp_path)
            model_dir = temp_path / "temporary-models"

            def fake_snapshot_download(repo_id, local_dir, **kwargs):
                make_model_dir(Path(local_dir))
                return str(local_dir)

            with mock.patch.dict(os.environ, state["env"], clear=False):
                with mock.patch("indextts.utils.model_download.snapshot_download", side_effect=fake_snapshot_download):
                    exit_code, stdout, stderr = self.run_cli(
                        ["download", "--model-dir", str(model_dir), "--no-save"]
                    )
                config_exists = state["config_path"].exists()

        self.assertEqual(exit_code, 0)
        self.assertIn(f"Downloaded model resources to: {model_dir}", stdout)
        self.assertEqual(stderr, "")
        self.assertFalse(config_exists)

    def test_download_returns_runtime_unavailable_when_source_package_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = user_state_paths(Path(temp_dir).resolve())

            def raise_import(*args, **kwargs):
                raise ImportError("No module named huggingface_hub")

            with mock.patch.dict(os.environ, state["env"], clear=False):
                with mock.patch("indextts.utils.model_download.snapshot_download", side_effect=raise_import):
                    exit_code, stdout, stderr = self.run_cli(["download"])
                config_exists = state["config_path"].exists()

        self.assertEqual(exit_code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: runtime unavailable for auto download source", stderr)
        self.assertIn("pip install huggingface_hub modelscope", stderr)
        self.assertFalse(config_exists)

    def test_download_from_modelscope_returns_runtime_unavailable_when_source_package_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = user_state_paths(Path(temp_dir).resolve())

            def raise_import(*args, **kwargs):
                raise ImportError("No module named modelscope")

            with mock.patch.dict(os.environ, state["env"], clear=False):
                with mock.patch("indextts.utils.model_download._snapshot_from_modelscope", side_effect=raise_import):
                    exit_code, stdout, stderr = self.run_cli(["download", "--source", "modelscope"])
                config_exists = state["config_path"].exists()

        self.assertEqual(exit_code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: runtime unavailable for modelscope download source", stderr)
        self.assertIn("modelscope", stderr)
        self.assertFalse(config_exists)

    def test_download_validates_downloaded_resources_before_persisting_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir).resolve()
            state = user_state_paths(temp_path)
            model_dir = temp_path / "incomplete-models"

            def fake_snapshot_download(repo_id, local_dir, **kwargs):
                target = Path(local_dir)
                target.mkdir(parents=True, exist_ok=True)
                (target / "config.yaml").write_text("placeholder", encoding="utf-8")
                return str(local_dir)

            with mock.patch.dict(os.environ, state["env"], clear=False):
                with mock.patch("indextts.utils.model_download.snapshot_download", side_effect=fake_snapshot_download):
                    exit_code, stdout, stderr = self.run_cli(["download", "--model-dir", str(model_dir)])
                config_exists = state["config_path"].exists()

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR: missing required model files", stderr)
        self.assertIn("bpe.model", stderr)
        self.assertIn("qwen0.6bemo4-merge", stderr)
        self.assertIn(f"Model directory: {model_dir}", stderr)
        self.assertIn("Missing resources:", stderr)
        self.assertIn("rerun", stderr)
        self.assertFalse(config_exists)


if __name__ == "__main__":
    unittest.main()
