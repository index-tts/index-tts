import contextlib
import importlib
import io
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
]


def make_model_dir(base_dir):
    model_dir = base_dir / "checkpoints"
    model_dir.mkdir()
    for filename in REQUIRED_MODEL_FILES:
        (model_dir / filename).write_text("placeholder", encoding="utf-8")
    return model_dir


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


if __name__ == "__main__":
    unittest.main()
