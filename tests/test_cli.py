import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from indextts import cli
from indextts.utils import model_download


class CliConfigTest(unittest.TestCase):
    def test_missing_nonstandard_config_falls_back_to_downloaded_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            voice_path = os.path.join(tmp, "voice.wav")
            output_path = os.path.join(tmp, "out.wav")
            with open(voice_path, "w") as f:
                f.write("voice")

            def fake_download(_repo_id, _filename, local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "w") as f:
                    f.write("config")

            class FakeTTS:
                cfg_path = None

                def __init__(self, cfg_path, **_kwargs):
                    FakeTTS.cfg_path = cfg_path

                def infer(self, **_kwargs):
                    pass

            fake_torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: False),
                xpu=types.SimpleNamespace(is_available=lambda: False),
                mps=types.SimpleNamespace(is_available=lambda: False),
            )
            fake_infer = types.SimpleNamespace(IndexTTS=FakeTTS)
            argv = [
                "indextts",
                "hello",
                "-v",
                voice_path,
                "-o",
                output_path,
                "-c",
                "my.yaml",
            ]

            previous_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(model_download, "_download_single_file", fake_download), \
                        mock.patch.dict(sys.modules, {"torch": fake_torch, "indextts.infer": fake_infer}):
                    cli.main()
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(FakeTTS.cfg_path, os.path.join(".", "config.yaml"))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "config.yaml")))


if __name__ == "__main__":
    unittest.main()
