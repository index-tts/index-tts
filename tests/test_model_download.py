import os
import tempfile
import unittest
from unittest import mock

from indextts.utils import model_download


def _add_hf_snapshot(cache_dir, repo_id, files, commit_hash="abc123", write_ref=True):
    repo_key = repo_id.replace("/", "--")
    repo_dir = os.path.join(cache_dir, f"models--{repo_key}")
    snapshot_dir = os.path.join(repo_dir, "snapshots", commit_hash)

    if write_ref:
        refs_dir = os.path.join(repo_dir, "refs")
        os.makedirs(refs_dir, exist_ok=True)
        with open(os.path.join(refs_dir, "main"), "w") as f:
            f.write(commit_hash)

    for rel_path, content in files.items():
        path = os.path.join(snapshot_dir, *rel_path.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    return snapshot_dir


class ModelDownloadCacheMigrationTest(unittest.TestCase):
    def test_migrates_existing_auxiliary_models_without_downloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "checkpoints")
            local_cache = os.path.join(model_dir, "hf_cache")
            default_cache = os.path.join(tmp, "huggingface", "hub")

            _add_hf_snapshot(local_cache, "facebook/w2v-bert-2.0", {"config.json": "w2v"})
            _add_hf_snapshot(
                default_cache,
                "amphion/MaskGCT",
                {"semantic_codec/model.safetensors": "maskgct"},
            )
            _add_hf_snapshot(default_cache, "funasr/campplus", {"campplus_cn_common.bin": "campplus"})
            _add_hf_snapshot(
                default_cache,
                "nvidia/bigvgan_v2_22khz_80band_256x",
                {
                    "config.json": "bigvgan-config",
                    "bigvgan_generator.pt": "bigvgan-weights",
                },
            )

            def fail_download(*_args, **_kwargs):
                raise AssertionError("download should not be called")

            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": default_cache}), \
                    mock.patch.object(model_download, "snapshot_download", fail_download), \
                    mock.patch.object(model_download, "_download_single_file", fail_download):
                paths = model_download.ensure_models_available(model_dir)

            with open(os.path.join(paths["w2v_bert"], "config.json")) as f:
                self.assertEqual(f.read(), "w2v")
            with open(paths["semantic_codec"]) as f:
                self.assertEqual(f.read(), "maskgct")
            with open(paths["campplus"]) as f:
                self.assertEqual(f.read(), "campplus")
            with open(os.path.join(paths["bigvgan"], "config.json")) as f:
                self.assertEqual(f.read(), "bigvgan-config")
            with open(os.path.join(paths["bigvgan"], "bigvgan_generator.pt")) as f:
                self.assertEqual(f.read(), "bigvgan-weights")

    def test_ambiguous_snapshot_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "hf_cache")
            _add_hf_snapshot(cache_dir, "facebook/w2v-bert-2.0", {"config.json": "old"}, "old", write_ref=False)
            _add_hf_snapshot(cache_dir, "facebook/w2v-bert-2.0", {"config.json": "new"}, "new", write_ref=False)

            self.assertIsNone(model_download._find_hf_cache_snapshot(cache_dir, "facebook/w2v-bert-2.0"))

    def test_missing_config_downloads_only_config_file(self):
        with tempfile.TemporaryDirectory() as model_dir:
            calls = []

            def fake_download(repo_id, filename, local_path):
                calls.append((repo_id, filename, local_path))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "w") as f:
                    f.write("config")

            with mock.patch.object(model_download, "_download_single_file", fake_download):
                model_download.ensure_config_available(model_dir)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "IndexTeam/IndexTTS-2")
            self.assertEqual(calls[0][1], "config.yaml")
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "config.yaml")))


if __name__ == "__main__":
    unittest.main()
