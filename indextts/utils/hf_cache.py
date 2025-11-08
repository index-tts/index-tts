"""Helper to configure a project-local Hugging Face Hub cache.
If HF_HUB_CACHE is not set in the environment, this module sets it to the
`/checkpoints/hf_cache` of the repo directory.
"""

import os


def set_hf_env() -> None:
    if "HF_HUB_CACHE" not in os.environ:
        CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        HF_HUB_CACHE_DIR = os.path.abspath(
            os.path.join(CURRENT_FILE_DIR, "..", "checkpoints", "hf_cache")
        )
        os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE_DIR
