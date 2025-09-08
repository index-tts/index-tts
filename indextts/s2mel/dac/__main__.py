import sys

import argbind

from dac.utils import download
from dac.utils.decode import decode
from dac.utils.encode import encode

STAGES = ["encode", "decode", "download"]


def run(stage: str):
    """Run stages.

    Parameters
    ----------
    stage : str
        Stage to run
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown command: {stage}. Allowed commands are {STAGES}")
    stage_fn = globals()[stage]

    if stage == "download":
        stage_fn()
        return

    stage_fn()


if __name__ == "__main__":
    group = sys.argv.pop(1)
    args = argbind.parse_args(group=group)

    with argbind.scope(args):
        run(group)
