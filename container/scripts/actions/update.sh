#!/usr/bin/env bash

set -eu


# Ensure that we are in the correct directory.
cd /app


# Enable "LFS" smudge and clean filters in the global Git config (`~/.gitconfig`).
# NOTE: This simplifies Git workflows by automatically using LFS during various
# Git commands, such as when checking out other branches that contain LFS files.
# NOTE: We do this every time because we can't be sure that it's been pre-applied.
git lfs install


# Download the latest upstream commits.
# NOTE: If the LFS filters are working correctly, it also downloads the LFS files.
# NOTE: It's safe to update inside the container, and it will still maintain full
# compatibility with the host, since Git repos are completely portable.
git pull


# Download the actual large files that the LFS pointers are referencing.
# NOTE: This is extra verification that all LFS files have truly been downloaded.
git lfs pull
