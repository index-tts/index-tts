# Build via NVIDIA's CUDA Toolkit container.
# NOTE: Using their development container, since we need `nvcc`.
# SEE: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda
FROM docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

# Install the `uv` Python package manager.
# NOTE: `uv` automatically installs Python versions on-demand.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install extra utilities:
# - The `ffmpeg` binary is required by the `ffmpeg-python` package.
# - `git`, `git-lfs` and `curl` enables pulling code inside the container.
# - `ssh` is sometimes required for Git LFS file transfers.
# NOTE: We install certain packages without "recommends" to reduce the size.
RUN apt-get -qq update && apt-get -qq install -y --no-install-recommends \
    ffmpeg \
    curl \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get -qq update && apt-get -qq install -y \
    git \
    git-lfs \
    openssh-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire scripts directory from the build context into the image.
# NOTE: The source path is relative to the Containerfile's location.
COPY scripts /opt/scripts

# Create `/app`, and mark all entrypoint scripts as executable.
RUN mkdir -p /app \
    && find /opt/scripts -name "*.sh" -exec chmod +x {} \;

# Set up the entrypoint.
WORKDIR /app

ENTRYPOINT ["/opt/scripts/entrypoint.sh"]

# Set the default command if none is provided.
# NOTE: This will be passed to ENTRYPOINT.
CMD ["webui"]
