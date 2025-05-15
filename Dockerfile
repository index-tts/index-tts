ARG BASE_IMAGE=continuumio/miniconda3
FROM ${BASE_IMAGE}
# build args
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.3.1+cu121
ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/torch_stable.html"
ARG PYPI_INDEX_URL="https://pypi.org/simple"
ARG OVERRIDE_CONDA_CHANNEL=0
# Optional packages to install, e.g. "ninja deepspeed"
ARG PIP_EXTRA_PACKAGES=""
ARG GIT_REPO="https://github.com/index-tts/index-tts"
ARG GIT_BRANCH="main"
LABEL github_repo=$GIT_REPO
LABEL maintainer="Index-tts Team"

SHELL ["/bin/bash","-l", "-c"]

# Conda environment setup
RUN test "$OVERRIDE_CONDA_CHANNEL" -ne 1 || \
    (echo "channels:" > ~/.condarc && \
     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" >> ~/.condarc && \
     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r" >> ~/.condarc && \
     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2" >> ~/.condarc && \
     echo "show_channel_urls: true" >> ~/.condarc && \
     echo "custom_channels:" >> ~/.condarc && \
     echo "  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
     echo "  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc)
RUN conda create -n index-tts python=${PYTHON_VERSION} && \
    echo "conda activate index-tts" >> ~/.bashrc

ENV PATH="/opt/conda/envs/index-tts/bin:$PATH"

RUN if [ -n "$PYPI_INDEX_URL" ]; then \
        conda run -n index-tts pip config set global.index-url ${PYPI_INDEX_URL}; \
    fi

# Install pytorch 
RUN conda run -n index-tts pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} torchaudio==${PYTORCH_VERSION} -f ${PYTORCH_INDEX_URL}

WORKDIR /app

# Clone the repository
RUN git clone --depth 1 --branch ${GIT_BRANCH} ${GIT_REPO} /app

# Install dependencies 
RUN if [ "$TARGETPLATFORM" -eq "linux/amd64" ]; then \
        conda install -n index-tts -y -c conda-forge pynini && \
        conda run -n index-tts pip install --no-cache-dir WeTextProcessing --no-deps; \
    fi

RUN if [ -n "$PIP_EXTRA_PACKAGES" ]; then \
        conda run -n index-tts pip install --no-cache-dir ${PIP_EXTRA_PACKAGES}; \
    fi

RUN conda run -n index-tts pip install --no-cache-dir -e ".[webui]"
# Clean up
RUN conda clean --all -y

EXPOSE 7860

VOLUME ["/app/checkpoints", "/root/.cache"] 

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "index-tts", "python", "webui.py"]
