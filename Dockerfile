FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and essential packages
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt gradio==3.50.2 pydub

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e ".[webui]"

# Expose port for web UI
EXPOSE 7860

# Default command to run web UI
CMD ["python", "webui.py", "--host", "0.0.0.0", "--port", "7860"]
