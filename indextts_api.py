# Standard library imports
import argparse
import base64
import importlib.util
import io
import logging
import os
import sys
import uuid
from typing import Any
import asyncio

# Third-party imports
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# IndexTTS import (make sure the package is discoverable)
# If the repository layout differs, adjust the path accordingly before import.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  # Allow relative import of sibling packages

import torch
from indextts.infer import IndexTTS

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

tts_model = None
current_model_dir = None

# -----------------------------------------------------------------------------
# Concurrency control – limit simultaneous inference to prevent GPU contention
# -----------------------------------------------------------------------------

# Maximum number of concurrent TTS inferences allowed. Increase with caution –
# each inference can be very memory-intensive and running several in parallel
# may exhaust GPU/CPU/RAM resources. A value of 1 is the safest default.
MAX_CONCURRENT_REQUESTS = 1

# Async semaphore that will be awaited before each inference.  We create it at
# module import time so it is shared across all requests and all workers inside
# the same Python process.
model_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Configure logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger("indextts_api")


logger = setup_logging()  # Will be updated with verbose setting in main()

app = FastAPI()

# Add CORS middleware to allow requests from the same origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, will be restricted by host binding
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    voice: str = Form("voice_a"),
    speed: float = Form(1.0),
    model: str = Form("checkpoints"),
) :

    """
    POST an x-www-form-urlencoded form with 'text' (and optional 'voice', 'speed', and 'model').
    voice: voice_a, voice_b, voice_c,  with reference audio files loaded from voice/voice_a.wav, voice/voice_b.wav, voice/voice_c.wav
    We run TTS on the text and return JSON with:
    - audio_base64: Base64 encoded WAV audio data for direct playback
    - filename: Filename for backward compatibility
    - format: Audio format (wav)
    - sample_rate: Audio sample rate (24000)
    - duration: Audio duration in seconds
    """

    global tts_model
    global current_model_dir
    global model_semaphore

    # Validate and sanitise inputs -------------------------------------------------
    if not text or not text.strip():
        return JSONResponse({"error": "Text is empty"}, status_code=400)

    # Voice mapping – extend this dict with more voices/prompts as needed.
    voice_map = {
        "voice_a": os.path.join("voice", "voice_a.wav"),
        "voice_b": os.path.join("voice", "voice_b.wav"),
        "voice_c": os.path.join("voice", "voice_c.wav"),
        "voice_d": os.path.join("voice", "voice_d.wav"),
    }

    if voice not in voice_map:
        return JSONResponse(
            {"error": f"Invalid voice. Must be one of: {', '.join(voice_map.keys())}"},
            status_code=400,
        )

    voice_path = voice_map[voice]
    if not os.path.exists(voice_path):
        return JSONResponse({"error": f"Voice reference file not found: {voice_path}"}, status_code=404)

    # Ensure output directory exists ------------------------------------------------
    OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), ".indextts", "outputs")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load / (re)use the model ------------------------------------------------------
    try:
        if "config.yaml" in model or model.endswith(".yaml"):
            # Allow passing explicit config path – infer model_dir
            cfg_path = model
            model_dir = os.path.dirname(cfg_path)
        else:
            cfg_path = os.path.join(model, "config.yaml")
            model_dir = model

        if not os.path.exists(cfg_path):
            return JSONResponse({"error": f"Config not found at {cfg_path}"}, status_code=404)

        if tts_model is None or current_model_dir != model_dir:
            logger.info(f"Loading IndexTTS model from {model_dir}")
            tts_model = IndexTTS(cfg_path=cfg_path, model_dir=model_dir, is_fp16=True, device=device)
            current_model_dir = model_dir
            logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load IndexTTS model")
        return JSONResponse({"error": f"Failed to load model: {str(e)}"}, status_code=500)

    # Generate unique filename for output ------------------------------------------
    unique_id = str(uuid.uuid4())
    filename = f"tts_{unique_id}.wav"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    # Run inference -----------------------------------------------------------------
    try:
        logger.info(
            f"Received TTS Request – text len={len(text)}, voice={voice}, model_dir={model_dir}"
        )

        # ------------------------------------------------------------------
        # Acquire semaphore and run the blocking inference in a thread pool so
        # that we don't block the event-loop.  The semaphore guarantees that
        # at most MAX_CONCURRENT_REQUESTS inferences run at the same time.
        # ------------------------------------------------------------------

        async with model_semaphore:
            logger.info(f"Acquired semaphore for inference")
            logger.info(f"Generating TTS – text len={len(text)}, voice={voice}, model_dir={model_dir} ...")
            loop = asyncio.get_running_loop()
            from typing import cast
            model_ref: IndexTTS = cast(IndexTTS, tts_model)
            await loop.run_in_executor(
                None,
                lambda: model_ref.infer(  # type: ignore[attr-defined]
                    audio_prompt=voice_path,
                    text=text,
                    output_path=output_path,
                    verbose=False,
                ),
            )

    except Exception as e:
        logger.exception("Inference failed")
        return JSONResponse({"error": f"Inference failed: {str(e)}"}, status_code=500)

    # Encode audio to base64 --------------------------------------------------------
    try:
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        duration_seconds = len(audio_bytes) / (2 * 24000)  # int16 -> 2 bytes per sample, mono
    except Exception as e:
        logger.exception("Failed to read generated audio")
        return JSONResponse({"error": f"Failed to read generated audio: {str(e)}"}, status_code=500)

    return {
        "audio_base64": audio_base64,
        "filename": filename,
        "format": "wav",
        "sample_rate": 24000,
        "duration": duration_seconds,
    }


# ----------------------------------------------------------------------------------
# Global objects & CLI helper
# ----------------------------------------------------------------------------------

# These are defined *after* the endpoint so they appear as globals in the function.
tts_model: IndexTTS | None = None  # Loaded lazily
current_model_dir: str | None = None


def main():
    """
    
    Start the IndexTTS FastAPI server.

    curl -X POST http://localhost:9001/tts \
        -H "Content-Type: application/x-www-form-urlencoded" \
        --data-urlencode "text=Hello world" \
        --data-urlencode "voice=voice_a"

    http --form POST localhost:9001/tts text="Starting IndexTTS API server on localhost" voice=voice_a
    
    """
    parser = argparse.ArgumentParser(description="Start the IndexTTS API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Re-configure logging if verbose requested
    global logger
    logger = setup_logging(args.verbose)

    logger.info(
        f"Starting IndexTTS API server on {args.host}:{args.port} (verbose={args.verbose})"
    )

    uvicorn.run(
        "indextts_api:app",
        host=args.host,
        port=args.port,
        log_level="debug" if args.verbose else "info",
        reload=False,
    )


if __name__ == "__main__":
    main()
