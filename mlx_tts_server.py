import argparse
import asyncio
import base64
import gc
import importlib.util
import io
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

"""
Fixed version with proper async handling and resource management for MLX TTS

Start the service with:
python -m techdaily.api.mlx_tts_server --host 0.0.0.0 --port 9000

```
mlx==0.26.1
mlx-audio==0.2.3
mlx-lm==0.24.1
#mlx-lm==0.25.2
soundfile==0.13.1
misaki[zh]==0.9.4
```

PYTHONPATH=. uv run -m techdaily.api.mlx_tts_server --host 0.0.0.0 --port 9000

from mlx_lm.utils import dequantize_model, quantize_model, save_config, save_weights
ImportError: cannot import name 'save_weights' from 'mlx_lm.utils' 

mlx-audio 0.2.3 works with mlx-lm 0.24.1,
mlx-audio 0.2.4 works with mlx-lm 0.25.2.
"""

# Validate model parameter
valid_models = [
        "mlx-community/IndexTTS-1.5",
        "mlx-community/Kokoro-82M-4bit",
        "mlx-community/Kokoro-82M-6bit",
        "mlx-community/Kokoro-82M-8bit",
        "mlx-community/Kokoro-82M-bf16",
    ]

DEFAULT_MODEL = "mlx-community/IndexTTS-1.5"

# Configure logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if verbose:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger("mlx_audio_server")

logger = setup_logging()

# Import from mlx_audio package
from mlx_audio.tts.utils import load_model

# Global model and async semaphore for safe concurrent access
tts_model = None
# Use async semaphore instead of threading.Lock for FastAPI
MAX_CONCURRENT_REQUESTS = 1  # Limit to 1 to prevent GPU conflicts
model_semaphore = None
request_counter = 0

# Request queue management
request_queue = asyncio.Queue()
queue_processing_task = None
DEBOUNCE_TIME = 1  # 1s debounce time for incoming requests
MAX_QUEUE_SIZE = 10  # Maximum number of requests in queue
last_request_time = {}

# Output folder setup
OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), ".mlx_audio", "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logger.debug(f"Using output folder: {OUTPUT_FOLDER}")

def cleanup_mlx_memory():
    """Cleanup MLX memory and resources"""
    try:
        import mlx.core as mx
        # Force evaluation of any pending operations
        mx.eval([])
        # Clear metal cache if available
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"MLX cleanup warning: {e}")

def cleanup_audio_arrays(audio_arrays):
    """Explicitly cleanup numpy/MLX arrays"""
    try:
        for arr in audio_arrays:
            del arr
        del audio_arrays
    except Exception as e:
        logger.debug(f"Array cleanup warning: {e}")
    finally:
        gc.collect()
        cleanup_mlx_memory()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    global tts_model, model_semaphore
    
    # Startup
    logger.info("Starting MLX TTS server...")
    
    # Initialize semaphore
    model_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    try:
        await setup_server()
        logger.info("TTS model loaded successfully, server ready")
        yield
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down MLX TTS server...")
        await cleanup_server()
        logger.info("Server shutdown complete")

# Create FastAPI app with lifespan context
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/tts")
async def tts_endpoint(
    request: Request,
    text: str = Form(...),
    voice: str = Form("af_heart"),
    speed: float = Form(1.0),
    client_id: Optional[str] = Form(None),  # 添加客户端ID参数用于防止重复请求
):
    """
    Async TTS endpoint with proper resource management
    """
    global tts_model, request_counter, model_semaphore
    
    # Increment request counter for monitoring
    request_counter += 1
    current_request = request_counter
    
    # 获取客户端标识符，如果没有提供client_id则使用IP地址
    effective_client_id = client_id
    if not effective_client_id:
        # 获取客户端IP地址
        client_host = request.client.host if request.client else "unknown"
        effective_client_id = f"ip_{client_host}"
        logger.info(f"Request #{current_request}: No client_id provided, using IP address: {client_host}")
    
    # 实现客户端请求节流
    current_time = asyncio.get_event_loop().time()
    if effective_client_id in last_request_time:
        time_since_last = current_time - last_request_time[effective_client_id]
        if time_since_last < DEBOUNCE_TIME:
            logger.info(f"Request #{current_request}: Debounced request from client {effective_client_id}, ignoring")
            return JSONResponse({"status": "debounced", "message": "Request too frequent"}, status_code=429)
    
    # 更新最后请求时间
    last_request_time[effective_client_id] = current_time
    
    # 检查队列大小
    if request_queue.qsize() >= MAX_QUEUE_SIZE:
        logger.warning(f"Request #{current_request}: Queue full, rejecting request")
        return JSONResponse({"error": "Server busy, try again later"}, status_code=503)
    
    logger.info(f"Request #{current_request}: Starting TTS generation")

    if not text.strip():
        return JSONResponse({"error": "Text is empty"}, status_code=400)

    # Validate speed parameter
    try:
        speed_float = float(speed)
        if speed_float < 0.5 or speed_float > 2.0:
            return JSONResponse(
                {"error": "Speed must be between 0.5 and 2.0"}, status_code=400
            )
    except ValueError:
        return JSONResponse({"error": "Invalid speed value"}, status_code=400)

    # Ensure the global model is loaded
    if tts_model is None:
        logger.error("TTS model is not loaded.")
        return JSONResponse(
            {"error": "TTS model not available"}, status_code=500
        )

    # Determine voice based on the model type
    effective_voice = voice
    if DEFAULT_MODEL and "IndexTTS" in DEFAULT_MODEL and "voice" not in voice:
        voice_mapping = {
            "af_heart": "voice_a",
            "af_sarah": "voice_c", 
            "af_sky": "voice_d"
        }
        effective_voice = voice_mapping.get(voice, "voice_b")

    ref_audio = None

    # Load reference audio if needed
    if "voice" in effective_voice:
        audio_path = f"voice/{effective_voice}.wav"
        try:
            ref_audio_data, _ = sf.read(audio_path)
            ref_audio = ref_audio_data
        except FileNotFoundError:
            logger.warning(f"Reference audio file {audio_path} not found")
            ref_audio = None

    logger.info(
        f"Request #{current_request}: Generating TTS for text: '{text[:50]}...' "
        f"with voice: {effective_voice}, speed: {speed_float}, model: {DEFAULT_MODEL}"
    )

    # 创建请求任务
    request_data = {
        "id": current_request,
        "text": text,
        "voice": effective_voice,
        "ref_audio": ref_audio,
        "speed": speed_float,
        "client_id": effective_client_id
    }
    
    # 将请求放入队列
    try:
        # 使用队列管理请求，而不是直接处理
        future = asyncio.Future()
        await request_queue.put((request_data, future))
        
        # 确保队列处理任务正在运行
        global queue_processing_task
        if queue_processing_task is None or queue_processing_task.done():
            queue_processing_task = asyncio.create_task(process_tts_queue())
            
        # 等待请求处理完成
        try:
            results = await asyncio.wait_for(future, timeout=300.0)  # 设置超时时间为300秒
        except asyncio.TimeoutError:
            logger.error(f"Request #{current_request}: Processing timed out")
            return JSONResponse({"error": "Processing timed out"}, status_code=504)

        # Collect audio segments and convert MLX arrays to numpy
        audio_arrays = []
        try:
            for segment in results:
                # Convert MLX array to numpy array
                if hasattr(segment.audio, '__array__'):
                    # MLX arrays can be converted using np.array()
                    audio_data = np.array(segment.audio)
                else:
                    # Fallback if it's already a numpy array
                    audio_data = segment.audio
                audio_arrays.append(audio_data)
            
            logger.debug(f"Request #{current_request}: Collected {len(audio_arrays)} audio segments")
        
        except Exception as e:
            logger.error(f"Request #{current_request}: Error processing audio segments: {e}")
            # Clean up any partial results
            cleanup_audio_arrays(audio_arrays if 'audio_arrays' in locals() else [])
            return JSONResponse(
                {"error": f"Failed to process audio segments: {str(e)}"}, status_code=500
            )

        # Check if any audio was generated
        if not audio_arrays:
            logger.error(f"Request #{current_request}: No audio segments generated")
            return JSONResponse({"error": "No audio generated"}, status_code=500)

        # Concatenate all segments outside of the semaphore
        concatenated_audio = None
        try:
            concatenated_audio = np.concatenate(audio_arrays, axis=0)
            logger.debug(f"Request #{current_request}: Concatenated audio shape: {concatenated_audio.shape}")
            
            # Clean up the individual arrays immediately
            cleanup_audio_arrays(audio_arrays)
            
        except Exception as e:
            cleanup_audio_arrays(audio_arrays)
            logger.error(f"Request #{current_request}: Error concatenating audio: {e}")
            return JSONResponse(
                {"error": f"Failed to concatenate audio: {str(e)}"}, status_code=500
            )

        # Convert audio to base64 encoded WAV format
        audio_buffer = None
        try:
            # Create an in-memory buffer for the WAV file
            audio_buffer = io.BytesIO()
            
            # Write the audio data to the buffer as WAV format
            sf.write(audio_buffer, concatenated_audio, 24000, format='WAV')
            
            # Get the audio data as bytes
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.getvalue()
            
            # Encode to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Calculate duration
            duration = len(concatenated_audio) / 24000
            
            logger.info(f"Request #{current_request}: Successfully generated audio "
                       f"with {len(audio_bytes)} bytes, duration: {duration:.2f}s")

        except Exception as e:
            logger.error(f"Request #{current_request}: Error generating audio: {str(e)}")
            return JSONResponse(
                {"error": f"Failed to generate audio: {str(e)}"}, status_code=500
            )
        finally:
            # Clean up resources
            if audio_buffer:
                audio_buffer.close()
            if concatenated_audio is not None:
                del concatenated_audio
            gc.collect()
            cleanup_mlx_memory()

        # Generate unique filename for backward compatibility
        unique_id = str(uuid.uuid4())
        filename = f"tts_{unique_id}.wav"

        logger.info(f"Request #{current_request}: TTS generation completed successfully")

        return {
            "audio_base64": audio_base64,
            "filename": filename,
            "format": "wav",
            "sample_rate": 24000,
            "duration": duration
        }

    except Exception as e:
        logger.error(f"Request #{current_request}: Unexpected error: {str(e)}")
        # Ensure cleanup on any error
        cleanup_mlx_memory()
        gc.collect()
        return JSONResponse(
            {"error": f"TTS generation failed: {str(e)}"}, status_code=500
        )

async def process_tts_queue():
    """处理TTS请求队列的异步任务"""
    global tts_model, model_semaphore
    
    logger.info("Starting TTS queue processor")
    
    while True:
        try:
            # 从队列获取请求
            request_data, future = await request_queue.get()
            current_request = request_data["id"]
            
            logger.debug(f"Request #{current_request}: Processing from queue, queue size: {request_queue.qsize()}")
            
            try:
                # 使用信号量限制并发访问
                async with model_semaphore:
                    logger.debug(f"Request #{current_request}: Acquired model semaphore")
                    
                    # 在线程池中运行同步模型生成，避免阻塞事件循环
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        lambda: tts_model.generate(
                            text=request_data["text"],
                            voice=request_data["voice"],
                            ref_audio=request_data["ref_audio"],
                            speed=request_data["speed"],
                            lang_code='z',
                            verbose=False,
                        )
                    )
                    
                    # 设置结果
                    if not future.done():
                        future.set_result(results)
                    
            except Exception as e:
                logger.error(f"Request #{current_request}: Error processing request: {str(e)}")
                if not future.done():
                    future.set_exception(e)
            finally:
                # 标记任务完成
                request_queue.task_done()
                
        except asyncio.CancelledError:
            logger.info("TTS queue processor cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in queue processor: {str(e)}")
            # 继续处理队列，不要因为一个请求的错误而停止整个队列
            await asyncio.sleep(1)

async def setup_server():
    """Setup the server by loading the model and creating the output directory."""
    global tts_model, OUTPUT_FOLDER

    # Setup output directory with fallback
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(OUTPUT_FOLDER, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("Test write permissions")
        os.remove(test_file)
        logger.debug(f"Output directory {OUTPUT_FOLDER} is writable")
    except Exception as e:
        logger.error(f"Error with output directory {OUTPUT_FOLDER}: {str(e)}")
        fallback_dir = os.path.join("/tmp", "mlx_audio_outputs")
        logger.debug(f"Trying fallback directory: {fallback_dir}")
        try:
            os.makedirs(fallback_dir, exist_ok=True)
            OUTPUT_FOLDER = fallback_dir
            logger.debug(f"Using fallback output directory: {OUTPUT_FOLDER}")
        except Exception as fallback_error:
            logger.error(f"Error with fallback directory: {str(fallback_error)}")

    # Load the model
    if tts_model is None:
        try:
            logger.info(f"Loading TTS model from {DEFAULT_MODEL}")
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            tts_model = await loop.run_in_executor(None, load_model, DEFAULT_MODEL)
            
            logger.info("TTS model loaded successfully")
            
            # Force initial GPU memory allocation to avoid issues later
            try:
                test_results = await loop.run_in_executor(
                    None,
                    lambda: tts_model.generate(
                        text="Test initialization",
                        voice="voice_a",
                        speed=1.0,
                        lang_code='z',
                        verbose=False,
                    )
                )
                # Clean up test results - handle MLX arrays properly
                for segment in test_results:
                    del segment
                del test_results
                gc.collect()
                cleanup_mlx_memory()
                    
                logger.info("Model initialization test completed")
            except Exception as test_error:
                logger.warning(f"Model initialization test failed: {test_error}")
                
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            raise

async def cleanup_server():
    """Cleanup resources on server shutdown"""
    global tts_model, queue_processing_task
    logger.info("Cleaning up server resources...")
    
    # 取消队列处理任务
    if queue_processing_task is not None and not queue_processing_task.done():
        queue_processing_task.cancel()
        try:
            await queue_processing_task
        except asyncio.CancelledError:
            pass
    
    # 清空请求队列
    while not request_queue.empty():
        try:
            request_data, future = request_queue.get_nowait()
            if not future.done():
                future.set_exception(Exception("Server shutting down"))
            request_queue.task_done()
        except asyncio.QueueEmpty:
            break
    
    if tts_model is not None:
        # Clear the model reference
        tts_model = None
    
    # Force garbage collection and MLX cleanup
    gc.collect()
    cleanup_mlx_memory()
    
    logger.info("Server resource cleanup completed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "requests_processed": request_counter,
        "queue_size": request_queue.qsize() if request_queue else 0,
        "queue_active": queue_processing_task is not None and not queue_processing_task.done() if queue_processing_task else False
    }

def main():
    """Parse command line arguments for the server and start it."""
    parser = argparse.ArgumentParser(description="Start the MLX-Audio TTS server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port to bind the server to (default: 9000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with detailed debug information",
    )
    args = parser.parse_args()

    # Update logger with verbose setting
    global logger
    logger = setup_logging(args.verbose)

    # Configure uvicorn to minimize resource leaks
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.verbose else "info",
        workers=1,  # Keep single worker to avoid model loading issues
        loop="asyncio",
        # Additional configurations to help with resource management
        access_log=False,  # Reduce logging overhead
        server_header=False,  # Reduce response overhead
    )

if __name__ == "__main__":
    main()