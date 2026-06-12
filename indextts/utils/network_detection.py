"""
Network detection utility for determining whether the current network
environment needs a proxy to access HuggingFace, to decide whether to
use ModelScope for model downloads.
"""

import os
import socket
import time
import logging

logger = logging.getLogger(__name__)

# Cache the detection result so we only check once per process
_detection_cache = None


def _tcp_latency(host: str, port: int = 443, timeout: float = 3.0):
    """TCP handshake latency in seconds, or None if unreachable."""
    try:
        start = time.perf_counter()
        sock = socket.create_connection((host, port), timeout=timeout)
        latency = time.perf_counter() - start
        sock.close()
        return latency
    except (socket.timeout, socket.error, OSError):
        return None


def need_proxy(timeout: float = 3.0) -> bool:
    """
    Detect if the current network environment needs a proxy to access HF.

    Returns True if a proxy is needed (use ModelScope / hf-mirror),
    False otherwise.

    Detection methods (in order):
    1. Check environment variable ``USE_MODELSCOPE`` for manual override
    2. Try TCP connection to huggingface.co (if unreachable, need proxy)
    3. Compare latency between modelscope.cn and huggingface.co

    The result is cached after the first call so subsequent calls are instant.
    """
    global _detection_cache
    if _detection_cache is not None:
        return _detection_cache

    # Allow manual override via environment variable
    env_override = os.environ.get("USE_MODELSCOPE", "").lower()
    if env_override == "true":
        logger.info("Network detection: forced to proxy mode (USE_MODELSCOPE=true)")
        _detection_cache = True
        return True
    if env_override == "false":
        logger.info("Network detection: forced to direct mode (USE_MODELSCOPE=false)")
        _detection_cache = False
        return False

    # Check if huggingface.co is accessible and measure latency
    hf_latency = _tcp_latency("huggingface.co", timeout=timeout)
    if hf_latency is None:
        logger.info("Network detection: huggingface.co is unreachable, need proxy")
        _detection_cache = True
        return True

    # Compare: if modelscope is significantly faster, likely in China
    ms_latency = _tcp_latency("modelscope.cn", timeout=timeout)
    if ms_latency is not None and ms_latency < hf_latency * 0.5:
        logger.info(
            f"Network detection: modelscope.cn ({ms_latency:.2f}s) is significantly "
            f"faster than huggingface.co ({hf_latency:.2f}s), need proxy"
        )
        _detection_cache = True
        return True

    logger.info("Network detection: huggingface.co is accessible, direct mode")
    _detection_cache = False
    return False
