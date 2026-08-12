"""Shared utilities for the TensorRT backend."""

import os


def resolve_engine_paths(precision="fp16", backend_dir=None, project_root=None):
    """Resolve TRT engine paths from precision and backend directory.

    Args:
        precision: One of "fp32", "fp16", "int8", "int4".
        backend_dir: Path to the backends/trt/ directory. Auto-detected if None.
        project_root: Path to the project root. Auto-detected if None.

    Returns:
        dict with keys: trt_engine_dir, gpt_engine_dir, speed_emb_path, model_dir,
        and individual engine paths.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if backend_dir is None:
        backend_dir = os.path.join(project_root, "backends", "trt")

    trt_suffix = "fp32" if precision == "fp32" else "fp16"
    trt_engine_dir = os.path.join(backend_dir, f"trt_engines_{trt_suffix}")
    gpt_engine_dir = os.path.join(backend_dir, f"tllm_engines_{precision}")
    onnx_dir = os.path.join(backend_dir, "onnx_models")
    speed_emb_path = os.path.join(onnx_dir, "speed_emb.pt")
    model_dir = os.path.join(project_root, "checkpoints")

    return {
        "model_dir": model_dir,
        "trt_engine_dir": trt_engine_dir,
        "gpt_engine_dir": gpt_engine_dir,
        "speed_emb_path": speed_emb_path,
        "speech_semantic_encoder_engine": os.path.join(trt_engine_dir, "speech_semantic_encoder.engine"),
        "semantic_codec_engine": os.path.join(trt_engine_dir, "semantic_codec.engine"),
        "speaker_perceiver_conditioner_engine": os.path.join(trt_engine_dir, "speaker_perceiver_conditioner.engine"),
        "emotion_perceiver_conditioner_engine": os.path.join(trt_engine_dir, "emotion_perceiver_conditioner.engine"),
        "latent_projector_engine": os.path.join(trt_engine_dir, "latent_projector.engine"),
        "length_regulator_engine": os.path.join(trt_engine_dir, "length_regulator.engine"),
        "campplus_engine": os.path.join(trt_engine_dir, "campplus.engine"),
        "dit_engine": os.path.join(trt_engine_dir, "dit.engine"),
        "bigvgan_engine": os.path.join(trt_engine_dir, "bigvgan.engine"),
    }
