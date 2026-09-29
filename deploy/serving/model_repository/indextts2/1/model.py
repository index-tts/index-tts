"""Non-streaming FasterIndexTTS2 model for Triton Python Backend.

Batched inference: all concurrent requests grouped by Triton's dynamic batcher
are passed together as a single batch to the pipeline, then results are split
back to individual responses.
"""

import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict

import numpy as np
import triton_python_backend_utils as pb_utils


class SpeakerCache:
    """Thread-safe LRU cache for pre-computed speaker conditions."""

    def __init__(self, pipeline, max_size=64):
        self._pipeline = pipeline
        self._max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compute(self, audio_bytes: bytes):
        key = hashlib.sha256(audio_bytes).hexdigest()[:16]
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            condition = self._pipeline.preload_speaker(tmp_path)
        finally:
            os.unlink(tmp_path)

        with self._lock:
            self._cache[key] = condition
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return condition


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        device_id = int(args.get("model_instance_device_id", "0"))

        # Restrict GPU visibility BEFORE importing torch/TRT.
        # This ensures TRT-LLM and PyTorch only see the assigned GPU as cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        import torch
        torch.cuda.set_device(0)

        from deploy.pipeline import FasterIndexTTS2
        from deploy.utils import resolve_engine_paths

        paths = resolve_engine_paths("fp16")

        self.pipeline = FasterIndexTTS2(
            config_path=os.path.join(paths["model_dir"], "config.yaml"),
            model_dir=paths["model_dir"],
            gpt_engine_dir=paths["gpt_engine_dir"],
            speed_emb_path=paths["speed_emb_path"],
            speech_semantic_encoder_engine=paths["speech_semantic_encoder_engine"],
            semantic_codec_engine=paths["semantic_codec_engine"],
            speaker_perceiver_conditioner_engine=paths["speaker_perceiver_conditioner_engine"],
            emotion_perceiver_conditioner_engine=paths["emotion_perceiver_conditioner_engine"],
            latent_projector_engine=paths["latent_projector_engine"],
            length_regulator_engine=paths["length_regulator_engine"],
            campplus_engine=paths["campplus_engine"],
            dit_engine=paths["dit_engine"],
            bigvgan_engine=paths["bigvgan_engine"],
            device="cuda:0",
        )

        self.speaker_cache = SpeakerCache(self.pipeline, max_size=64)

    def _decode_requests(self, requests):
        """Decode a batch of Triton requests into pipeline-ready inputs."""
        texts = []
        spk_conditions = []
        emo_spk_conditions = []
        emo_alphas = []
        emo_vectors = []

        for request in requests:
            text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()[0][0]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            texts.append(text)

            speaker_audio = pb_utils.get_input_tensor_by_name(request, "speaker_audio").as_numpy()[0][0]
            if isinstance(speaker_audio, np.void):
                speaker_audio = speaker_audio.tobytes()
            spk_conditions.append(self.speaker_cache.get_or_compute(speaker_audio))

            emo_audio_tensor = pb_utils.get_input_tensor_by_name(request, "emo_speaker_audio")
            if emo_audio_tensor is not None:
                emo_bytes = emo_audio_tensor.as_numpy()[0][0]
                if isinstance(emo_bytes, np.void):
                    emo_bytes = emo_bytes.tobytes()
                if len(emo_bytes) > 0:
                    emo_spk_conditions.append(self.speaker_cache.get_or_compute(emo_bytes))
                else:
                    emo_spk_conditions.append(None)
            else:
                emo_spk_conditions.append(None)

            emo_alpha_tensor = pb_utils.get_input_tensor_by_name(request, "emo_alpha")
            if emo_alpha_tensor is not None:
                emo_alphas.append(float(emo_alpha_tensor.as_numpy()[0][0]))
            else:
                emo_alphas.append(1.0)

            emo_vec_tensor = pb_utils.get_input_tensor_by_name(request, "emo_vector")
            if emo_vec_tensor is not None:
                vec = emo_vec_tensor.as_numpy()[0]
                if np.any(vec != 0):
                    emo_vectors.append(vec.tolist())
                else:
                    emo_vectors.append(None)
            else:
                emo_vectors.append(None)

        return texts, spk_conditions, emo_spk_conditions, emo_alphas, emo_vectors

    def execute(self, requests):
        """Batched non-streaming inference.

        All requests are passed as a single batch to pipeline.generate(),
        then results are split back to individual responses.
        """
        batch_size = len(requests)
        texts, spk_conds, emo_spk_conds, emo_alphas, emo_vecs = self._decode_requests(requests)

        results = self.pipeline.generate(
            text=texts if batch_size > 1 else texts[0],
            speaker=spk_conds if batch_size > 1 else spk_conds[0],
            emo_speaker=emo_spk_conds if batch_size > 1 else emo_spk_conds[0],
            emo_alpha=emo_alphas if batch_size > 1 else emo_alphas[0],
            emo_vector=emo_vecs if batch_size > 1 else emo_vecs[0],
        )

        if batch_size == 1:
            results = [results]

        responses = []
        for b in range(batch_size):
            sr, audio = results[b]
            audio_int16 = audio.flatten().astype(np.int16)

            out_audio = pb_utils.Tensor("audio", audio_int16.reshape(1, -1))
            out_sr = pb_utils.Tensor("sample_rate", np.array([[sr]], dtype=np.int32))
            out_len = pb_utils.Tensor("audio_length", np.array([[len(audio_int16)]], dtype=np.int32))

            responses.append(pb_utils.InferenceResponse(output_tensors=[out_audio, out_sr, out_len]))

        return responses

    def finalize(self):
        pass
