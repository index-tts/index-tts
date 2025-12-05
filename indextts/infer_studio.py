import os
import logging
import warnings
import multiprocessing

try:
    sys_threads = multiprocessing.cpu_count()
    ALLOCATED_THREADS = max(1, sys_threads - 2)

except Exception as e:
    sys_threads = "Unknown"
    ALLOCATED_THREADS = 4

print(f">> 🖥️  CPU Auto-Config: Detected {sys_threads} threads.")
print(f">> ⚙️  Setting AI to use {ALLOCATED_THREADS} threads (leaving 2 for System).")

os.environ["OMP_NUM_THREADS"] = str(ALLOCATED_THREADS)
os.environ["MKL_NUM_THREADS"] = str(ALLOCATED_THREADS)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*past_key_values.*")

import gc

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

import json
import re
import time
import librosa
import torch
import torchaudio
import psutil
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from omegaconf import OmegaConf
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram
from transformers import AutoTokenizer, SeamlessM4TFeatureExtractor
from modelscope import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import safetensors
import random
import torch.nn.functional as F

torch.set_num_threads(ALLOCATED_THREADS)


class IndexTTS2:
    @staticmethod
    def _load_gpt_state_dict(path: str) -> dict:
        checkpoint = torch.load(path, map_location="cpu")
        return checkpoint.get("model", checkpoint)

    @staticmethod
    def _infer_vocab_size(state_dict: dict) -> int | None:
        for key in ("text_embedding.weight", "text_head.weight", "text_head.bias"):
            tensor = state_dict.get(key)
            if tensor is not None:
                return tensor.shape[0]
        return None

    @staticmethod
    def _resolve_attr(module, key: str):
        obj = module
        for part in key.split("."):
            obj = getattr(obj, part)
        return obj

    @staticmethod
    def _copy_resized_weight(name: str, param, weight: torch.Tensor) -> None:
        target = param.data
        source = weight.to(device=target.device, dtype=target.dtype)
        if target.shape != source.shape:
            pass
        if target.ndim == 1:
            length = min(target.shape[0], source.shape[0])
            target[:length].copy_(source[:length])
        elif target.ndim == 2:
            rows = min(target.shape[0], source.shape[0])
            cols = min(target.shape[1], source.shape[1])
            target[:rows, :cols].copy_(source[:rows, :cols])

    def _load_gpt_weights(self, model: UnifiedVoice, state_dict: dict) -> None:
        filtered_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("inference_model."):
                continue
            if ".lora_" in key:
                continue
            new_key = key.replace(".base_layer.", ".")
            filtered_state[new_key] = value

        resizable_keys = ("text_embedding.weight", "text_head.weight", "text_head.bias")
        resizable: dict[str, torch.Tensor] = {}
        for key in resizable_keys:
            tensor = filtered_state.pop(key, None)
            if tensor is not None:
                resizable[key] = tensor

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        for key, weight in resizable.items():
            param = self._resolve_attr(model, key)
            self._copy_resized_weight(key, param, weight)

    def __init__(
        self,
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        is_fp16=False,
        device=None,
        use_cuda_kernel=None,
        use_torch_compile=False,
    ):
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = (
                use_cuda_kernel is not None and use_cuda_kernel and "cuda" in device
            )
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = False
            print(">> 🟢 Intel XPU Detected.")
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> 🍎 Apple Silicon MPS Detected.")
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(
                f">> Running on CPU (Optimized Mode with {ALLOCATED_THREADS} threads)."
            )

        self.use_torch_compile = use_torch_compile and ("cuda" in self.device)

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.qwen_emo = None
        self.qwen_model_path = os.path.join(self.model_dir, self.cfg.qwen_emo_path)

        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        gpt_state = self._load_gpt_state_dict(self.gpt_path)
        vocab = self._infer_vocab_size(gpt_state)
        if vocab and self.cfg.gpt.get("number_text_tokens") != vocab:
            self.cfg.gpt.number_text_tokens = vocab

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self._load_gpt_weights(self.gpt, gpt_state)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT loaded.")

        self.gpt.post_init_gpt2_config(
            use_deepspeed=False, kv_cache=True, half=self.is_fp16
        )

        if self.use_cuda_kernel:
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load

                _ = load.load()
            except:
                self.use_cuda_kernel = False

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = (
            build_semantic_model(os.path.join(self.model_dir, self.cfg.w2v_stat))
        )
        self.semantic_model = self.semantic_model.to(self.device).eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download(
            "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
        )
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device).eval()

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device).eval()
        self.s2mel.models["cfm"].estimator.setup_caches(
            max_batch_size=1, max_seq_length=8192
        )

        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(
            torch.load(campplus_ckpt_path, map_location="cpu")
        )
        self.campplus_model = campplus_model.to(self.device).eval()

        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(
            bigvgan_name, use_cuda_kernel=False
        )
        self.bigvgan = self.bigvgan.to(self.device).eval()
        self.bigvgan.remove_weight_norm()

        if self.use_torch_compile:
            print(">> 🚀 Enabled torch.compile (First run will be slow)...")
            try:
                self.s2mel = torch.compile(self.s2mel, mode="reduce-overhead")
                self.bigvgan = torch.compile(self.bigvgan, mode="reduce-overhead")
            except Exception as e:
                print(f">> ⚠️ torch.compile failed: {e}")

        print(">> Models ready.")

        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        if os.path.exists(self.glossary_path):
            self.normalizer.load_glossary_from_yaml(self.glossary_path)
            print(">> 📖 Glossary loaded from YAML.")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)

        self.emo_matrix = torch.load(
            os.path.join(self.model_dir, self.cfg.emo_matrix), map_location=self.device
        ).to(self.device)
        self.emo_num = list(self.cfg.emo_num)
        self.spk_matrix = torch.load(
            os.path.join(self.model_dir, self.cfg.spk_matrix), map_location=self.device
        ).to(self.device)
        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        # SPEED OPTIMIZATION: QUANTIZATION ONLY (NO COMPILE)
        if self.device == "cpu":
            print(">> ⚡ Applying Int8 Quantization (Speed + Low RAM)...")
            try:
                self.gpt = torch.quantization.quantize_dynamic(
                    self.gpt, {torch.nn.Linear}, dtype=torch.qint8
                )
            except:
                pass
            try:
                self.s2mel = torch.quantization.quantize_dynamic(
                    self.s2mel, {torch.nn.Linear}, dtype=torch.qint8
                )
            except:
                pass
            # BigVGAN often has issues with quant, skipping to be safe
            print(">> ✅ Models Optimized (Best Effort).")

        mel_fn_args = {
            "n_fft": self.cfg.s2mel["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": self.cfg.s2mel["preprocess_params"]["spect_params"][
                "win_length"
            ],
            "hop_size": self.cfg.s2mel["preprocess_params"]["spect_params"][
                "hop_length"
            ],
            "num_mels": self.cfg.s2mel["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": (
                None
                if self.cfg.s2mel["preprocess_params"]["spect_params"].get(
                    "fmax", "None"
                )
                == "None"
                else 8000
            ),
            "center": False,
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None
        self.gr_progress = None

    def _ensure_qwen_loaded(self):
        if self.qwen_emo is None:
            mem = psutil.virtual_memory()
            if mem.available < 4 * 1024 * 1024 * 1024:
                print(">> WARNING: Low RAM. Loading Qwen might cause swapping.")

            print(">> Lazy loading Qwen Emotion model...")
            try:
                self.qwen_emo = QwenEmotion(self.qwen_model_path)
            except:
                self.qwen_emo = None

    def unload_qwen(self):
        if self.qwen_emo is not None:
            print(">> Unloading Qwen to free RAM...")
            del self.qwen_emo.model
            del self.qwen_emo
            self.qwen_emo = None
            gc.collect()

    def warmup(self):
        if self.device == "cpu":
            return
        try:
            dummy = torch.randn(1, 80, 10).to(self.device)
            if self.is_fp16:
                dummy = dummy.half()
            with torch.no_grad():
                _ = self.bigvgan(dummy)
        except:
            pass

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]
        return (feat - self.semantic_mean) / self.semantic_std

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        if not wavs or interval_silence <= 0:
            return wavs
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(wavs[0].size(0), sil_dur, device="cpu")
        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav.cpu())
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)
        return wavs_list

    def crossfade_join(self, chunks, overlap_ms=50, sr=22050):
        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0].cpu()

        overlap_frames = int(sr * overlap_ms / 1000)
        final_wave = chunks[0]

        for i in range(1, len(chunks)):
            next_chunk = chunks[i].to(final_wave.device)
            fade_out = torch.linspace(1, 0, overlap_frames).to(final_wave.device)
            fade_in = torch.linspace(0, 1, overlap_frames).to(final_wave.device)

            if (
                final_wave.shape[-1] < overlap_frames
                or next_chunk.shape[-1] < overlap_frames
            ):
                final_wave = torch.cat([final_wave, next_chunk], dim=-1)
                continue

            overlap_zone_prev = final_wave[..., -overlap_frames:] * fade_out
            overlap_zone_next = next_chunk[..., :overlap_frames] * fade_in
            merged_zone = overlap_zone_prev + overlap_zone_next

            final_wave = torch.cat(
                [
                    final_wave[..., :-overlap_frames],
                    merged_zone,
                    next_chunk[..., overlap_frames:],
                ],
                dim=-1,
            )

        return final_wave.cpu()

    def _seconds_to_srt_time(self, seconds):
        millis = int((seconds % 1) * 1000)
        seconds = int(seconds)
        minutes = seconds // 60
        hours = minutes // 60
        minutes %= 60
        seconds %= 60
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    def infer(
        self,
        spk_audio_prompt,
        text,
        output_path,
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text=None,
        use_random=False,
        interval_silence=200,
        seed=-1,
        diffusion_steps=50,
        inference_cfg_rate=0.7,
        verbose=False,
        max_text_tokens_per_sentence=120,
        split_text=True,
        **generation_kwargs,
    ):

        print(
            f">> 🧵 DEBUG: PyTorch is actively using {torch.get_num_threads()} threads for this generation."
        )

        start_time = time.perf_counter()
        yield (0.01, "Initializing...")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = int(seed)

        if verbose:
            print(f"-> Using Seed: {actual_seed}")

        torch.manual_seed(actual_seed)
        random.seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)
            torch.cuda.manual_seed_all(actual_seed)

        # Emotion
        if use_emo_text:
            yield (0.02, "Loading Emotion Model (Qwen)...")
            self._ensure_qwen_loaded()
            if self.qwen_emo:
                yield (0.05, "Analyzing Emotion from text...")
                emo_audio_prompt = None
                emo_alpha = 1.0
                if emo_text is None:
                    emo_text = text
                emo_dict = self.qwen_emo.inference(emo_text)
                print(f"Emotions: {emo_dict}")
                emo_vector = list(emo_dict.values())
                self.unload_qwen()
            else:
                use_emo_text = False

        if emo_vector is not None:
            emo_audio_prompt = None
            emo_alpha = 1.0
        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0

        # Prompt
        yield (0.08, "Processing Voice Prompt...")
        if (
            self.cache_spk_cond is None
            or self.cache_spk_audio_prompt != spk_audio_prompt
        ):
            try:
                audio, sr = librosa.load(spk_audio_prompt)
            except:
                yield (1.0, "Error loading audio")
                return
            audio = torch.tensor(audio).unsqueeze(0)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
            inputs = self.extract_features(
                audio_16k, sampling_rate=16000, return_tensors="pt"
            )
            spk_cond_emb = self.get_emb(
                inputs["input_features"].to(self.device),
                inputs["attention_mask"].to(self.device),
            )
            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = self.mel_fn(audio_22k.to(self.device).float())
            ref_len = torch.LongTensor([ref_mel.size(2)]).to(self.device)
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(self.device),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = self.campplus_model(feat.unsqueeze(0))
            prompt_condition = self.s2mel.models["length_regulator"](
                S_ref, ylens=ref_len, n_quantizers=3, f0=None
            )[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style, prompt_condition, spk_cond_emb, ref_mel = (
                self.cache_s2mel_style,
                self.cache_s2mel_prompt,
                self.cache_spk_cond,
                self.cache_mel,
            )

        # Vectors
        emovec_mat = None
        weight_vector = None
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector).to(self.device)
            random_index = (
                [random.randint(0, x - 1) for x in self.emo_num]
                if use_random
                else [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
            )
            emo_matrix = torch.cat(
                [
                    tmp[index].unsqueeze(0)
                    for index, tmp in zip(random_index, self.emo_matrix)
                ],
                0,
            )
            emovec_mat = (weight_vector.unsqueeze(1) * emo_matrix).sum(0).unsqueeze(0)

        # Emo Prompt
        if (
            self.cache_emo_cond is None
            or self.cache_emo_audio_prompt != emo_audio_prompt
        ):
            emo_audio, _ = librosa.load(emo_audio_prompt, sr=16000)
            emo_inputs = self.extract_features(
                emo_audio, sampling_rate=16000, return_tensors="pt"
            )
            emo_cond_emb = self.get_emb(
                emo_inputs["input_features"].to(self.device),
                emo_inputs["attention_mask"].to(self.device),
            )
            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        yield (0.1, "Tokenizing text...")
        print(">> Tokenizing text...")

        processing_data = []

        if split_text:
            raw_sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in raw_sentences:
                if s.strip():
                    if len(s) > 300:
                        subs = self.tokenizer.split_segments(
                            self.tokenizer.tokenize(s), max_text_tokens_per_sentence
                        )
                        for idx_s, sub_s in enumerate(subs):
                            display = s if idx_s == 0 else ""
                            processing_data.append((display, sub_s))
                    else:
                        tokens = self.tokenizer.tokenize(s)
                        processing_data.append((s, tokens))
        else:
            words = text.split()
            text_tokens_list = []
            batch_size = 50
            for i in range(0, len(words), batch_size):
                batch_text = " ".join(words[i : i + batch_size])
                if batch_text.strip():
                    text_tokens_list.extend(self.tokenizer.tokenize(batch_text))

            sent_tokens_list = self.tokenizer.split_segments(
                text_tokens_list,
                max_text_tokens_per_segment=max_text_tokens_per_sentence,
            )
            for i, s in enumerate(sent_tokens_list):
                display = text if i == 0 else ""
                processing_data.append((display, s))

        total_sentences = len(processing_data)

        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = float(generation_kwargs.pop("top_p", 0.95))
        top_k = int(generation_kwargs.pop("top_k", 50))
        temperature = float(generation_kwargs.pop("temperature", 1.0))
        length_penalty = float(generation_kwargs.pop("length_penalty", 0.0))
        num_beams = int(generation_kwargs.pop("num_beams", 1))
        repetition_penalty = float(generation_kwargs.pop("repetition_penalty", 10.0))
        max_mel_tokens = int(generation_kwargs.pop("max_mel_tokens", 1500))

        wavs = []
        srt_entries = []
        current_audio_cursor = 0.0
        silence_sec = interval_silence / 1000.0

        print(
            f">> Generating {total_sentences} segments. Seed: {actual_seed}, Steps: {diffusion_steps}, CFG: {inference_cfg_rate}"
        )

        for idx, (sentence_text, sent_tokens) in enumerate(processing_data):
            token_count = len(sent_tokens)
            current_progress = 0.1 + (0.8 * (idx / total_sentences))

            msg = (
                f"Segment {idx+1}/{total_sentences}: Processing {token_count} tokens..."
            )
            print(f">> {msg}")
            yield (current_progress, msg)

            text_tokens = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(sent_tokens),
                dtype=torch.int32,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                with torch.amp.autocast(
                    self.device.split(":")[0],
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=self.device),
                        alpha=emo_alpha,
                    )
                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor(
                            [spk_cond_emb.shape[-1]], device=self.device
                        ),
                        emo_cond_lengths=torch.tensor(
                            [emo_cond_emb.shape[-1]], device=self.device
                        ),
                        emo_vec=emovec,
                        do_sample=do_sample,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=1,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        **generation_kwargs,
                    )

                code_lens_list = []
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_lens_list.append(len(code))
                    else:
                        code_lens_list.append(
                            (code == self.stop_mel_token)
                            .nonzero(as_tuple=False)[0]
                            .item()
                        )
                codes = codes[:, : code_lens_list[0]]
                code_lens = torch.LongTensor(code_lens_list).to(self.device)

                with torch.amp.autocast(
                    self.device.split(":")[0],
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=self.device),
                        codes,
                        torch.tensor([codes.shape[-1]], device=self.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor(
                            [spk_cond_emb.shape[-1]], device=self.device
                        ),
                        emo_cond_mel_lengths=torch.tensor(
                            [emo_cond_emb.shape[-1]], device=self.device
                        ),
                        emo_vec=emovec,
                        use_speed=torch.zeros(spk_cond_emb.size(0))
                        .to(self.device)
                        .long(),
                    )

                with torch.amp.autocast(self.device.split(":")[0], enabled=False):
                    latent = self.s2mel.models["gpt_layer"](latent)
                    S_infer = (
                        self.semantic_codec.quantizer.vq2emb(
                            codes.unsqueeze(1)
                        ).transpose(1, 2)
                        + latent
                    )
                    target_lengths = (code_lens * 1.72).long()
                    cond = self.s2mel.models["length_regulator"](
                        S_infer, ylens=target_lengths, n_quantizers=3, f0=None
                    )[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)

                    vc_target = self.s2mel.models["cfm"].inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(self.device),
                        ref_mel,
                        style,
                        None,
                        diffusion_steps,
                        inference_cfg_rate=inference_cfg_rate,
                    )

                    wav_seg = (
                        self.bigvgan(vc_target[:, :, ref_mel.size(-1) :].float())
                        .squeeze()
                        .unsqueeze(0)
                        .squeeze(1)
                    )
                    wav_seg = torch.clamp(32767 * wav_seg, -32767.0, 32767.0).cpu()

                    duration_seconds = wav_seg.shape[-1] / 22050.0
                    abs_start = current_audio_cursor
                    abs_end = current_audio_cursor + duration_seconds

                    display_text = sentence_text if sentence_text else "(...)"
                    srt_entries.append(
                        {
                            "index": idx + 1,
                            "start": self._seconds_to_srt_time(abs_start),
                            "end": self._seconds_to_srt_time(abs_end),
                            "text": display_text.strip(),
                        }
                    )

                    current_audio_cursor += duration_seconds + silence_sec
                    wavs.append(wav_seg)

                    # Yield Chunk for Streaming
                    yield (wav_seg, actual_seed, display_text)

                    del (
                        vc_target,
                        wav_seg,
                        latent,
                        S_infer,
                        cond,
                        cat_condition,
                        codes,
                        text_tokens,
                    )
                    gc.collect()

        yield (0.95, "Finalizing & Saving...")

        if interval_silence > 0:
            wavs = self.insert_interval_silence(
                wavs, sampling_rate=22050, interval_silence=interval_silence
            )
            if not wavs:
                return None
            wav = torch.cat(wavs, dim=1)
        else:
            wav = self.crossfade_join(wavs, overlap_ms=50)

        if wav is None:
            return None

        # Generate SRT Content
        srt_content = ""
        for entry in srt_entries:
            srt_content += f"{entry['index']}\n"
            srt_content += f"{entry['start']} --> {entry['end']}\n"
            srt_content += f"{entry['text']}\n\n"

        end_time = time.perf_counter()
        wav_length = wav.shape[-1] / 22050
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        if wav_length > 0:
            print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        if output_path:
            if os.path.isfile(output_path):
                os.remove(output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), 22050)

            # Save SRT
            srt_path = os.path.splitext(output_path)[0] + ".srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            print(f">> wav file saved to: {output_path}")
            print(f">> srt file saved to: {srt_path}")

            yield (1.0, "Done")
            yield (output_path, actual_seed, srt_path)
        else:
            yield ((22050, wav.type(torch.int16).numpy().T), actual_seed, None)


def find_most_similar_cosine(query_vector, matrix):
    return torch.argmax(
        F.cosine_similarity(query_vector.float(), matrix.float(), dim=1)
    )


class QwenEmotion:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, quantization_config=bnb_config, device_map="auto"
            )
            print(">> Qwen 4-bit loaded.")
        except:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype="float16", device_map="auto"
            )
            print(">> Qwen FP16 loaded.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = [
            "高兴",
            "愤怒",
            "悲伤",
            "恐惧",
            "反感",
            "低落",
            "惊讶",
            "自然",
        ]
        self.melancholic_words = {
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }

    def clamp_score(self, value):
        return max(0.0, min(1.2, value))

    def inference(self, text_input):
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": text_input},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except:
            index = 0
        content_str = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        )
        try:
            content = json.loads(content_str)
        except:
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content_str)
            }
        if any(w in text_input.lower() for w in self.melancholic_words):
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get(
                "悲伤", 0.0
            )
        emo_dict = {
            self.cn_key_to_en[k]: self.clamp_score(content.get(k, 0.0))
            for k in self.desired_vector_order
        }
        if all(v <= 0 for v in emo_dict.values()):
            emo_dict["calm"] = 1.0
        return emo_dict
