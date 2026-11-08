import os
from subprocess import CalledProcessError

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import json
import re
import time
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from omegaconf import OmegaConf

from indextts.codec.models import EnhancedCodec
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.common import fade_out_tail, save_pcm_wav
from indextts.utils.front import TextNormalizer
from indextts.utils.tokenizer import get_tokenizer, lang_to_token
from indextts.utils.ja_g2p import JapaneseG2PProcessor
from indextts.utils.nemo_tn import normalize_text as nemo_text_normalize

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram
from transformers import AutoTokenizer
from modelscope import AutoModelForCausalLM
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
import random
import torch.nn.functional as F

PRONUNCIATION_ANNOTATION_PATTERN = re.compile(r'<([^|>\n]+)\|([^>\n]+)>')
"""
匹配发音标注格式：<文字|发音>
例如：<going|G OW1 . IH0 NG>，<行|XING2>
"""
def is_kana(s: str) -> bool:
    hira = re.compile(r'^[\u3040-\u309F]+$')
    kata = re.compile(r'^[\u30A0-\u30FF]+$')
    if hira.fullmatch(s):
        return True
    elif kata.fullmatch(s):
        return True
    return False

def apply_pronunciation_annotations(text: str) -> str:
    """
    处理发音标注格式 <文字|发音>，转换为特殊token包裹的发音
    - 文字含中文 -> <|SPECIAL_TOKEN_2|>发音<|SPECIAL_TOKEN_2|>
    - 文字为英文 -> <|SPECIAL_TOKEN_1|>发音<|SPECIAL_TOKEN_1|>
    发音内容统一转大写

    例如：
        <going|G OW1 . IH0 NG> -> <|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>
        <行|XING2>              -> <|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>
    """
    def _replace(match):
        word = match.group(1)
        pronunciation = match.group(2).upper()
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", word))
        if is_kana(pronunciation):
            return f' {pronunciation} '
        else:
            token = 'SPECIAL_TOKEN_2' if has_chinese else 'SPECIAL_TOKEN_1'
            return f'<|{token}|>{pronunciation}<|{token}|>'
    return PRONUNCIATION_ANNOTATION_PATTERN.sub(_replace, text)


class IndexTTS2:
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=False, device=None,
            use_cuda_kernel=None,use_deepspeed=False, use_accel=False, use_torch_compile=False, use_qwen_emo=False
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            use_bf16 (bool): whether to use bf16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_deepspeed (bool): whether to use DeepSpeed or not.
            use_accel (bool): whether to use acceleration engine for GPT2 or not.
            use_torch_compile (bool): whether to use torch.compile for optimization or not.
            use_qwen_emo (bool): if True, load the QwenEmotion text-to-emotion model.
                Required for ``infer(..., use_emo_text=True)``. Attempting to use emotion-text
                guidance when this is disabled will raise a RuntimeError.
        """
        if device is not None:
            self.device = device
            self.use_bf16 = False if device == "cpu" else use_bf16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_bf16 = use_bf16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_bf16 = use_bf16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_bf16 = False  # Use bfloat16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_bf16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.bfloat16 if self.use_bf16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile

        # Detect low-VRAM GPUs (< 10 GB) to enable automatic text chunking
        self.low_vram = False
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            dev_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            total_vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
            if total_vram_gb < 10.0:
                self.low_vram = True
                print(f">> Low-VRAM mode enabled ({total_vram_gb:.1f} GB < 10 GB), long text will be split into chunks")

        if use_qwen_emo:
            self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))
        else:
            self.qwen_emo = None
            print(">> QwenEmotion not loaded (use_qwen_emo=False)")

        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=self.use_accel, spk_cond_mode="campplus")
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_bf16:
            self.gpt.eval().bfloat16()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)

        if use_deepspeed:
            try:
                import deepspeed
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}")

        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_bf16)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d

                print(">> Preload custom CUDA kernel for BigVGAN", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                print(f"{e!r}")
                self.use_cuda_kernel = False

        w2v_bert_dir = os.path.join(self.model_dir, "hf_cache", "w2v-bert-2.0")
        if not os.path.isdir(w2v_bert_dir):
            from indextts.utils.model_download import ensure_models_available
            aux_paths = ensure_models_available(self.model_dir)
            w2v_bert_dir = aux_paths["w2v_bert"]
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_dir, local_files_only=True)
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_bert_dir, local_files_only=True)
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        stat_mean_var = torch.load(os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_mean = stat_mean_var["mean"].to(self.device)
        self.semantic_std = torch.sqrt(stat_mean_var["var"]).to(self.device)

        start_time = time.perf_counter()
        self.semantic_codec = EnhancedCodec(**self.cfg.semantic_codec, cfg=self.cfg.semantic_codec)
        codec_ckpt_path = os.path.join(self.model_dir, "codec.pth")
        self.semantic_codec.load_checkpoint(codec_ckpt_path)
        print('>> semantic_codec weights restored from:', codec_ckpt_path)
        self.semantic_codec = self.semantic_codec.to(self.device)
        print('>> semantic_codec weights restored cost: ', time.perf_counter() - start_time)

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Enable torch.compile optimization if requested
        if self.use_torch_compile:
            print(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            print(">> torch.compile optimization enabled successfully")

        self.s2mel.eval()
        print(">> s2mel weights restored from:", s2mel_path)

        # load campplus_model
        campplus_ckpt_path = os.path.join(self.model_dir, "hf_cache", "campplus_cn_common.bin")
        if not os.path.isfile(campplus_ckpt_path):
            from indextts.utils.model_download import ensure_models_available
            aux_paths = ensure_models_available(self.model_dir)
            campplus_ckpt_path = aux_paths["campplus"]
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        print(">> campplus_model weights restored from:", campplus_ckpt_path)

        bigvgan_dir = os.path.join(self.model_dir, "hf_cache", "bigvgan")
        if not os.path.isdir(bigvgan_dir):
            from indextts.utils.model_download import ensure_models_available
            aux_paths = ensure_models_available(self.model_dir)
            bigvgan_dir = aux_paths["bigvgan"]
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_dir, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", bigvgan_dir)

        self.tokenizer = get_tokenizer(multilingual=True, model_dir=self.model_dir)
        self.ja_text_process = JapaneseG2PProcessor(g2p_ratio=0)
        self.text_process = TextNormalizer(enable_glossary=True)
        self.text_process.load()

        # 加载术语词汇表（如果存在）
        self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        if os.path.exists(self.glossary_path):
            self.text_process.load_glossary_from_yaml(self.glossary_path)
            print(">> Glossary loaded from:", self.glossary_path)

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # 缓存参考音频：
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None

        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    @torch.no_grad()
    def get_scode(self, inputs):
        semantic_code, feat = self.semantic_codec.quantize(inputs)
        # vq = self.semantic_codec.quantizer.vq2emb(semantic_code.unsqueeze(1))
        # vq = vq.transpose(1,2)
        return semantic_code

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[
                               k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Silences to be insert between generated segments.
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between generated segments.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def _load_and_cut_audio(self,audio_path,max_audio_length_seconds,verbose=False,sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path,sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, sr

    SPLIT_PROTECTED_PATTERN = re.compile(r'<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>')

    def _token_len(self, text):
        return len(self.tokenizer.encode(text, allowed_special='all'))

    def _split_atomic_pieces(self, text):
        pieces, pos = [], 0
        for m in self.SPLIT_PROTECTED_PATTERN.finditer(text):
            if m.start() > pos:
                pieces.append((text[pos:m.start()], False))
            pieces.append((m.group(0), True))
            pos = m.end()
        if pos < len(text):
            pieces.append((text[pos:], False))
        return pieces

    def split_text_by_tokens(self, text, max_tokens, lang_prefix=""):
        capacity = self.gpt.text_pos_embedding.emb.num_embeddings
        budget = min(max_tokens, capacity - 2) - self._token_len(lang_prefix)
        budget = max(1, budget)
        if self._token_len(text) <= budget:
            return [text]

        chunks = []
        for piece, atomic in self._split_atomic_pieces(text):
            if atomic:
                chunks.append(piece)
                continue
            for part in re.split(r'(?<=[，。！？、；：,\.!\?;:\n])', piece):
                if not part:
                    continue
                if self._token_len(part) <= budget:
                    chunks.append(part)
                    continue
                current = ""
                for ch in part:
                    if current and self._token_len(current + ch) > budget:
                        chunks.append(current)
                        current = ch
                    else:
                        current += ch
                if current:
                    chunks.append(current)

        segments, current = [], ""
        for chunk in chunks:
            if current and self._token_len(current + chunk) > budget:
                segments.append(current)
                current = chunk
            else:
                current += chunk
        if current:
            segments.append(current)
        return segments or [text]

    @staticmethod
    def split_text_by_punctuation(text, max_chars=40):
        """Split text into segments of at most `max_chars` characters,
        breaking at punctuation boundaries. If a segment exceeds the limit
        and contains no punctuation, it is kept as-is to avoid mid-word splits."""
        import re
        # Split while keeping the delimiter attached to the preceding segment
        parts = re.split(r'(?<=[，。！？、；：,\.!\?;:\n])', text)
        segments = []
        current = ""
        for part in parts:
            if not part:
                continue
            if len(current) + len(part) <= max_chars:
                current += part
            else:
                if current:
                    segments.append(current)
                current = part
        if current:
            segments.append(current)
        return segments
    
    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        # apply biased emotion factors for better user experience,
        # by de-emphasizing emotions that can cause strange results
        if apply_bias:
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # the total emotion sum must be 0.8 or less
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]

        return emo_vector

    # 原始推理模式
    def infer(self, spk_audio_prompt, text, output_path, lang,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None, use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0, duration_factor=1.0, text_normalization=True, **generation_kwargs):
        if self.low_vram and not stream_return and len(text) > 40:
            segments = self.split_text_by_punctuation(text, max_chars=40)
            if verbose:
                print(f">> Low-VRAM: split into {len(segments)} segments: {segments}")
            sampling_rate = 22050
            all_wavs = []
            for seg_text in segments:
                gen = self.infer_generator(
                    spk_audio_prompt, seg_text, None, lang,
                    emo_audio_prompt, emo_alpha, emo_vector,
                    use_emo_text, emo_text, use_random, 0,
                    verbose, max_text_tokens_per_segment, False, 0, duration_factor=duration_factor, text_normalization=text_normalization, **generation_kwargs
                )
                result = None
                for result in gen:
                    pass
                if result is not None and isinstance(result, tuple):
                    sr, wav_data = result
                    all_wavs.append(torch.from_numpy(wav_data.T).to(torch.int16))
            if all_wavs:
                silence = torch.zeros(1, int(sampling_rate * interval_silence / 1000), dtype=torch.int16)
                wav_parts = []
                for i, w in enumerate(all_wavs):
                    wav_parts.append(w)
                    if i < len(all_wavs) - 1:
                        wav_parts.append(silence)
                wav = torch.cat(wav_parts, dim=1)
                if output_path:
                    if os.path.isfile(output_path):
                        os.remove(output_path)
                    if os.path.dirname(output_path) != "":
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    save_pcm_wav(output_path, wav, sampling_rate)
                    print(">> wav file saved to:", output_path)
                    return output_path
                else:
                    return (sampling_rate, wav.numpy().T)
            return None

        if stream_return:
            return self.infer_generator(
                spk_audio_prompt, text, output_path, lang,
                emo_audio_prompt, emo_alpha,
                emo_vector,
                use_emo_text, emo_text, use_random, interval_silence,
                verbose, max_text_tokens_per_segment, stream_return, more_segment_before, duration_factor=duration_factor, text_normalization=text_normalization, **generation_kwargs
            )
        else:
            try:
                return list(self.infer_generator(
                    spk_audio_prompt, text, output_path, lang,
                    emo_audio_prompt, emo_alpha,
                    emo_vector,
                    use_emo_text, emo_text, use_random, interval_silence,
                    verbose, max_text_tokens_per_segment, stream_return, more_segment_before, duration_factor=duration_factor, text_normalization=text_normalization, **generation_kwargs
                ))[0]
            except IndexError:
                return None


    def infer_generator(self, spk_audio_prompt, text, output_path, lang,
              emo_audio_prompt=None, emo_alpha=1.0, emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, quick_streaming_tokens=0, duration_factor=1.0, text_normalization=True, **generation_kwargs):
        print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                  f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                  f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                  f"emo_text:{emo_text}")
        start_time = time.perf_counter()

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if self.qwen_emo is None:
                raise RuntimeError(
                    "use_emo_text=True requires QwenEmotion, but it was not loaded at init "
                    "(use_qwen_emo=False). Re-construct IndexTTS2 with use_qwen_emo=True."
                )
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)
            print(f"detected emotion vectors from text: {emo_dict}")
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        # 如果参考音频改变了，才需要重新生成, 提升速度
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            input_features = input_features.to(self.device)
            attention_mask = attention_mask.to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                    num_mel_bins=80,
                                                    dither=0,
                                                    sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
            style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

            prompt_condition = self.s2mel.models['length_regulator'](
                # S_ref,
                spk_cond_emb,
                ylens=ref_target_lengths,
                n_quantizers=3,
                f0=None)[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt,15,verbose,sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"]
            emo_attention_mask = emo_inputs["attention_mask"]
            emo_input_features = emo_input_features.to(self.device)
            emo_attention_mask = emo_attention_mask.to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")
        lang_prefix = f'<|{lang.lower()}|> '

        text = self.text_process.clean_pattern.sub(lambda x: self.text_process.char_rep_map[x.group()], text)

        if text_normalization:
            if lang.lower() in ['zh', 'zhen', 'en']:
                text = self.text_process.normalize(text)
            elif lang.lower() in ['ja', 'es']:
                text = nemo_text_normalize(text, lang.lower())
            print(f'text after normalization: {text}')

        if lang.lower() in ['ja', 'zh', 'zhen', 'en']:
            text = text.lower()
        if lang.lower() == 'es':
            text = text.upper()
        text = apply_pronunciation_annotations(text)

        if lang.lower() == 'ja':
            text = self.ja_text_process.process_ja_text(text)
        text = re.sub(r'<\|([^|]+)\|>', lambda m: f'<|{m.group(1).upper()}|>', text)
        segments = self.split_text_by_tokens(text, max_text_tokens_per_segment, lang_prefix)
        segments_count = len(segments)
        segment_tokens = []
        for seg_text in segments:
            toks = self.tokenizer.encode(lang_prefix + seg_text, allowed_special='all')
            toks = torch.IntTensor(toks).unsqueeze(0).to(self.device)
            segment_tokens.append(F.pad(toks, (0, 1), value=1))
        lang = torch.LongTensor([lang_to_token(lang)]).to(self.device)
        if verbose:
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None # for stream_return
        for seg_idx, text_tokens in enumerate(segment_tokens):
            self._set_gr_progress(0.2 + 0.7 * seg_idx / segments_count,
                                  f"speech synthesis {seg_idx + 1}/{segments_count}...")
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emo_alpha
                    )

                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
                        # emovec = emovec_mat

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        lang,
                        emo_cond_emb,
                        cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        campplus_embedding=style,
                        wav=spk_audio_prompt,
                        do_sample=do_sample,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        **generation_kwargs
                    )

                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                #                 if verbose:
                #                     print(codes, type(codes))
                #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                #                     print(f"code len: {code_lens}")

                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(max_code_len, code_len)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    S_infer = self.semantic_codec.decode(codes)
                    target_lengths = torch.LongTensor([int(S_infer.shape[1] * 1.72 * duration_factor)]).to(codes.device)

                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                 ylens=target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]

                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                   torch.LongTensor([cat_condition.size(1)]).to(
                                                                       cond.device),
                                                                   ref_mel, style, None, diffusion_steps,
                                                                   inference_cfg_rate=inference_cfg_rate)
                    vc_target = vc_target[:, :, ref_mel.size(-1):]
                    s2mel_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    print(wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
                if stream_return:
                    yield wav.cpu()
                    if silence == None:
                        silence = self.interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
                    yield silence
        end_time = time.perf_counter()

        self._set_gr_progress(0.9, "saving audio...")
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        # 停止符采样偏早时，末尾可能停在话音中间
        # (index-tts/index-tts#247, #488, #523, #633)，详见 fade_out_tail()。
        wav = fade_out_tail(wav, sampling_rate)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            save_pcm_wav(output_path, wav, sampling_rate)
            print(">> wav file saved to:", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            yield (sampling_rate, wav_data)

    # ------------------------------------------------------------------
    # Batched inference: many texts, one reference voice.
    #
    # `infer` runs GPT generation for one text segment at a time. GPT decoding
    # is launch/CPU bound on a single sequence, so the GPU is mostly idle and a
    # step costs about the same for a few dozen sequences as for one.
    # `infer_batch` keeps every per-text step of `infer` (normalization,
    # segment splitting, sampling settings, s2mel, BigVGAN, interval silence,
    # saving) and only changes the GPT step: the segments of all texts are
    # sorted by length and decoded together in left-padded batches.
    # ------------------------------------------------------------------

    # Rough number of mel codes per text token, used only to size batches.
    # Measured on Chinese narration with this model: about 4.2 codes per text
    # token (25 codes per second of speech).
    BATCH_MEL_PER_TEXT_TOKEN = 4.6
    # Peak KV memory relative to one copy: during a beam step the old cache,
    # the concatenated new cache and the beam-reordered copy are alive at once
    # (measured about 2.4x on an RTX 5060 Ti).
    BATCH_KV_PEAK_FACTOR = 2.6

    @torch.no_grad()
    def _prepare_reference(self, spk_audio_prompt, emo_audio_prompt, verbose=False):
        """Reference conditioning for `infer_batch`.

        Same computation and the same cache attributes as `infer_generator`,
        so mixing `infer` and `infer_batch` calls keeps the cache valid.
        """
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

            audio_16k = torchaudio.transforms.Resample(sr, 16000)(self._load_and_cut_audio(spk_audio_prompt, 15, verbose)[0])
            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                    num_mel_bins=80,
                                                    dither=0,
                                                    sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = self.campplus_model(feat.unsqueeze(0))

            prompt_condition = self.s2mel.models['length_regulator'](
                spk_cond_emb,
                ylens=ref_target_lengths,
                n_quantizers=3,
                f0=None)[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"].to(self.device)
            emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        return spk_cond_emb, style, prompt_condition, ref_mel, emo_cond_emb

    def _prepare_text_segments(self, text, lang, max_text_tokens_per_segment, text_normalization, verbose=False):
        """Text front end of `infer_generator`: returns the segment strings and their token tensors."""
        lang_prefix = f'<|{lang.lower()}|> '

        text = self.text_process.clean_pattern.sub(lambda x: self.text_process.char_rep_map[x.group()], text)

        if text_normalization:
            if lang.lower() in ['zh', 'zhen', 'en']:
                text = self.text_process.normalize(text)
            elif lang.lower() in ['ja', 'es']:
                text = nemo_text_normalize(text, lang.lower())
            print(f'text after normalization: {text}')

        if lang.lower() in ['ja', 'zh', 'zhen', 'en']:
            text = text.lower()
        if lang.lower() == 'es':
            text = text.upper()
        text = apply_pronunciation_annotations(text)

        if lang.lower() == 'ja':
            text = self.ja_text_process.process_ja_text(text)
        text = re.sub(r'<\|([^|]+)\|>', lambda m: f'<|{m.group(1).upper()}|>', text)
        segments = self.split_text_by_tokens(text, max_text_tokens_per_segment, lang_prefix)
        segment_tokens = []
        for seg_text in segments:
            toks = self.tokenizer.encode(lang_prefix + seg_text, allowed_special='all')
            toks = torch.IntTensor(toks).unsqueeze(0).to(self.device)
            segment_tokens.append(F.pad(toks, (0, 1), value=1))
        if verbose:
            print("segments count:", len(segments))
            print(*segments, sep="\n")
        return segments, segment_tokens

    def _gpt_batch_budget_bytes(self, vram_budget_gb=None):
        """Bytes of VRAM the GPT KV cache of one batch may use at its peak."""
        if not (torch.cuda.is_available() and str(self.device).startswith("cuda")):
            return None
        dev = torch.device(self.device)
        free, _total = torch.cuda.mem_get_info(dev)
        allocated = torch.cuda.memory_allocated(dev)
        reusable = torch.cuda.memory_reserved(dev) - allocated
        available = free + reusable
        gib = 1024 ** 3
        if vram_budget_gb is None:
            # leave room for s2mel/BigVGAN work buffers and for other programs
            budget = available - 2 * gib
        else:
            budget = min(available - gib // 2, int(vram_budget_gb * gib) - allocated)
        return max(budget, 256 * 1024 ** 2)

    def _plan_gpt_batches(self, seg_keys, seg_lens, num_beams, max_mel_tokens, max_batch_segments, budget_bytes):
        """Group segments (already sorted longest first) into GPT batches.

        A batch decodes until its longest member stops, and every sequence in it
        holds a KV cache that long, so the cost of a batch is
        members * beams * longest_length."""
        elem = 2 if self.dtype is not None else 4
        kv_per_token = 2 * self.cfg.gpt.layers * self.cfg.gpt.model_dim * elem
        groups, cur, longest = [], [], 0
        for key, n in zip(seg_keys, seg_lens):
            est = n + 8 + min(max_mel_tokens, int(self.BATCH_MEL_PER_TEXT_TOKEN * n) + 64)
            if not cur:
                longest = est
            cost = (len(cur) + 1) * num_beams * longest * kv_per_token * self.BATCH_KV_PEAK_FACTOR
            if cur and (len(cur) >= max_batch_segments or (budget_bytes is not None and cost > budget_bytes)):
                groups.append(cur)
                cur, longest = [], est
            cur.append(key)
        if cur:
            groups.append(cur)
        return groups

    @torch.no_grad()
    def _gpt_generate_batch(self, seg_tokens, lang, spk_cond_emb, emo_cond_emb, emovec, style,
                            spk_audio_prompt, gen_kwargs):
        """Decode several text segments in one left-padded batch.

        Returns one (codes[1, n], hit_max_length) pair per segment, where codes
        stop before the first stop token, as `infer_generator` trims them."""
        device = torch.device(self.device)
        stop_text = self.cfg.gpt.stop_text_token
        b = len(seg_tokens)
        width = max(t.shape[1] for t in seg_tokens)
        # pad with the stop token: prepare_gpt_inputs strips it and left-pads the embeddings
        text_inputs = torch.full((b, width), stop_text, dtype=seg_tokens[0].dtype, device=device)
        for i, t in enumerate(seg_tokens):
            text_inputs[i, :t.shape[1]] = t[0]
        langs = torch.full((b,), lang_to_token(lang), dtype=torch.long, device=device)
        gen_kwargs = dict(gen_kwargs)
        max_mel_tokens = gen_kwargs.pop("max_generate_length")
        with torch.amp.autocast(device.type, enabled=self.dtype is not None, dtype=self.dtype):
            codes, _ = self.gpt.inference_speech(
                spk_cond_emb,
                text_inputs,
                langs,
                emo_cond_emb,
                cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=device),
                emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=device),
                emo_vec=emovec,
                campplus_embedding=style,
                wav=spk_audio_prompt,
                num_return_sequences=1,
                max_generate_length=max_mel_tokens,
                **gen_kwargs
            )
        out = []
        for i in range(b):
            row = codes[i]
            hit_max = bool(row[-1] != self.stop_mel_token)
            stops = (row == self.stop_mel_token).nonzero(as_tuple=False)
            n = stops[0].item() if stops.numel() > 0 else row.shape[0]
            out.append((row[:n].unsqueeze(0), hit_max))
        return out

    def _gpt_generate_batch_safe(self, seg_tokens, *args):
        """`_gpt_generate_batch`, splitting the batch in halves on CUDA out-of-memory."""
        try:
            return self._gpt_generate_batch(seg_tokens, *args)
        except torch.cuda.OutOfMemoryError:
            if len(seg_tokens) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(seg_tokens) // 2
            print(f">> GPT batch of {len(seg_tokens)} ran out of memory, retrying as {half} + {len(seg_tokens) - half}")
            return (self._gpt_generate_batch_safe(seg_tokens[:half], *args)
                    + self._gpt_generate_batch_safe(seg_tokens[half:], *args))

    @torch.no_grad()
    def _codes_to_wav(self, codes, prompt_condition, ref_mel, style, duration_factor):
        """s2mel + BigVGAN for one segment, exactly as in `infer_generator`. Returns (wav, s2mel_s, bigvgan_s)."""
        with torch.amp.autocast(codes.device.type, enabled=False):
            m_start_time = time.perf_counter()
            diffusion_steps = 25
            inference_cfg_rate = 0.7
            S_infer = self.semantic_codec.decode(codes)
            target_lengths = torch.LongTensor([int(S_infer.shape[1] * 1.72 * duration_factor)]).to(codes.device)
            cond = self.s2mel.models['length_regulator'](S_infer,
                                                         ylens=target_lengths,
                                                         n_quantizers=3,
                                                         f0=None)[0]
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                           torch.LongTensor([cat_condition.size(1)]).to(
                                                               cond.device),
                                                           ref_mel, style, None, diffusion_steps,
                                                           inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, ref_mel.size(-1):]
            s2mel_time = time.perf_counter() - m_start_time

            m_start_time = time.perf_counter()
            wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
            bigvgan_time = time.perf_counter() - m_start_time
            wav = wav.squeeze(1)
        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
        return wav.cpu(), s2mel_time, bigvgan_time

    # s2mel batches are limited by the DiT attention size: segments * frames^2
    BATCH_S2MEL_FRAMES2 = 12_000_000
    BATCH_S2MEL_MAX = 8

    @torch.no_grad()
    def _codes_to_mel_batch(self, codes_list, prompt_condition, ref_mel, style, duration_factor):
        """s2mel for several segments at once.

        Per segment, the semantic decode and length regulator run exactly as in
        `infer_generator`; the 25-step flow-matching solve (same Euler steps and
        CFG as `cfm.inference`) runs on a zero-padded batch with the padding
        masked by `x_lens`. Returns one mel [1, 80, T] per segment."""
        diffusion_steps = 25
        inference_cfg_rate = 0.7
        device = codes_list[0].device
        with torch.amp.autocast(device.type, enabled=False):
            conds = []
            for codes in codes_list:
                S_infer = self.semantic_codec.decode(codes)
                target_lengths = torch.LongTensor([int(S_infer.shape[1] * 1.72 * duration_factor)]).to(device)
                cond = self.s2mel.models['length_regulator'](S_infer,
                                                             ylens=target_lengths,
                                                             n_quantizers=3,
                                                             f0=None)[0]
                conds.append(torch.cat([prompt_condition, cond], dim=1)[0])
            b = len(conds)
            x_lens = torch.LongTensor([c.shape[0] for c in conds]).to(device)
            mu = pad_sequence(conds, batch_first=True)
            cfm = self.s2mel.models['cfm']
            estimator = cfm.estimator
            prompt_len = ref_mel.size(-1)
            x = torch.randn([b, cfm.in_channels, mu.size(1)], device=device)
            prompt_x = torch.zeros_like(x)
            prompt_x[..., :prompt_len] = ref_mel[..., :prompt_len]
            x[..., :prompt_len] = 0
            if cfm.zero_prompt_speech_token:
                mu[..., :prompt_len] = 0
            style_b = style.expand(b, -1)
            t_span = torch.linspace(0, 1, diffusion_steps + 1, device=device)
            t = t_span[0]
            if inference_cfg_rate > 0:
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style_b, torch.zeros_like(style_b)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_lens = torch.cat([x_lens, x_lens], dim=0)

            # The WaveNet head masks its activations after every layer but not its
            # input; zero the padded frames there so that every segment sees the
            # same zero boundary as when it is decoded alone.
            def _mask_wavenet_input(_module, args):
                return (args[0] * args[1],) + tuple(args[1:])

            hook = estimator.wavenet.register_forward_pre_hook(_mask_wavenet_input)
            try:
                for step in range(1, len(t_span)):
                    dt = t_span[step] - t_span[step - 1]
                    if inference_cfg_rate > 0:
                        stacked_dphi_dt = estimator(
                            torch.cat([x, x], dim=0), stacked_prompt_x, stacked_lens,
                            t.expand(2 * b), stacked_style, stacked_mu,
                        )
                        dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                        dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
                    else:
                        dphi_dt = estimator(x, prompt_x, x_lens, t.expand(b), style_b, mu)
                    x = x + dt * dphi_dt
                    t = t + dt
                    x[:, :, :prompt_len] = 0
            finally:
                hook.remove()
        return [x[i:i + 1, :, prompt_len:int(x_lens[i])] for i in range(b)]

    def _codes_to_mel_batch_safe(self, codes_list, *args):
        try:
            return self._codes_to_mel_batch(codes_list, *args)
        except torch.cuda.OutOfMemoryError:
            if len(codes_list) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(codes_list) // 2
            return (self._codes_to_mel_batch_safe(codes_list[:half], *args)
                    + self._codes_to_mel_batch_safe(codes_list[half:], *args))

    @torch.no_grad()
    def _mel_to_wav(self, vc_target):
        with torch.amp.autocast(vc_target.device.type, enabled=False):
            wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
            wav = wav.squeeze(1)
        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
        return wav.cpu()

    def infer_batch(self, spk_audio_prompt, texts, output_paths=None, lang="ZH",
                    emo_audio_prompt=None, emo_alpha=1.0, emo_vector=None, use_random=False,
                    interval_silence=200, verbose=False, max_text_tokens_per_segment=120,
                    duration_factor=1.0, text_normalization=True,
                    max_batch_segments=32, vram_budget_gb=None, batch_s2mel=False, **generation_kwargs):
        """Synthesize many texts with one reference voice, batching GPT generation.

        Gives the same per-text result as calling `infer` once per text (same
        text front end, segment splitting, sampling settings, s2mel, BigVGAN,
        interval silence and saving); only the GPT decoding of the segments is
        done in batches. Sampling is random, so the audio is not bit-identical
        to `infer`, just as two `infer` calls are not.

        Args:
            texts: list of texts.
            output_paths: list of wav paths, one per text, or None to return
                ``(sampling_rate, int16 numpy array)`` tuples.
            max_batch_segments: most text segments decoded together (each
                segment uses ``num_beams`` sequences).
            vram_budget_gb: VRAM this process may use while decoding; by
                default, what is free on the device minus 2 GiB.
            batch_s2mel: also run the s2mel diffusion of several segments
                together (padded and masked). Off by default: s2mel runs in
                fp32 and already keeps the GPU busy, so padding made it slower
                (57.5 s vs 32.7 s per episode on an RTX 5060 Ti).
            Other arguments: as in `infer`. ``use_emo_text`` is not supported.

        Returns:
            list with one entry per text: its output path, or a
            ``(sampling_rate, wav)`` tuple when ``output_paths`` is None.
        """
        if isinstance(texts, str):
            texts = [texts]
        if output_paths is not None and len(output_paths) != len(texts):
            raise ValueError(f"got {len(texts)} texts but {len(output_paths)} output paths")
        if generation_kwargs.pop("use_emo_text", False) or generation_kwargs.pop("emo_text", None) is not None:
            raise ValueError("infer_batch does not support use_emo_text; call infer() per text instead")
        generation_kwargs.pop("stream_return", None)

        if self.low_vram:
            # keep the low-VRAM chunking path of `infer` unchanged
            return [self.infer(spk_audio_prompt, t, p, lang, emo_audio_prompt=emo_audio_prompt,
                               emo_alpha=emo_alpha, emo_vector=emo_vector, use_random=use_random,
                               interval_silence=interval_silence, verbose=verbose,
                               max_text_tokens_per_segment=max_text_tokens_per_segment,
                               duration_factor=duration_factor, text_normalization=text_normalization,
                               **dict(generation_kwargs))
                    for t, p in zip(texts, output_paths or [None] * len(texts))]

        print(f">> starting batched inference of {len(texts)} texts...")
        start_time = time.perf_counter()

        if emo_vector is not None:
            emo_audio_prompt = None
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")
        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0

        spk_cond_emb, style, prompt_condition, ref_mel, emo_cond_emb = self._prepare_reference(
            spk_audio_prompt, emo_audio_prompt, verbose)

        device = torch.device(self.device)
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=self.dtype is not None, dtype=self.dtype):
                emovec = self.gpt.merge_emovec(
                    spk_cond_emb,
                    emo_cond_emb,
                    torch.tensor([spk_cond_emb.shape[-1]], device=device),
                    torch.tensor([emo_cond_emb.shape[-1]], device=device),
                    alpha=emo_alpha
                )
                if emo_vector is not None:
                    weight_vector = torch.tensor(emo_vector, device=self.device)
                    if use_random:
                        random_index = [random.randint(0, x - 1) for x in self.emo_num]
                    else:
                        random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
                    emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
                    emo_matrix = torch.cat(emo_matrix, 0)
                    emovec_mat = torch.sum(weight_vector.unsqueeze(1) * emo_matrix, 0).unsqueeze(0)
                    emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

        t_ref = time.perf_counter()
        # text front end, per text, as infer does
        seg_tokens = {}      # (text index, segment index) -> token tensor [1, L]
        seg_count = []
        for ti, text in enumerate(texts):
            _segments, tokens = self._prepare_text_segments(
                text, lang, max_text_tokens_per_segment, text_normalization, verbose)
            seg_count.append(len(tokens))
            for si, tok in enumerate(tokens):
                seg_tokens[(ti, si)] = tok

        gen_kwargs = dict(
            do_sample=True,
            top_p=generation_kwargs.pop("top_p", 0.8),
            top_k=generation_kwargs.pop("top_k", 30),
            temperature=generation_kwargs.pop("temperature", 0.8),
            length_penalty=generation_kwargs.pop("length_penalty", 0.0),
            num_beams=generation_kwargs.pop("num_beams", 3),
            repetition_penalty=generation_kwargs.pop("repetition_penalty", 10.0),
            max_generate_length=generation_kwargs.pop("max_mel_tokens", 1500),
        )
        generation_kwargs.pop("do_sample", None)  # infer always samples
        gen_kwargs.update(generation_kwargs)
        max_mel_tokens = gen_kwargs["max_generate_length"]
        sampling_rate = 22050

        t_front = time.perf_counter()
        keys = sorted(seg_tokens, key=lambda k: -seg_tokens[k].shape[1])
        budget = self._gpt_batch_budget_bytes(vram_budget_gb)
        groups = self._plan_gpt_batches(keys, [seg_tokens[k].shape[1] for k in keys],
                                        gen_kwargs["num_beams"], max_mel_tokens, max_batch_segments, budget)
        print(f">> {len(keys)} segments in {len(groups)} GPT batches: {[len(g) for g in groups]}"
              + (f" (KV budget {budget / 1024 ** 3:.1f} GiB)" if budget is not None else ""))

        seg_wavs = [[None] * n for n in seg_count]
        remaining = list(seg_count)
        results = [None] * len(texts)
        lengths = [0.0] * len(texts)
        gpt_gen_time = s2mel_time = bigvgan_time = 0.0
        warned = set()

        def finish(ti):
            wavs = self.insert_interval_silence(seg_wavs[ti], sampling_rate=sampling_rate,
                                                interval_silence=interval_silence)
            wav = torch.cat(wavs, dim=1)
            seg_wavs[ti] = None
            lengths[ti] = wav.shape[-1] / sampling_rate
            if output_paths is not None:
                path = output_paths[ti]
                if os.path.isfile(path):
                    os.remove(path)
                    print(">> remove old wav file:", path)
                if os.path.dirname(path) != "":
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                save_pcm_wav(path, wav, sampling_rate)
                print(f">> wav file saved to: {path} ({wav.shape[-1] / sampling_rate:.2f} s)")
                results[ti] = path
            else:
                results[ti] = (sampling_rate, wav.type(torch.int16).numpy().T)

        for gi, group in enumerate(groups):
            self._set_gr_progress(0.1 + 0.85 * gi / len(groups), f"speech synthesis batch {gi + 1}/{len(groups)}...")
            m_start_time = time.perf_counter()
            outs = self._gpt_generate_batch_safe(
                [seg_tokens[k] for k in group], lang, spk_cond_emb, emo_cond_emb, emovec, style,
                spk_audio_prompt, gen_kwargs)
            batch_time = time.perf_counter() - m_start_time
            gpt_gen_time += batch_time
            if verbose:
                mem = ""
                if torch.cuda.is_available() and device.type == "cuda":
                    mem = f", peak allocated so far {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GiB"
                print(f">> GPT batch {gi + 1}/{len(groups)}: {len(group)} segments, "
                      f"{max(c.shape[1] for c, _ in outs)} steps, {batch_time:.2f} s{mem}")
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()   # drop the fragmented KV-cache blocks before s2mel
            for key, (codes, hit_max) in zip(group, outs):
                ti, si = key
                if hit_max and ti not in warned:
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {seg_tokens[key].shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    warned.add(ti)
                if verbose:
                    print(f"text {ti} segment {si}: {seg_tokens[key].shape[1]} text tokens -> {codes.shape[1]} codes")

            if batch_s2mel:
                # longest first; s2mel frames are about codes * 2 * 1.72 * duration_factor, plus the prompt
                order = sorted(range(len(group)), key=lambda j: -outs[j][0].shape[1])
                est = lambda j: ref_mel.size(-1) + int(outs[j][0].shape[1] * 2 * 1.72 * duration_factor) + 8
                s2_groups, cur = [], []
                for j in order:
                    if cur and (len(cur) >= self.BATCH_S2MEL_MAX
                                or (len(cur) + 1) * est(cur[0]) ** 2 > self.BATCH_S2MEL_FRAMES2):
                        s2_groups.append(cur)
                        cur = []
                    cur.append(j)
                if cur:
                    s2_groups.append(cur)
            else:
                s2_groups = [[j] for j in range(len(group))]

            for s2 in s2_groups:
                m_start_time = time.perf_counter()
                if batch_s2mel:
                    mels = self._codes_to_mel_batch_safe([outs[j][0] for j in s2], prompt_condition, ref_mel,
                                                         style, duration_factor)
                    s2mel_time += time.perf_counter() - m_start_time
                    pieces = []
                    for mel in mels:
                        m_start_time = time.perf_counter()
                        pieces.append(self._mel_to_wav(mel))
                        bigvgan_time += time.perf_counter() - m_start_time
                    if verbose:
                        print(f">> s2mel batch of {len(s2)} segments")
                else:
                    wav, t_s2mel, t_bigvgan = self._codes_to_wav(outs[s2[0]][0], prompt_condition, ref_mel,
                                                                 style, duration_factor)
                    s2mel_time += t_s2mel
                    bigvgan_time += t_bigvgan
                    pieces = [wav]
                for j, wav in zip(s2, pieces):
                    ti, si = group[j]
                    seg_wavs[ti][si] = wav
                    remaining[ti] -= 1
                    if remaining[ti] == 0:
                        finish(ti)

        end_time = time.perf_counter()
        audio_len = sum(lengths)
        print(f">> reference+emotion: {t_ref - start_time:.2f} s, text front end: {t_front - t_ref:.2f} s, "
              f"batches (GPT+s2mel+BigVGAN+save): {end_time - t_front:.2f} s")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {audio_len:.2f} seconds")
        if audio_len > 0:
            print(f">> RTF: {(end_time - start_time) / audio_len:.4f}")
        return results


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index

class QwenEmotion:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype="float16",  # "auto"
            device_map="auto"
        )
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            # TODO: the "低落" (melancholic) emotion will always be mapped to
            # "悲伤" (sad) by QwenEmotion's text analysis. it doesn't know the
            # difference between those emotions even if user writes exact words.
            # SEE: `self.melancholic_words` for current workaround.
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        self.melancholic_words = {
            # emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
            # to become "低落" (melancholic) instead, to fix limitations mentioned above.
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp_score(self, value):
        return max(self.min_score, min(self.max_score, value))

    def normalize_content(self, content):
        if isinstance(content, dict):
            normalized = dict(content)
        else:
            normalized = {}

        def label_to_cn_key(value):
            if not isinstance(value, str):
                return None

            value = value.strip()
            if value in self.cn_key_to_en:
                return value

            value_lower = value.lower()
            for cn_key, en_key in self.cn_key_to_en.items():
                if value_lower == en_key:
                    return cn_key
            return None

        detected_key = label_to_cn_key(content) if isinstance(content, str) else None
        if detected_key is None:
            for alias in ("emotion", "emotion_label", "label", "情感", "情绪"):
                detected_key = label_to_cn_key(normalized.get(alias))
                if detected_key is not None:
                    break
        if detected_key is not None and all(key not in normalized for key in self.desired_vector_order):
            normalized[detected_key] = 1.0

        for cn_key in self.desired_vector_order:
            detected_key = label_to_cn_key(normalized.get(cn_key))
            if detected_key is not None:
                normalized[cn_key] = 1.0 if detected_key == cn_key else 0.0
                if detected_key != cn_key:
                    normalized[detected_key] = 1.0

        return normalized

    def convert(self, content):
        # generate emotion vector dictionary:
        # - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # - convert Chinese keys to English
        # - clamp all values to the allowed min/max range
        # - use 0.0 for any values that were missing in `content`
        content = self.normalize_content(content)
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }

        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input):
        start = time.time()
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        # decode the JSON emotion detections as a dictionary
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            # print(">> parsing QwenEmotion response", content)
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }
            # print(">> dict result", content)

        # workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        # if we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # to encode the "sad" emotion as "melancholic" (instead of sadness).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            # print(">> before vec swap", content)
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)
            # print(">>  after vec swap", content)

        return self.convert(content)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="checkpoints/config.yaml")
    parser.add_argument("--model_dir", type=str, default="checkpoints")
    parser.add_argument("--prompt_wav", type=str, default="examples/voice_01.wav")
    parser.add_argument("--text", type=str, default="欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。")
    parser.add_argument("--lang", type=str, default="ZH")
    parser.add_argument("--output", type=str, default="gen.wav")
    parser.add_argument("--text_normalization", action="store_true", default=True)
    args = parser.parse_args()

    tts = IndexTTS2(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        use_bf16=True,
        use_cuda_kernel=False,
        use_torch_compile=False,
        use_qwen_emo=True,
    )

    tts.infer(spk_audio_prompt=args.prompt_wav, text=args.text, lang=args.lang, output_path=args.output, verbose=True, text_normalization=args.text_normalization)