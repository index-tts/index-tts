import os
import re
import time
from subprocess import CalledProcessError

import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.common import tokenize_by_CJK_char
from indextts.utils.front import TextNormalizer

import torch._inductor.config
import torch.compiler  # 添加这一行

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True

# 在文件顶部或主入口点添加
if torch.backends.mps.is_available():
    # 使用最佳可用的 SDPA 实现
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # 防止内存碎片
    # PyTorch 2.0+ 会自动为 MPS 选择最优的 SDPA 实现


class IndexTTS:
    def __init__(
        self,
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        is_fp16=True,
        device=None,
        use_cuda_kernel=None,
        compile=False,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = (
                use_cuda_kernel is not None
                and use_cuda_kernel
                and device.startswith("cuda")
            )
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif torch.mps.is_available():
            self.device = "mps"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.compile = compile

        # Comment-off to load the VQ-VAE model for debugging tokenizer
        #   https://github.com/index-tts/index-tts/issues/34
        #
        # from indextts.vqvae.xtts_dvae import DiscreteVAE
        # self.dvae = DiscreteVAE(**self.cfg.vqvae)
        # self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        # load_checkpoint(self.dvae, self.dvae_path)
        # self.dvae = self.dvae.to(self.device)
        # if self.is_fp16:
        #     self.dvae.eval().half()
        # else:
        #     self.dvae.eval()
        # print(">> vqvae weights restored from:", self.dvae_path)
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            try:
                import deepspeed

                use_deepspeed = True
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed加载失败，回退到标准推理: {e}")

            self.gpt.post_init_gpt2_config(
                use_deepspeed=use_deepspeed, kv_cache=True, half=True
            )
        else:
            self.gpt.post_init_gpt2_config(
                use_deepspeed=False, kv_cache=False, half=False
            )

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load

                anti_alias_activation_cuda = load.load()
                print(
                    ">> Preload custom CUDA kernel for BigVGAN",
                    anti_alias_activation_cuda,
                )
            except:
                print(
                    ">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch."
                )
                self.use_cuda_kernel = False
        self.bigvgan = BigVGAN(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.bpe_path)
        print(">> bpe model loaded from:", self.bpe_path)
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        # 缓存参考音频mel：
        self.cache_audio_prompt = []
        self.cache_cond_mel = {}
        # 进度引用显示（可选）
        self.gr_progress = None

    def preprocess_text(self, text):
        # chinese_punctuation = "，。！？；：“”‘’（）【】《》"
        # english_punctuation = ",.!?;:\"\"''()[]<>"
        #
        # # 创建一个映射字典
        # punctuation_map = str.maketrans(chinese_punctuation, english_punctuation)

        # 使用translate方法替换标点符号
        # return text.translate(punctuation_map)
        return self.normalizer.infer(text)

    # @torch.compile(
    #     fullgraph=True,  # 整个图编译，提供最大优化
    #     backend=(
    #         "inductor" if torch.cuda.is_available() else "aot_eager"
    #     ),  # 根据设备选择后端
    #     mode=(
    #         "reduce-overhead" if torch.cuda.is_available() else None
    #     ),  # GPU上使用减少开销模式
    # )
    def remove_long_silence(self, codes, silent_token=52, max_consecutive=30):
        """
        移除音频中的长静音片段，使用纯张量操作实现以支持torch.compile

        Args:
            codes: 音频编码张量
            silent_token: 表示静音的token id
            max_consecutive: 最大连续静音token数量

        Returns:
            修改后的音频编码和长度
        """
        # 1. 处理特殊情况
        if codes.ndim == 0:
            # 单个标量值，转为一维张量并返回
            scalar_code = codes.unsqueeze(0).unsqueeze(0)
            return scalar_code, torch.ones(1, device=codes.device, dtype=torch.long)

        if codes.ndim == 1:
            # 如果是一维张量，转为二维
            codes = codes.unsqueeze(0)

        if codes.shape[1] == 0 or codes.numel() == 0:
            # 空序列的情况
            return codes, torch.zeros(
                codes.shape[0], device=codes.device, dtype=torch.long
            )

        # 2. 批量处理每个序列
        batch_size = codes.shape[0]
        device = codes.device
        max_len = codes.shape[1]

        # 初始化输出张量
        filtered_codes_list = []
        filtered_lengths = []

        # 对每个样本单独处理
        for b in range(batch_size):
            # 获取当前样本
            seq = codes[b]
            seq_len = torch.sum(torch.ones_like(seq, dtype=torch.long))

            # 找出所有静音帧的位置
            is_silent = seq == silent_token

            if not torch.any(is_silent):
                # 没有静音帧，直接添加
                filtered_codes_list.append(seq)
                filtered_lengths.append(seq_len)
                continue

            # 检测连续的静音段
            # 1表示当前位置是新段的开始
            # 0表示当前位置与前一位置在同一段中
            if seq_len > 1:
                # 在序列长度大于1的情况下
                segment_starts = torch.cat(
                    [
                        torch.ones(1, device=device, dtype=torch.bool),
                        (seq[1:] == silent_token) & (seq[:-1] != silent_token),
                    ]
                )

                # 使用cumsum来标识每个连续段
                silent_segments = torch.zeros_like(seq, dtype=torch.long)
                silent_segments[is_silent & segment_starts] = 1
                segment_ids = torch.cumsum(silent_segments, dim=0)

                # 计算每个段的长度
                segment_lengths = torch.zeros(
                    segment_ids.max() + 1 if segment_ids.numel() > 0 else 1,
                    device=device,
                    dtype=torch.long,
                )

                if torch.any(is_silent):
                    # 只有在有静音时才进行segment_ids的统计
                    segment_ids_silent = segment_ids[is_silent]
                    for i in range(segment_ids_silent.numel()):
                        segment_lengths[segment_ids_silent[i]] += 1

                # 创建一个掩码，标记需要保留的位置（所有非静音位置）
                keep_mask = ~is_silent

                # 对于每个静音段，如果长度超过max_consecutive，仅保留前max_consecutive个
                for seg_id in range(1, segment_lengths.numel()):
                    if segment_lengths[seg_id] > max_consecutive:
                        # 找出此segment_id的所有位置
                        seg_positions = (segment_ids == seg_id) & is_silent
                        # 计算每个位置在此段中的偏移
                        if torch.any(seg_positions):
                            pos_indices = torch.arange(seq_len, device=device)
                            first_pos = torch.min(pos_indices[seg_positions])
                            positions_to_keep = (
                                torch.arange(max_consecutive, device=device) + first_pos
                            )
                            positions_to_keep = positions_to_keep[
                                positions_to_keep < seq_len
                            ]

                            # 更新keep_mask，将segment中在max_consecutive范围内的位置设为保留
                            for pos in positions_to_keep:
                                keep_mask[pos] = True

                # 应用掩码
                filtered_seq = seq[keep_mask]

            else:
                # 序列长度为1的特殊情况，直接保留
                filtered_seq = seq

            filtered_codes_list.append(filtered_seq)
            filtered_lengths.append(filtered_seq.numel())

        # 3. 对齐和填充所有序列到最大长度
        max_filtered_len = max(filtered_lengths) if filtered_lengths else 1
        padded_codes = []

        for seq, length in zip(filtered_codes_list, filtered_lengths):
            if length < max_filtered_len:
                # 用0填充到最大长度
                padding = torch.zeros(
                    max_filtered_len - length, device=device, dtype=seq.dtype
                )
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_codes.append(padded_seq)

        # 4. 堆叠所有序列
        result_codes = torch.stack(padded_codes, dim=0)
        result_lengths = torch.tensor(filtered_lengths, device=device, dtype=torch.long)

        return result_codes, result_lengths

    def split_sentences(self, text):
        """
        Split the text into sentences based on punctuation marks.
        """
        # 匹配标点符号（包括中英文标点）
        pattern = r"(?<=[.!?;。！？；])\s*"
        sentences = re.split(pattern, text)
        # 过滤掉空字符串和仅包含标点符号的字符串
        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip() and sentence.strip() not in {"'", ".", ","}
        ]

    def bucket_sentences(self, sentences, enable):
        """
        Sentence data bucketing
        """
        max_len = max(len(s) for s in sentences)
        half = max_len // 2
        outputs = [[], []]
        for idx, sent in enumerate(sentences):
            if enable == False or len(sent) <= half:
                outputs[0].append({"idx": idx, "sent": sent})
            else:
                outputs[1].append({"idx": idx, "sent": sent})
        return [item for item in outputs if item]

    def pad_tokens_cat(self, tokens):
        if len(tokens) <= 1:
            return tokens[-1]
        max_len = max(t.size(1) for t in tokens)
        outputs = []
        for tensor in tokens:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(
                    tensor, (0, n), value=self.cfg.gpt.stop_text_token
                )
                tensor = torch.nn.functional.pad(
                    tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token
                )
            tensor = tensor[:, :max_len]
            outputs.append(tensor)
        tokens = torch.cat(outputs, dim=0)
        return tokens

    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    # 添加新方法到IndexTTS类中

    def process_audio_prompts(
        self,
        audio_prompts,
        prompt_id="",
        verbose=False,
        fusion_method="average",
        weights=None,
    ):
        """
        处理参考音频，提取并可能融合多个音频的Mel特征

        Args:
            audio_prompts: 单个音频路径字符串或多个音频路径的列表
            prompt_id: 缓存标识符，如未提供则使用第一个音频路径
            verbose: 是否显示详细信息
            fusion_method: 融合方法，"average"或"weighted"
            weights: 多个音频的权重列表（用于weighted融合）

        Returns:
            tuple: (cond_mel, cond_mel_frame) - 条件Mel特征及其帧数
        """
        # 确保audio_prompts是列表格式
        if isinstance(audio_prompts, str):
            audio_prompts = [audio_prompts]

        # 确保有prompt_id
        if not prompt_id:
            prompt_id = audio_prompts[0]  # 使用第一个音频路径作为默认ID

        # 检查缓存
        cache_key = f"{prompt_id}_{fusion_method}"
        if (
            self.cache_cond_mel.get(cache_key, None) is None
            or cache_key not in self.cache_audio_prompt
        ):
            if verbose:
                print(f"处理音频样本: {audio_prompts}")

            # 处理所有音频文件
            processed_audios = []
            for audio_path in audio_prompts:
                audio, sr = torchaudio.load(audio_path)
                audio = torch.mean(audio, dim=0, keepdim=True)
                if audio.shape[0] > 1:
                    audio = audio[0].unsqueeze(0)
                audio = torchaudio.transforms.Resample(sr, 24000)(audio)
                processed_audios.append(audio)

            # 使用MelSpectrogramFeatures同时处理所有音频并融合
            if len(processed_audios) == 1:
                cond_mel = MelSpectrogramFeatures()(processed_audios[0]).to(self.device)
            else:
                # 使用融合功能处理多个音频
                cond_mel = MelSpectrogramFeatures()(
                    processed_audios, weights=weights, fusion_method=fusion_method
                ).to(self.device)

            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)
                if len(processed_audios) > 1:
                    print(f"融合方法: {fusion_method}")
                    if weights:
                        print(f"融合权重: {weights}")

            # 更新缓存
            self.cache_audio_prompt.append(cache_key)
            self.cache_cond_mel[cache_key] = cond_mel
        else:
            # 使用缓存数据
            cond_mel = self.cache_cond_mel.get(cache_key, None)
            assert cond_mel is not None, f"cache_cond_mel: {cache_key} is None!!!"
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"使用缓存的音频特征: {cache_key}")

        return cond_mel, cond_mel_frame

    def process_text_to_tokens(
        self,
        sentences,
        bucket_enable=True,
        verbose=False,
        progress_value=0.1,
        progress_desc="text processing...",
    ):
        """
        处理文本到token序列的函数

        Args:
            sentences: split_sentences处理后的句子列表
            bucket_enable: 是否启用桶处理，默认True
            verbose: 是否显示详细日志，默认False
            progress_value: 进度条值，默认0.1
            progress_desc: 进度描述，默认"text processing..."

        Returns:
            list: 包含所有文本token的列表
        """
        all_text_tokens = []
        self._set_gr_progress(progress_value, progress_desc)

        # 对句子进行分桶处理
        all_sentences = self.bucket_sentences(sentences, enable=bucket_enable)

        for sentences in all_sentences:
            temp_tokens = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                cleand_text = tokenize_by_CJK_char(sent)
                if verbose:
                    print("cleand_text:", cleand_text)

                text_tokens = torch.tensor(
                    self.tokenizer.EncodeAsIds(cleand_text),
                    dtype=torch.int32,
                    device=self.device,
                ).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(
                        f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}"
                    )
                    # debug tokenizer
                    text_token_syms = self.tokenizer.IdToPiece(text_tokens[0].tolist())
                    print(text_token_syms)

                temp_tokens.append(text_tokens)

        return all_text_tokens, all_sentences

    # 快速推理：对于“多句长文本”，可实现至少 2~10 倍以上的速度提升~ （First modified by sunnyboxs 2025-04-16）
    def infer_fast(
        self,
        audio_prompt,
        text,
        output_path,
        verbose=False,
        prompt_id="",
        fusion_method="average",
        weights=None,
        no_chunk=False,
    ):
        print(">> start fast inference...")
        self._set_gr_progress(0, "start fast inference...")
        if not prompt_id:
            prompt_id = audio_prompt
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()
        normalized_text = self.preprocess_text(text)
        print(f"normalized text:{normalized_text}")

        # 使用新方法处理参考音频
        auto_conditioning, cond_mel_frame = self.process_audio_prompts(
            audio_prompt,
            prompt_id=prompt_id,
            verbose=verbose,
            fusion_method=fusion_method,
            weights=weights,
        )

        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # text_tokens
        sentences = self.split_sentences(normalized_text)
        if verbose:
            print("sentences:", sentences)

        top_p = 0.8
        top_k = 30
        temperature = 1.0
        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 3
        repetition_penalty = 10.0
        max_mel_tokens = 600
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # text processing
        all_text_tokens = []
        self._set_gr_progress(0.1, "text processing...")
        bucket_enable = True  # 预分桶开关，优先保证质量=True。优先保证速度=False。
        all_sentences = self.bucket_sentences(sentences, enable=bucket_enable)
        for sentences in all_sentences:
            temp_tokens = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                cleand_text = tokenize_by_CJK_char(sent)
                if verbose:
                    print("cleand_text:", cleand_text)

                text_tokens = torch.tensor(
                    self.tokenizer.EncodeAsIds(cleand_text),
                    dtype=torch.int32,
                    device=self.device,
                ).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(
                        f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}"
                    )
                    # debug tokenizer
                    text_token_syms = self.tokenizer.IdToPiece(text_tokens[0].tolist())
                    print(text_token_syms)

                temp_tokens.append(text_tokens)

        # Sequential processing of bucketing data
        all_batch_num = 0
        all_batch_codes = []
        for item_tokens in all_text_tokens:
            batch_num = len(item_tokens)
            batch_text_tokens = self.pad_tokens_cat(item_tokens)
            batch_cond_mel_lengths = torch.cat([cond_mel_lengths] * batch_num, dim=0)
            batch_auto_conditioning = torch.cat([auto_conditioning] * batch_num, dim=0)
            all_batch_num += batch_num

            # gpt speech
            self._set_gr_progress(0.2, "gpt inference speech...")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=self.device,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    # 检查是否在 CUDA 上下文中
                    device_type = self.device  # 'cuda' 或 'cpu'

                    # 获取当前的 autocast 状态
                    # 注意：这是正确的方法，我们直接使用 is_autocast_enabled 函数
                    is_enabled = torch.is_autocast_enabled(self.device)
                    # 获取当前的 autocast dtype
                    if is_enabled:
                        current_dtype = torch.get_autocast_dtype(self.device)
                    else:
                        # 默认值
                        current_dtype = torch.float32

                    print(
                        f"当前 autocast 状态: enabled={is_enabled}, dtype={current_dtype}, device_type={device_type}"
                    )
                    # 这里compile会拉慢速度，所以不使用compile
                    inference_fn = self.gpt.inference_speech
                    temp_codes = inference_fn(
                        batch_auto_conditioning,
                        batch_text_tokens,
                        cond_mel_lengths=batch_cond_mel_lengths,
                        # text_lengths=text_len,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                    )

                    if "cuda" in str(self.device) and self.compile:
                        temp_codes = temp_codes.clone()

                    all_batch_codes.append(temp_codes)
            gpt_gen_time += time.perf_counter() - m_start_time

        # gpt latent
        self._set_gr_progress(0.5, "gpt inference latents...")
        all_idxs = []
        all_latents = []
        for batch_codes, batch_tokens, batch_sentences in zip(
            all_batch_codes, all_text_tokens, all_sentences
        ):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]  # [x]
                codes = codes[codes != self.cfg.gpt.stop_mel_token]
                codes, _ = torch.unique_consecutive(codes, return_inverse=True)
                codes = codes.unsqueeze(0)  # [x] -> [1, x]
                code_lens = torch.tensor(
                    [codes.shape[-1]], device=codes.device, dtype=codes.dtype
                )
                codes, code_lens = self.remove_long_silence(
                    codes, silent_token=52, max_consecutive=30
                )
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast(
                        self.device, enabled=self.dtype is not None, dtype=self.dtype
                    ):
                        # 在调用optimized_forward前添加此行
                        if "cuda" in str(self.device) and self.compile:
                            torch.compiler.cudagraph_mark_step_begin()

                        forward_fn = (
                            self.gpt.optimized_forward
                            if self.compile
                            else self.gpt.forward
                        )
                        latent = forward_fn(
                            auto_conditioning,
                            text_tokens,
                            torch.tensor(
                                [text_tokens.shape[-1]], device=text_tokens.device
                            ),
                            codes,
                            code_lens * self.gpt.mel_length_compression,
                            cond_mel_lengths=torch.tensor(
                                [auto_conditioning.shape[-1]], device=text_tokens.device
                            ),
                            return_latent=True,
                            clip_inputs=False,
                        )
                        if "cuda" in str(self.device) and self.compile:
                            latent = latent.clone()

                        gpt_forward_time += time.perf_counter() - m_start_time
                        all_latents.append(latent)

        # bigvgan chunk

        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        latent_length = len(all_latents)
        # 直接连接所有latents
        latent = torch.cat(all_latents, dim=1)

        # bigvgan chunk decode
        self._set_gr_progress(0.7, "bigvgan decode...")
        tqdm_progress = tqdm(total=latent_length, desc="bigvgan")
        if no_chunk:
            all_latents = None  # 释放内存
            # 将所有bigvgan的解码合并到一起进行
            with torch.no_grad():
                with torch.amp.autocast(
                    self.device, enabled=self.dtype is not None, dtype=self.dtype
                ):
                    if "cuda" in str(self.device) and self.compile:
                        torch.compiler.cudagraph_mark_step_begin()

                    m_start_time = time.perf_counter()
                    bigvgan_forward_fn = (
                        self.bigvgan.compile_forward
                        if self.compile
                        else self.bigvgan.forward
                    )
                    mel_ref = auto_conditioning.transpose(1, 2)
                    if verbose:
                        print(
                            f"latent shape: {latent.shape}, mel_ref shape: {mel_ref.shape}"
                        )

                    wav, _ = bigvgan_forward_fn(latent, mel_ref)

                    if "cuda" in str(self.device) and self.compile:
                        wav = wav.clone()

                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1).detach().cpu()  # 立刻转移到cpu 来减少 显存消耗
            tqdm_progress.update(1)
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs = [wav]  # 使用单个波形而不是多个
        else:
            chunk_size = 2
            chunk_latents = [
                all_latents[i : i + chunk_size] for i in range(0, len(all_latents), chunk_size)
            ]
            chunk_length = len(chunk_latents)
            latent_length = len(all_latents)
            all_latents = None
            for items in chunk_latents:
                tqdm_progress.update(len(items))
                latent = torch.cat(items, dim=1)
                with torch.no_grad():
                    with torch.amp.autocast(
                        self.device, enabled=self.dtype is not None, dtype=self.dtype
                    ):
                        # 在调用optimized_forward前添加此行
                        if "cuda" in str(self.device) and self.compile:
                            torch.compiler.cudagraph_mark_step_begin()

                        m_start_time = time.perf_counter()
                        bigvgan_forward_fn = (
                            self.bigvgan.compile_forward
                            if self.compile
                            else self.bigvgan.forward
                        )
                        mel_ref = auto_conditioning.transpose(1, 2)
                        if verbose:
                            print(
                                f"latent shape: {latent.shape}, mel_ref shape: {mel_ref.shape}"
                            )
                        wav, _ = bigvgan_forward_fn(latent, mel_ref)
                        if "cuda" in str(self.device) and self.compile:
                            wav = wav.clone()

                        bigvgan_time += time.perf_counter() - m_start_time
                        wav = wav.squeeze(1)
                        pass
                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                wavs.append(wav)

        # clear cache
        tqdm_progress.close()  # 确保进度条被关闭
        # chunk_latents.clear()
        end_time = time.perf_counter()
        self.torch_empty_cache()

        # wav audio output
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(
            f">> Reference audio length: {cond_mel_frame*256 / sampling_rate:.2f} seconds"
        )
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total fast inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        # print(f">> [fast] bigvgan chunk_length: {chunk_length}")
        print(f">> [fast] batch_num: {all_batch_num} bucket_enable: {bucket_enable}")
        print(f">> [fast] RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)

    # 原始推理模式
    def infer(
        self,
        audio_prompt,
        text,
        output_path,
        verbose=False,
        fusion_method="average",
        weights=None,
    ):
        print(">> start inference...")
        self._set_gr_progress(0, "start inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()
        normalized_text = self.preprocess_text(text)
        print(f"normalized text:{normalized_text}")

        # 使用新方法处理参考音频
        auto_conditioning, cond_mel_frame = self.process_audio_prompts(
            audio_prompt,
            prompt_id=prompt_id,
            verbose=verbose,
            fusion_method=fusion_method,
            weights=weights,
        )

        sentences = self.split_sentences(normalized_text)
        if verbose:
            print("sentences:", sentences)

        top_p = 0.8
        top_k = 30
        temperature = 1.0
        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 3
        repetition_penalty = 10.0
        max_mel_tokens = 600
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        for sent in sentences:
            # sent = " ".join([char for char in sent.upper()]) if lang == "ZH" else sent.upper()
            cleand_text = tokenize_by_CJK_char(sent)
            # cleand_text = "他 那 像 HONG3 小 孩 似 的 话 , 引 得 人 们 HONG1 堂 大 笑 , 大 家 听 了 一 HONG3 而 散 ."
            if verbose:
                print("cleand_text:", cleand_text)

            text_tokens = torch.tensor(
                self.tokenizer.EncodeAsIds(cleand_text),
                dtype=torch.int32,
                device=self.device,
            ).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            if verbose:
                print(text_tokens)
                print(
                    f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}"
                )
                # debug tokenizer
                text_token_syms = self.tokenizer.IdToPiece(text_tokens[0].tolist())
                print(text_token_syms)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=self.device,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    codes = self.gpt.inference_speech(
                        auto_conditioning,
                        text_tokens,
                        cond_mel_lengths=torch.tensor(
                            [auto_conditioning.shape[-1]], device=text_tokens.device
                        ),
                        # text_lengths=text_len,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                    )
                gpt_gen_time += time.perf_counter() - m_start_time
                # codes = codes[:, :-2]
                code_lens = torch.tensor(
                    [codes.shape[-1]], device=codes.device, dtype=codes.dtype
                )
                if verbose:
                    print(codes, type(codes))
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")
                # <代码实现结束></代码实现结束>
                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                codes, code_lens = self.remove_long_silence(
                    codes, silent_token=52, max_consecutive=30
                )
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(
                    self.device, enabled=self.dtype is not None, dtype=self.dtype
                ):
                    latent = self.gpt(
                        auto_conditioning,
                        text_tokens,
                        torch.tensor(
                            [text_tokens.shape[-1]], device=text_tokens.device
                        ),
                        codes,
                        code_lens * self.gpt.mel_length_compression,
                        cond_mel_lengths=torch.tensor(
                            [auto_conditioning.shape[-1]], device=text_tokens.device
                        ),
                        return_latent=True,
                        clip_inputs=False,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav)
        end_time = time.perf_counter()

        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(
            f">> Reference audio length: {cond_mel_frame*256 / sampling_rate:.2f} seconds"
        )
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # torchaudio.save(output_path, wav.cpu().type(torch.int16), sampling_rate)
        # print(">> wav file saved to:", output_path)

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)

    def infer_real_stream(
        self,
        audio_prompt,
        text,
        output_path,
        verbose=False,
        prompt_id="",
        fusion_method="average",
        weights=None,
        buffer_size=25,  # token缓冲区大小
        stream_callback=None,  # 用于实时返回生成的音频片段的回调函数
    ):
        """
        真正的端到端流式语音合成，在生成tokens的同时处理latent并合成音频

        Args:
            audio_prompt: 参考音频路径字符串或多个音频路径的列表
            text: 要合成的文本内容
            output_path: 输出音频文件路径
            verbose: 是否显示详细信息
            prompt_id: 缓存标识符
            fusion_method: 融合方法，"average"或"weighted"
            weights: 多个音频的权重列表
            buffer_size: token缓冲区大小，决定多少tokens处理一次
            stream_callback: 回调函数，用于处理实时生成的音频片段

        Returns:
            str或tuple: 保存的音频文件路径或Gradio格式的(采样率, 音频数据)
        """
        print(">> 开始真实流式推理...")
        self._set_gr_progress(0, "开始真实流式推理...")
        if not prompt_id:
            prompt_id = audio_prompt
        if verbose:
            print(f"原始文本: {text}")

        start_time = time.perf_counter()
        normalized_text = self.preprocess_text(text)
        print(f"规范化文本: {normalized_text}")

        # 使用现有方法处理参考音频
        auto_conditioning, cond_mel_frame = self.process_audio_prompts(
            audio_prompt,
            prompt_id=prompt_id,
            verbose=verbose,
            fusion_method=fusion_method,
            weights=weights,
        )

        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # 处理文本到tokens
        sentences = self.split_sentences(normalized_text)
        all_text_tokens, all_sentences = self.process_text_to_tokens(
            sentences, bucket_enable=True, verbose=verbose
        )

        # 设置生成参数
        top_p = 0.8
        top_k = 30
        temperature = 1.0
        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 1
        repetition_penalty = 10.0
        max_mel_tokens = 600
        sampling_rate = 24000

        # 创建结构化存储，按原始句子索引组织音频片段
        wav_chunks_by_sentence = {}  # 按原始句子索引存储音频片段

        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # 按照分桶处理数据
        all_batch_num = 0

        for item_idx, (item_tokens, sentences_batch) in enumerate(
            zip(all_text_tokens, all_sentences)
        ):
            batch_num = len(item_tokens)
            batch_text_tokens = self.pad_tokens_cat(item_tokens)
            batch_cond_mel_lengths = torch.cat([cond_mel_lengths] * batch_num, dim=0)
            batch_auto_conditioning = torch.cat([auto_conditioning] * batch_num, dim=0)
            all_batch_num += batch_num

            # 流式生成并处理
            self._set_gr_progress(
                0.2 + 0.6 * (item_idx / len(all_text_tokens)),
                f"流式生成批次 {item_idx+1}/{len(all_text_tokens)}",
            )

            # 为当前批次创建音频片段存储
            batch_wav_chunks = [[] for _ in range(batch_num)]

            m_start_time = time.perf_counter()

            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=self.device,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    print(f"开始批次 {item_idx+1}/{len(all_text_tokens)} 的流式生成")

                    # 为每个样本创建token缓冲区
                    token_buffers = [[] for _ in range(batch_num)]
                    end_flags = [False for _ in range(batch_num)]

                    # 记录已经处理过的token数量
                    processed_tokens = [0 for _ in range(batch_num)]

                    # 使用流式生成API
                    for tokens in self.gpt.inference_speech_stream(
                        batch_auto_conditioning,
                        batch_text_tokens,
                        cond_mel_lengths=batch_cond_mel_lengths,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                    ):
                        if tokens is None:
                            continue

                        # 将新生成的tokens添加到对应的缓冲区
                        for batch_index in range(batch_num):
                            if not end_flags[batch_index]:
                                # 检查是否是结束标记
                                if tokens[batch_index] == self.cfg.gpt.stop_mel_token:
                                    end_flags[batch_index] = True
                                else:
                                    token_item = tokens[batch_index].item()
                                    token_buffers[batch_index].append(token_item)

                        # 检查每个样本的缓冲区是否满足处理条件
                        for batch_index in range(batch_num):
                            if (
                                len(token_buffers[batch_index]) >= buffer_size
                                or end_flags[batch_index]
                            ):
                                if len(token_buffers[batch_index]) > 0:
                                    # 处理当前缓冲区中的tokens
                                    current_tokens = token_buffers[batch_index]

                                    # 创建tokens tensor
                                    curr_codes = torch.tensor(
                                        current_tokens,
                                        device=self.device,
                                        dtype=torch.long,
                                    ).unsqueeze(
                                        0
                                    )  # [1, buffer_size]

                                    # 移除重复的连续tokens
                                    curr_codes, _ = torch.unique_consecutive(
                                        curr_codes, return_inverse=True
                                    )

                                    # 处理长静音
                                    curr_codes, code_lens = self.remove_long_silence(
                                        curr_codes, silent_token=52, max_consecutive=30
                                    )

                                    # 获取对应的文本tokens
                                    text_tokens = item_tokens[batch_index]

                                    # 计算latent
                                    latent_start_time = time.perf_counter()

                                    latent = self.gpt(
                                        auto_conditioning,
                                        text_tokens,
                                        torch.tensor(
                                            [text_tokens.shape[-1]],
                                            device=text_tokens.device,
                                        ),
                                        curr_codes,
                                        code_lens * self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor(
                                            [auto_conditioning.shape[-1]],
                                            device=text_tokens.device,
                                        ),
                                        return_latent=True,
                                        clip_inputs=False,
                                    )

                                    gpt_forward_time += (
                                        time.perf_counter() - latent_start_time
                                    )

                                    # 使用BigVGAN生成音频
                                    bigvgan_start_time = time.perf_counter()

                                    mel_ref = auto_conditioning.transpose(1, 2)
                                    wav_chunk, _ = self.bigvgan(latent, mel_ref)
                                    wav_chunk = wav_chunk.squeeze(1).detach().cpu()

                                    bigvgan_time += (
                                        time.perf_counter() - bigvgan_start_time
                                    )

                                    # 处理生成的音频
                                    wav_chunk = torch.clamp(
                                        32767 * wav_chunk, -32767.0, 32767.0
                                    )

                                    # 存储音频片段，保持顺序
                                    batch_wav_chunks[batch_index].append(wav_chunk)

                                    # 如果有回调函数，则调用回调函数处理生成的音频片段
                                    if stream_callback is not None:
                                        stream_callback(
                                            wav_chunk,
                                            sampling_rate,
                                            end_flags[batch_index],
                                        )

                                    # 更新已处理的token数量
                                    processed_tokens[batch_index] += len(
                                        token_buffers[batch_index]
                                    )

                                    # 清空缓冲区
                                    token_buffers[batch_index] = []

                    gpt_gen_time += time.perf_counter() - m_start_time

                # 处理当前批次中可能残留在缓冲区中的tokens
                for batch_index in range(batch_num):
                    if len(token_buffers[batch_index]) > 0:
                        # 处理剩余tokens
                        current_tokens = token_buffers[batch_index]

                        # 创建tokens tensor
                        curr_codes = torch.tensor(
                            current_tokens,
                            device=self.device,
                            dtype=torch.long,
                        ).unsqueeze(
                            0
                        )  # [1, buffer_size]

                        # 移除重复的连续tokens
                        curr_codes, _ = torch.unique_consecutive(
                            curr_codes, return_inverse=True
                        )

                        # 处理长静音
                        curr_codes, code_lens = self.remove_long_silence(
                            curr_codes, silent_token=52, max_consecutive=30
                        )

                        # 获取对应的文本tokens
                        text_tokens = item_tokens[batch_index]

                        # 计算latent
                        latent_start_time = time.perf_counter()

                        latent = self.gpt(
                            auto_conditioning,
                            text_tokens,
                            torch.tensor(
                                [text_tokens.shape[-1]],
                                device=text_tokens.device,
                            ),
                            curr_codes,
                            code_lens * self.gpt.mel_length_compression,
                            cond_mel_lengths=torch.tensor(
                                [auto_conditioning.shape[-1]],
                                device=text_tokens.device,
                            ),
                            return_latent=True,
                            clip_inputs=False,
                        )

                        gpt_forward_time += time.perf_counter() - latent_start_time

                        # 使用BigVGAN生成音频
                        bigvgan_start_time = time.perf_counter()

                        mel_ref = auto_conditioning.transpose(1, 2)
                        wav_chunk, _ = self.bigvgan(latent, mel_ref)
                        wav_chunk = wav_chunk.squeeze(1).detach().cpu()

                        bigvgan_time += time.perf_counter() - bigvgan_start_time

                        # 处理生成的音频
                        wav_chunk = torch.clamp(32767 * wav_chunk, -32767.0, 32767.0)

                        # 存储音频片段，保持顺序
                        batch_wav_chunks[batch_index].append(wav_chunk)

                        # 如果有回调函数，则调用回调函数处理生成的音频片段
                        if stream_callback is not None:
                            stream_callback(
                                wav_chunk, sampling_rate, True  # 这是最后一个片段
                            )

                        # 清空缓冲区
                        token_buffers[batch_index] = []

                    # 将当前批次的音频片段加入到按句子索引组织的字典中
                    sentence_idx = sentences_batch[batch_index]["idx"]
                    if sentence_idx not in wav_chunks_by_sentence:
                        wav_chunks_by_sentence[sentence_idx] = []

                    # 如果该批次的这个样本有生成的音频片段，则合并它们
                    if batch_wav_chunks[batch_index]:
                        batch_wav = torch.cat(batch_wav_chunks[batch_index], dim=1)
                        wav_chunks_by_sentence[sentence_idx].append(batch_wav)

        # 按原始句子顺序合并所有音频片段
        self._set_gr_progress(0.9, "合并音频片段...")
        all_wav_chunks = []
        for i in range(len(sentences)):
            if i in wav_chunks_by_sentence and wav_chunks_by_sentence[i]:
                # 合并同一句子的所有音频片段
                sentence_wav = torch.cat(wav_chunks_by_sentence[i], dim=1)
                all_wav_chunks.append(sentence_wav)

        if all_wav_chunks:
            wav = torch.cat(all_wav_chunks, dim=1)
            wav_length = wav.shape[-1] / sampling_rate

            # 打印统计信息
            print(f">> 参考音频长度: {cond_mel_frame*256 / sampling_rate:.2f} 秒")
            print(f">> gpt_gen_time: {gpt_gen_time:.2f} 秒")
            print(f">> gpt_forward_time: {gpt_forward_time:.2f} 秒")
            print(f">> bigvgan_time: {bigvgan_time:.2f} 秒")
            print(f">> 总流式推理时间: {time.perf_counter() - start_time:.2f} 秒")
            print(f">> 生成音频长度: {wav_length:.2f} 秒")
            print(f">> RTF: {(time.perf_counter() - start_time) / wav_length:.4f}")

            # 保存或返回生成的音频
            if output_path:
                if os.path.isfile(output_path):
                    os.remove(output_path)
                    print(">> 删除旧的音频文件:", output_path)
                if os.path.dirname(output_path) != "":
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
                print(">> 音频文件已保存到:", output_path)
                return output_path
            else:
                wav_data = wav.type(torch.int16)
                wav_data = wav_data.numpy().T
                return (sampling_rate, wav_data)
        else:
            print(">> 警告: 未生成任何音频片段")
            return None
