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

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.common import tokenize_by_CJK_char

from indextts.utils.front import TextNormalizer

class IndexTTS:
    def __init__(
        self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device=None, use_cuda_kernel=None,
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
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
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
            except (ImportError, OSError,CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed加载失败，回退到标准推理: {e}")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)
        
        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load
                anti_alias_activation_cuda = load.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset['bpe_model'])
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.bpe_path)
        print(">> bpe model loaded from:", self.bpe_path)
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        # 缓存参考音频mel：
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
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

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if self.cfg.gpt.stop_mel_token not in code:
                code_lens.append(len(code))
                len_ = len(code)
            else:
                # len_ = code.cpu().tolist().index(8193)+1
                len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                len_ = len_ - 2

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                code = code.cpu().tolist()
                ncode = []
                n = 0
                for k in range(0, len_):
                    if code[k] != silent_token:
                        ncode.append(code[k])
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode.append(code[k])
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                len_ = len(ncode)
                ncode = torch.LongTensor(ncode)
                codes_list.append(ncode.to(device, dtype=dtype))
                isfix = True
                #codes[i] = self.stop_mel_token
                #codes[i, 0:len_] = ncode
            else:
                codes_list.append(codes[i])
            code_lens.append(len_)

        codes = pad_sequence(codes_list, batch_first=True) if isfix else codes[:, :-2]
        code_lens = torch.LongTensor(code_lens).to(device, dtype=dtype)
        return codes, code_lens

    def split_sentences(self, text):
        """
        Split the text into sentences based on punctuation marks.
        """
        # 匹配标点符号（包括中英文标点）
        pattern = r'(?<=[.!?;。！？；])\s*'
        sentences = re.split(pattern, text)
        # 过滤掉空字符串和仅包含标点符号的字符串
        return [
            sentence.strip() for sentence in sentences if sentence.strip() and sentence.strip() not in {"'", ".", ","}
        ]
        
    def bucket_sentences(self, sentences, enable):
        """
        Sentence data bucketing
        """
        max_len = max(len(s) for s in sentences)
        half = max_len // 2
        outputs = [[],[]]
        for idx, sent in enumerate(sentences):
            if enable == False or len(sent) <= half:
                outputs[0].append({"idx":idx,"sent":sent})
            else:
                outputs[1].append({"idx":idx,"sent":sent})
        return [item for item in outputs if item]
        
    def pad_tokens_cat(self, tokens):
        if len(tokens) <= 1:return tokens[-1]
        max_len = max(t.size(1) for t in tokens)
        outputs = []
        for tensor in tokens:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(tensor, 
                        (0, n),
                        value=self.cfg.gpt.stop_text_token
                )
                tensor = torch.nn.functional.pad(tensor, 
                        (0, pad_len - n),
                        value=self.cfg.gpt.start_text_token
                )
            tensor = tensor[:,:max_len]
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
        if self.gr_progress is not None:self.gr_progress(value, desc=desc)
        
        
        
    # 快速推理：对于“多句长文本”，可实现至少 2~10 倍以上的速度提升~ （First modified by sunnyboxs 2025-04-16）
    def infer_fast(self, audio_prompt, text, output_path, verbose=False):
        print(">> start fast inference...")
        self._set_gr_progress(0, "start fast inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()
        normalized_text = self.preprocess_text(text)
        print(f"normalized text:{normalized_text}")
        

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)
            
            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass
        
        auto_conditioning = cond_mel
        cond_mel_lengths = torch.tensor([cond_mel_frame],device=self.device)
        
        # text_tokens
        sentences = self.split_sentences(normalized_text)
        if verbose:
            print("sentences:", sentences)
            
        top_p = .8
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
        bucket_enable = True # 预分桶开关，优先保证质量=True。优先保证速度=False。
        all_sentences = self.bucket_sentences(sentences, enable=bucket_enable) 
        for sentences in all_sentences:
            temp_tokens = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                # sent = " ".join([char for char in sent.upper()]) if lang == "ZH" else sent.upper()
                cleand_text = tokenize_by_CJK_char(sent)
                # cleand_text = "他 那 像 HONG3 小 孩 似 的 话 , 引 得 人 们 HONG1 堂 大 笑 , 大 家 听 了 一 HONG3 而 散 ."
                if verbose:
                    print("cleand_text:", cleand_text)
                    
                text_tokens = torch.tensor(self.tokenizer.EncodeAsIds(cleand_text),dtype=torch.int32, device=self.device).unsqueeze(0)
                # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
                # text_tokens = F.pad(text_tokens, (1, 0), value=0)
                # text_tokens = F.pad(text_tokens, (0, 1), value=1)
                if verbose:
                    print(text_tokens)
                    print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
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
                with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
                    temp_codes = self.gpt.inference_speech(batch_auto_conditioning, batch_text_tokens,
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
                                        max_generate_length=max_mel_tokens)
                    all_batch_codes.append(temp_codes)
            gpt_gen_time += time.perf_counter() - m_start_time
        
        
        # gpt latent
        self._set_gr_progress(0.5, "gpt inference latents...")
        all_idxs = []
        all_latents = []
        for batch_codes, batch_tokens, batch_sentences in zip(all_batch_codes, all_text_tokens, all_sentences):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i] # [x]
                codes = codes[codes != self.cfg.gpt.stop_mel_token]
                codes, _ = torch.unique_consecutive(codes, return_inverse=True)
                codes = codes.unsqueeze(0) # [x] -> [1, x]
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                        code_lens*self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                        return_latent=True, clip_inputs=False)
                        gpt_forward_time += time.perf_counter() - m_start_time
                        all_latents.append(latent)
                        
        
        # bigvgan chunk
        chunk_size = 2      
        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        chunk_latents = [all_latents[i:i + chunk_size] for i in range(0, len(all_latents), chunk_size)]
        chunk_length = len(chunk_latents)
        latent_length = len(all_latents)
        all_latents = None
        
        # bigvgan chunk decode
        self._set_gr_progress(0.7, "bigvgan decode...")
        tqdm_progress = tqdm(total=latent_length, desc="bigvgan")
        for items in chunk_latents:
            tqdm_progress.update(len(items))
            latent = torch.cat(items, dim=1)
            with torch.no_grad():
                with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)
                    pass
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav)
                
        # clear cache   
        tqdm_progress.close()  # 确保进度条被关闭
        chunk_latents.clear()
        end_time = time.perf_counter()
        self.torch_empty_cache()
        
        # wav audio output
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame*256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total fast inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> [fast] bigvgan chunk_length: {chunk_length}")
        print(f">> [fast] batch_num: {all_batch_num} bucket_enable: {bucket_enable}")
        print(f">> [fast] RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu() # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            os.makedirs(os.path.dirname(output_path),exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T  
            return (sampling_rate, wav_data)
        
    
    
    # 原始推理模式
    def infer(self, audio_prompt, text, output_path, verbose=False):
        print(">> start inference...")
        self._set_gr_progress(0, "start inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()
        normalized_text = self.preprocess_text(text)
        print(f"normalized text:{normalized_text}")


        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)
            
            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass
        

        auto_conditioning = cond_mel

        sentences = self.split_sentences(normalized_text)
        if verbose:
            print("sentences:", sentences)

        top_p = .8
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

            text_tokens = torch.tensor(self.tokenizer.EncodeAsIds(cleand_text),dtype=torch.int32, device=self.device).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.IdToPiece(text_tokens[0].tolist())
                print(text_token_syms)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                      device=text_tokens.device),
                                                        # text_lengths=text_len,
                                                        do_sample=True,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        temperature=temperature,
                                                        num_return_sequences=autoregressive_batch_size,
                                                        length_penalty=length_penalty,
                                                        num_beams=num_beams,
                                                        repetition_penalty=repetition_penalty,
                                                        max_generate_length=max_mel_tokens)
                gpt_gen_time += time.perf_counter() - m_start_time
                #codes = codes[:, :-2]
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                if verbose:
                    print(codes, type(codes))
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = \
                        self.gpt(auto_conditioning, text_tokens,
                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                    code_lens*self.gpt.mel_length_compression,
                                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                    return_latent=True, clip_inputs=False)
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
        print(f">> Reference audio length: {cond_mel_frame*256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # torchaudio.save(output_path, wav.cpu().type(torch.int16), sampling_rate)
        # print(">> wav file saved to:", output_path)
        
        # save audio
        wav = wav.cpu() # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path),exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T  
            return (sampling_rate, wav_data)


if __name__ == "__main__":
    prompt_wav="test_data/input.wav"
    #text="晕 XUAN4 是 一 种 GAN3 觉"
    #text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'
    text="There is a vehicle arriving in dock number 7?"

    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, use_cuda_kernel=False)
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
