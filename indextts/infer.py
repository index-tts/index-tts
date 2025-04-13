import os
import re
import time
import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.common import tokenize_by_CJK_char
from indextts.vqvae.xtts_dvae import DiscreteVAE

from indextts.utils.front import TextNormalizer

class IndexTTS:
    def __init__(self, cfg_path='checkpoints/config.yaml', model_dir='checkpoints', is_fp16=True, device=None):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == 'cpu' else is_fp16
        elif torch.cuda.is_available():
            self.device = 'cuda:0'
            self.is_fp16 = is_fp16
        elif torch.mps.is_available():
            self.device = 'mps'
            self.is_fp16 = is_fp16
        else:
            self.device = 'cpu'
            self.is_fp16 = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.dvae = DiscreteVAE(**self.cfg.vqvae)
        self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        load_checkpoint(self.dvae, self.dvae_path)
        self.dvae = self.dvae.to(self.device)
        if self.is_fp16:
            self.dvae.eval().half()
        else:
            self.dvae.eval()
        print(">> vqvae weights restored from:", self.dvae_path)
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
            self.gpt.post_init_gpt2_config(use_deepspeed=True, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

        self.bigvgan = Generator(self.cfg.bigvgan)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location='cpu')
        self.bigvgan.load_state_dict(vocoder_dict['generator'])
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset['bpe_model'])
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.bpe_path)
        print(">> bpe model loaded from:", self.bpe_path)
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")

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

    def infer(self, audio_prompt, text, output_path):
        print(f"origin text:{text}")
        text = self.preprocess_text(text)
        print(f"normalized text:{text}")


        audio, sr = torchaudio.load(audio_prompt)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
        print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

        auto_conditioning = cond_mel

        sentences = self.split_sentences(text)
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
        print(">> start inference...")
        
        start_time = time.time()

        for sent in sentences:
            print(sent)
            # sent = " ".join([char for char in sent.upper()]) if lang == "ZH" else sent.upper()
            cleand_text = tokenize_by_CJK_char(sent)
            # cleand_text = "他 那 像 HONG3 小 孩 似 的 话 , 引 得 人 们 HONG1 堂 大 笑 , 大 家 听 了 一 HONG3 而 散 ."
            print("cleand_text:", cleand_text)

            text_tokens = torch.tensor(self.tokenizer.EncodeAsIds(cleand_text),dtype=torch.int32, device=self.device).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)

            print(text_tokens)
            print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
            # debug tokenizer
            text_token_syms = self.tokenizer.IdToPiece(text_tokens[0].tolist())
            print(text_token_syms)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)

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
                #codes = codes[:, :-2]
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                print(codes, type(codes))
                print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                print(f"code len: {code_lens}")
                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                print(codes, type(codes))
                print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                print(f"code len: {code_lens}")

                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = \
                        self.gpt(auto_conditioning, text_tokens,
                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                    code_lens*self.gpt.mel_length_compression,
                                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                    return_latent=True, clip_inputs=False)
                    latent = latent.transpose(1, 2)
                    wav, _ = self.bigvgan(latent.transpose(1, 2), auto_conditioning.transpose(1, 2))
                    wav = wav.squeeze(1).cpu()

                wav = torch.clip(32767 * wav, -32767.0, 32767.0)
                print(f"wav shape: {wav.shape}")
                # wavs.append(wav[:, :-512])
                wavs.append(wav)

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)
        print(f">> inference done. time: {minutes:02d}:{seconds:02d}.{milliseconds:03d}")
        print(">> saving wav file")
        wav = torch.cat(wavs, dim=1)
        torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
        print(">> wav file saved to:", output_path)


if __name__ == "__main__":
    prompt_wav="test_data/input.wav"
    prompt_wav="testwav/spk_1744181067_1.wav"
    #text="晕 XUAN4 是 一 种 GAN3 觉"
    #text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'
    text="There is a vehicle arriving in dock number 7?"

    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True)
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
