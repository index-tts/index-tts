import os
import re
import sys

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
    def __init__(self, cfg_path='checkpoints/config.yaml', model_dir='checkpoints', is_fp16=True):
        self.cfg = OmegaConf.load(cfg_path)
        self.device = 'cuda:0'
        self.model_dir = model_dir
        self.is_fp16 = is_fp16
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        if self.is_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = None
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

    def remove_long_silence(self, codes, silent_token=52, max_consecutive=30):
        code_lens = []
        codes_list = []
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
                codes_list.append(ncode.cuda())
                isfix = True
                #codes[i] = self.stop_mel_token
                #codes[i, 0:len_] = ncode
            else:
                codes_list.append(codes[i])
            code_lens.append(len_)

        codes = pad_sequence(codes_list, batch_first=True) if isfix else codes[:, :-2]
        code_lens = torch.LongTensor(code_lens).cuda()
        return codes, code_lens

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
        print(f"cond_mel shape: {cond_mel.shape}")

        auto_conditioning = cond_mel

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(self.bpe_path)

        punctuation = ["!", "?", ".", ";", "！", "？", "。", "；"]
        pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
        sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
        print(sentences)

        top_p = .8
        top_k = 30
        temperature = 1.0
        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 3
        repetition_penalty = 10.0
        max_mel_tokens = 600
        sampling_rate = 24000
        lang = "EN"
        lang = "ZH"
        wavs = []
        wavs1 = []

        for sent in sentences:
            print(sent)
            # sent = " ".join([char for char in sent.upper()]) if lang == "ZH" else sent.upper()
            cleand_text = tokenize_by_CJK_char(sent)
            # cleand_text = "他 那 像 HONG3 小 孩 似 的 话 , 引 得 人 们 HONG1 堂 大 笑 , 大 家 听 了 一 HONG3 而 散 ."
            print(cleand_text)
            text_tokens = torch.IntTensor(tokenizer.encode(cleand_text)).unsqueeze(0).to(self.device)

            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            text_tokens = text_tokens.to(self.device)
            print(text_tokens)
            print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
            text_token_syms = [tokenizer.IdToPiece(idx) for idx in text_tokens[0].tolist()]
            print(text_token_syms)
            text_len = [text_tokens.size(1)]
            text_len = torch.IntTensor(text_len).to(self.device)
            print(text_len)
            with torch.no_grad():
                if self.is_fp16:
                    with torch.cuda.amp.autocast(enabled=self.dtype is not None, dtype=self.dtype):
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
                else:
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
                code_lens = torch.tensor([codes.shape[-1]])
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
                if self.is_fp16:
                    with torch.cuda.amp.autocast(enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                     torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                     code_lens*self.gpt.mel_length_compression,
                                     cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                     return_latent=True, clip_inputs=False)
                        latent = latent.transpose(1, 2)
                        wav, _ = self.bigvgan(latent.transpose(1, 2), auto_conditioning.transpose(1, 2))
                        wav = wav.squeeze(1).cpu()

                else:
                    latent = \
                        self.gpt(auto_conditioning, text_tokens,
                                 torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                 code_lens*self.gpt.mel_length_compression,
                                 cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                 return_latent=True, clip_inputs=False)
                    latent = latent.transpose(1, 2)
                    '''
                    latent_list = []
                    for lat, t_len in zip(latent, text_lens_out):
                        lat = lat[:, t_len:]
                        latent_list.append(lat)
                    latent = torch.stack(latent_list)
                    print(f"latent shape: {latent.shape}")
                    '''

                    wav, _ = self.bigvgan(latent.transpose(1, 2), auto_conditioning.transpose(1, 2))
                    wav = wav.squeeze(1).cpu()

                wav = 32767 * wav
                torch.clip(wav, -32767.0, 32767.0)
                print(f"wav shape: {wav.shape}")
                # wavs.append(wav[:, :-512])
                wavs.append(wav)

        wav = torch.cat(wavs, dim=1)
        if output_path:
            torchaudio.save(output_path, wav.type(torch.int16), 24000)
        else:
            return wav.type(torch.int16)


if __name__ == "__main__":
    prompt_wav="test_data/input.wav"
    prompt_wav="testwav/spk_1744181067_1.wav"
    #text="晕 XUAN4 是 一 种 GAN3 觉"
    #text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'
    text="There is a vehicle arriving in dock number 7?"

    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True)
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen.wav")
