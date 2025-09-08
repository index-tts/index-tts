import torch
import librosa
import json5
from huggingface_hub import hf_hub_download
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
import safetensors
import numpy as np

from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec
from indextts.utils.maskgct.models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
from indextts.utils.maskgct.models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
import time


def _load_config(config_fn, lowercase=False):
    """Load configurations into a dictionary

    Args:
        config_fn (str): path to configuration file
        lowercase (bool, optional): whether changing keys to lower case. Defaults to False.

    Returns:
        dict: dictionary that stores configurations
    """
    with open(config_fn, "r") as f:
        data = f.read()
    config_ = json5.loads(data)
    if "base_config" in config_:
        # load configurations from new path
        p_config_path = os.path.join(os.getenv("WORK_DIR"), config_["base_config"])
        p_config_ = _load_config(p_config_path)
        config_ = override_config(p_config_, config_)
    if lowercase:
        # change keys in config_ to lower case
        config_ = get_lowercase_keys_config(config_)
    return config_


def load_config(config_fn, lowercase=False):
    """Load configurations into a dictionary

    Args:
        config_fn (str): path to configuration file
        lowercase (bool, optional): _description_. Defaults to False.

    Returns:
        JsonHParams: an object that stores configurations
    """
    config_ = _load_config(config_fn, lowercase=lowercase)
    # create an JsonHParams object with configuration dict
    cfg = JsonHParams(**config_)
    return cfg


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def build_semantic_model(path_='./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt'):
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    stat_mean_var = torch.load(path_)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec(cfg):
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    return semantic_codec


def build_s2a_model(cfg, device):
    soundstorm_model = MaskGCT_S2A(cfg=cfg)
    soundstorm_model.eval()
    soundstorm_model.to(device)
    return soundstorm_model


def build_acoustic_codec(cfg, device):
    codec_encoder = CodecEncoder(cfg=cfg.encoder)
    codec_decoder = CodecDecoder(cfg=cfg.decoder)
    codec_encoder.eval()
    codec_decoder.eval()
    codec_encoder.to(device)
    codec_decoder.to(device)
    return codec_encoder, codec_decoder


class Inference_Pipeline():
    def __init__(
            self,
            semantic_model,
            semantic_codec,
            semantic_mean,
            semantic_std,
            codec_encoder,
            codec_decoder,
            s2a_model_1layer,
            s2a_model_full,
            ):
        self.semantic_model = semantic_model
        self.semantic_codec = semantic_codec
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std

        self.codec_encoder = codec_encoder
        self.codec_decoder = codec_decoder
        self.s2a_model_1layer = s2a_model_1layer
        self.s2a_model_full = s2a_model_full

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)
        return feat

    @torch.no_grad()
    def extract_acoustic_code(self, speech):
        vq_emb = self.codec_encoder(speech.unsqueeze(1))
        _, vq, _, _, _ = self.codec_decoder.quantizer(vq_emb)
        acoustic_code = vq.permute(1, 2, 0)
        return acoustic_code

    @torch.no_grad()
    def get_scode(self, inputs):
        semantic_code, feat = self.semantic_codec.quantize(inputs)
        # vq = self.semantic_codec.quantizer.vq2emb(semantic_code.unsqueeze(1))
        # vq = vq.transpose(1,2)
        return semantic_code

    @torch.no_grad()
    def semantic2acoustic(
        self,
        combine_semantic_code,
        acoustic_code,
        n_timesteps=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        cfg=2.5,
        rescale_cfg=0.75,
    ):
        semantic_code = combine_semantic_code

        cond = self.s2a_model_1layer.cond_emb(semantic_code)
        prompt = acoustic_code[:, :, :]
        predict_1layer = self.s2a_model_1layer.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=n_timesteps[:1],
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        cond = self.s2a_model_full.cond_emb(semantic_code)
        prompt = acoustic_code[:, :, :]
        predict_full = self.s2a_model_full.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            gt_code=predict_1layer,
        )

        vq_emb = self.codec_decoder.vq2emb(
            predict_full.permute(2, 0, 1), n_quantizers=12
        )
        recovered_audio = self.codec_decoder(vq_emb)
        prompt_vq_emb = self.codec_decoder.vq2emb(
            prompt.permute(2, 0, 1), n_quantizers=12
        )
        recovered_prompt_audio = self.codec_decoder(prompt_vq_emb)
        recovered_prompt_audio = recovered_prompt_audio[0][0].cpu().numpy()
        recovered_audio = recovered_audio[0][0].cpu().numpy()
        combine_audio = np.concatenate([recovered_prompt_audio, recovered_audio])

        return combine_audio, recovered_audio

    def s2a_inference(
        self,
        prompt_speech_path,
        combine_semantic_code,
        cfg=2.5,
        n_timesteps_s2a=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        cfg_s2a=2.5,
        rescale_cfg_s2a=0.75,
    ):
        speech = librosa.load(prompt_speech_path, sr=24000)[0]
        acoustic_code = self.extract_acoustic_code(
            torch.tensor(speech).unsqueeze(0).to(combine_semantic_code.device)
        )
        _, recovered_audio = self.semantic2acoustic(
            combine_semantic_code,
            acoustic_code,
            n_timesteps=n_timesteps_s2a,
            cfg=cfg_s2a,
            rescale_cfg=rescale_cfg_s2a,
        )

        return recovered_audio

    @torch.no_grad()
    def gt_inference(
        self,
        prompt_speech_path,
        combine_semantic_code,
    ):
        speech = librosa.load(prompt_speech_path, sr=24000)[0]
        '''
        acoustic_code = self.extract_acoustic_code(
            torch.tensor(speech).unsqueeze(0).to(combine_semantic_code.device)
        )
        prompt = acoustic_code[:, :, :]
        prompt_vq_emb = self.codec_decoder.vq2emb(
            prompt.permute(2, 0, 1), n_quantizers=12
        )
        '''

        prompt_vq_emb = self.codec_encoder(torch.tensor(speech).unsqueeze(0).unsqueeze(1).to(combine_semantic_code.device))
        recovered_prompt_audio = self.codec_decoder(prompt_vq_emb)
        recovered_prompt_audio = recovered_prompt_audio[0][0].cpu().numpy()
        return recovered_prompt_audio
