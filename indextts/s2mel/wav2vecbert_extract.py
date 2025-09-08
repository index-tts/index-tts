from transformers import SeamlessM4TFeatureExtractor
from transformers import Wav2Vec2BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
import pickle
import math
import json
import safetensors
import json5
# from codec.kmeans.repcodec_model import RepCodec
from startts.examples.ftchar.models.codec.kmeans.repcodec_model import RepCodec

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

class Extract_wav2vectbert:
    def __init__(self,device):
    #semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model = Wav2Vec2BertModel.from_pretrained("./MaskGCT_model/w2v_bert/")
        self.semantic_model.eval()
        self.semantic_model.to(device)
        self.stat_mean_var = torch.load("./MaskGCT_model/wav2vec2bert_stats.pt")
        self.semantic_mean = self.stat_mean_var["mean"]
        self.semantic_std = torch.sqrt(self.stat_mean_var["var"])
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(
                "./MaskGCT_model/w2v_bert/")
        self.device = device
        
        cfg_maskgct = load_config('./MaskGCT_model/maskgct.json')
        cfg = cfg_maskgct.model.semantic_codec
        self.semantic_code_ckpt = r'./MaskGCT_model/semantic_codec/model.safetensors'
        self.semantic_codec = RepCodec(cfg=cfg)
        self.semantic_codec.eval()
        self.semantic_codec.to(device)
        safetensors.torch.load_model(self.semantic_codec, self.semantic_code_ckpt)

    @torch.no_grad()
    def extract_features(self, speech): # speech [b,T]
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        return input_features, attention_mask #[2, 620, 160] [2, 620]

    @torch.no_grad()
    def extract_semantic_code(self, input_features, attention_mask):
        vq_emb = self.semantic_model(           # Wav2Vec2BertModel
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        semantic_code, rec_feat = self.semantic_codec.quantize(feat)  # (B, T)
        return semantic_code, rec_feat

    def feature_extract(self, prompt_speech):
        
        input_features, attention_mask = self.extract_features(prompt_speech)
        input_features = input_features.to(self.device)
        attention_mask = attention_mask.to(self.device)
        semantic_code, rec_feat = self.extract_semantic_code(input_features, attention_mask)
        return semantic_code,rec_feat
            
if __name__=='__main__':
    speech_path = 'test/magi1.wav'
    speech = librosa.load(speech_path, sr=16000)[0]
    speech = np.c_[speech,speech,speech].T #[2, 198559] 
    print(speech.shape)
            
    Extract_feature = Extract_wav2vectbert('cuda:0')
    semantic_code,rec_feat = Extract_feature.feature_extract(speech)
    print(semantic_code.shape,rec_feat.shape)
    
