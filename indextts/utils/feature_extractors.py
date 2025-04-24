import torch
import torchaudio
from torch import nn
from typing import List, Union, Optional
from indextts.utils.common import safe_log


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, win_length=None,
                 n_mels=100, mel_fmin=0, mel_fmax=None, normalize=False, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1,
            normalized=normalize,
            f_min=mel_fmin,
            f_max=mel_fmax,
            n_mels=n_mels,
            center=padding == "center",
        )

    def forward(self, audio, weights=None, fusion_method="average", **kwargs):
        """
        Extract mel spectrogram features from audio inputs.
        
        Args:
            audio: Single audio tensor of shape (B, T) or list of audio tensors
            weights: Optional list of weights for each audio when using weighted fusion
            fusion_method: Method to fuse multiple audio features ('average', 'weighted')
            
        Returns:
            Tensor: Mel spectrogram features
        """
        # 处理单个音频输入的情况
        if isinstance(audio, torch.Tensor) and audio.dim() <= 2:
            return self._process_single_audio(audio)
        
        # 处理多个音频输入的情况
        if isinstance(audio, list):
            if len(audio) == 1:
                return self._process_single_audio(audio[0])
            
            # 提取每个音频的特征
            mel_features = []
            for single_audio in audio:
                mel = self._process_single_audio(single_audio)
                mel_features.append(mel)
            
            # 融合特征
            if fusion_method == "average":
                # 简单平均融合
                return torch.stack(mel_features).mean(dim=0)
            
            elif fusion_method == "weighted":
                # 加权融合
                if weights is None:
                    # 如果没有提供权重，使用均等权重
                    weights = torch.ones(len(mel_features), device=mel_features[0].device)
                    weights = weights / weights.sum()
                else:
                    if isinstance(weights, list):
                        weights = torch.tensor(weights, device=mel_features[0].device)
                    weights = weights / weights.sum()
                
                # 应用权重并相加
                weighted_sum = torch.zeros_like(mel_features[0])
                for i, mel in enumerate(mel_features):
                    weighted_sum += mel * weights[i]
                
                return weighted_sum
            
            else:
                raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # 如果输入是批次的音频 (B, C, T)
        elif audio.dim() == 3:
            batch_size = audio.size(0)
            mel_features = []
            
            # 处理每个批次
            for i in range(batch_size):
                mel = self._process_single_audio(audio[i])
                mel_features.append(mel)
            
            # 堆叠结果
            return torch.stack(mel_features)
        
        else:
            raise ValueError(f"Unsupported audio input format: {type(audio)}")
    
    def _process_single_audio(self, audio):
        """处理单个音频片段"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # 添加批次维度
            
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
            
        mel = self.mel_spec(audio)
        mel = safe_log(mel)
        return mel