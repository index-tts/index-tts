# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random

import numpy as np

import torchaudio
import librosa
from torch.nn import functional as F

from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.codec.codec_dataset import CodecDataset


class FAcodecDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        """
        Args:
            cfg: config
            dataset: dataset name
            is_valid: whether to use train or valid dataset
        """
        self.data_root_dir = cfg.dataset
        self.data_list = []
        # walk through the dataset directory recursively, save all files ends with .wav/.mp3/.opus/.flac/.m4a
        for root, _, files in os.walk(self.data_root_dir):
            for file in files:
                if file.endswith((".wav", ".mp3", ".opus", ".flac", ".m4a")):
                    self.data_list.append(os.path.join(root, file))
        self.sr = cfg.preprocess_params.sr
        self.duration_range = cfg.preprocess_params.duration_range
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=cfg.preprocess_params.spect_params.n_mels,
            n_fft=cfg.preprocess_params.spect_params.n_fft,
            win_length=cfg.preprocess_params.spect_params.win_length,
            hop_length=cfg.preprocess_params.spect_params.hop_length,
        )
        self.mean, self.std = -4, 4

    def preprocess(self, wave):
        wave_tensor = (
            torch.from_numpy(wave).float() if isinstance(wave, np.ndarray) else wave
        )
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def __len__(self):
        # return len(self.data_list)
        return len(self.data_list)  # return a fixed number for testing

    def __getitem__(self, index):
        wave, _ = librosa.load(self.data_list[index], sr=self.sr)
        wave = np.random.randn(self.sr * random.randint(*self.duration_range))
        wave = wave / np.max(np.abs(wave))
        mel = self.preprocess(wave).squeeze(0)
        wave = torch.from_numpy(wave).float()
        return wave, mel


class FAcodecCollator(object):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_wave_length = max([b[0].size(0) for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float() - 10
        waves = torch.zeros((batch_size, max_wave_length)).float()

        mel_lengths = torch.zeros(batch_size).long()
        wave_lengths = torch.zeros(batch_size).long()

        for bid, (wave, mel) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            waves[bid, : wave.size(0)] = wave
            mel_lengths[bid] = mel_size
            wave_lengths[bid] = wave.size(0)

        return waves, mels, wave_lengths, mel_lengths
