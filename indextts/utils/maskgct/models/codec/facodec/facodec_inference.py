# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import warnings
import argparse
import torch
import os
import yaml

warnings.simplefilter("ignore")

from .modules.commons import *
import time

import torchaudio
import librosa
from collections import OrderedDict


class FAcodecInference(object):
    def __init__(self, args=None, cfg=None):
        self.args = args
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self._load_checkpoint()

    def _build_model(self):
        model = build_model(self.cfg.model_params)
        _ = [model[key].to(self.device) for key in model]
        return model

    def _load_checkpoint(self):
        sd = torch.load(self.args.checkpoint_path, map_location="cpu")
        sd = sd["net"] if "net" in sd else sd
        new_params = dict()
        for key, state_dict in sd.items():
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            new_params[key] = new_state_dict
        for key in new_params:
            if key in self.model:
                self.model[key].load_state_dict(new_params[key])
        _ = [self.model[key].eval() for key in self.model]

    @torch.no_grad()
    def inference(self, source, output_dir):
        source_audio = librosa.load(source, sr=self.cfg.preprocess_params.sr)[0]
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)

        z = self.model.encoder(source_audio[None, ...].to(self.device).float())
        (
            z,
            quantized,
            commitment_loss,
            codebook_loss,
            timbre,
            codes,
        ) = self.model.quantizer(
            z,
            source_audio[None, ...].to(self.device).float(),
            n_c=self.cfg.model_params.n_c_codebooks,
            return_codes=True,
        )

        full_pred_wave = self.model.decoder(z)

        os.makedirs(output_dir, exist_ok=True)
        source_name = source.split("/")[-1].split(".")[0]
        torchaudio.save(
            f"{output_dir}/reconstructed_{source_name}.wav",
            full_pred_wave[0].cpu(),
            self.cfg.preprocess_params.sr,
        )

        print(
            "Reconstructed audio saved as: ",
            f"{output_dir}/reconstructed_{source_name}.wav",
        )

        return quantized, codes

    @torch.no_grad()
    def voice_conversion(self, source, reference, output_dir):
        source_audio = librosa.load(source, sr=self.cfg.preprocess_params.sr)[0]
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)

        reference_audio = librosa.load(reference, sr=self.cfg.preprocess_params.sr)[0]
        reference_audio = (
            torch.tensor(reference_audio).unsqueeze(0).float().to(self.device)
        )

        z = self.model.encoder(source_audio[None, ...].to(self.device).float())
        z, quantized, commitment_loss, codebook_loss, timbre = self.model.quantizer(
            z,
            source_audio[None, ...].to(self.device).float(),
            n_c=self.cfg.model_params.n_c_codebooks,
        )

        z_ref = self.model.encoder(reference_audio[None, ...].to(self.device).float())
        (
            z_ref,
            quantized_ref,
            commitment_loss_ref,
            codebook_loss_ref,
            timbre_ref,
        ) = self.model.quantizer(
            z_ref,
            reference_audio[None, ...].to(self.device).float(),
            n_c=self.cfg.model_params.n_c_codebooks,
        )

        z_conv = self.model.quantizer.voice_conversion(
            quantized[0] + quantized[1],
            reference_audio[None, ...].to(self.device).float(),
        )
        full_pred_wave = self.model.decoder(z_conv)

        os.makedirs(output_dir, exist_ok=True)
        source_name = source.split("/")[-1].split(".")[0]
        reference_name = reference.split("/")[-1].split(".")[0]
        torchaudio.save(
            f"{output_dir}/converted_{source_name}_to_{reference_name}.wav",
            full_pred_wave[0].cpu(),
            self.cfg.preprocess_params.sr,
        )

        print(
            "Voice conversion results saved as: ",
            f"{output_dir}/converted_{source_name}_to_{reference_name}.wav",
        )
