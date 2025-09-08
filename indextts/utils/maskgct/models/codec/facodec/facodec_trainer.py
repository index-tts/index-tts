# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import random
from pathlib import Path
import re
import glob

import accelerate
import json
import numpy as np
import torch
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio

from accelerate.logging import get_logger

from models.codec.facodec.facodec_dataset import FAcodecDataset, FAcodecCollator
from models.codec.codec_sampler import build_samplers
from models.codec.codec_trainer import CodecTrainer

from modules.dac.nn.loss import (
    MultiScaleSTFTLoss,
    MelSpectrogramLoss,
    GANLoss,
    L1Loss,
    FocalLoss,
)
from audiotools import AudioSignal

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    print(
        "Unable to import nemo_asr, titanet outputs will be set to random values, you may only run debugging mode. DO NOT USE THIS FOR TRAINING"
    )
    nemo_asr = None

from models.codec.facodec.modules.commons import (
    build_model,
    load_checkpoint,
    load_F0_models,
    log_norm,
)
from models.codec.facodec.optimizer import build_optimizer


class FAcodecTrainer(CodecTrainer):
    def __init__(self, args, cfg):
        super().__init__()

        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        # Init accelerator
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            self.logger = get_logger(args.exp_name, log_level=args.log_level)

        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New training process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")
        self.logger.debug(f"Using {args.log_level.upper()} logging level.")
        self.logger.info(f"Experiment name: {args.exp_name}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # Init training status
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0

        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        self.logger.info(
            "Max epoch: {}".format(
                self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
            )
        )

        # Check potential erorrs
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.run_eval = self.cfg.train.run_eval

        # Set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            self.logger.debug(
                f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            )
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # Build dataloader
        with self.accelerator.main_process_first():
            self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

        # Build model
        with self.accelerator.main_process_first():
            self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            for _, model in self.model.items():
                self.logger.debug(model)
            self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
            self.logger.info(f"Model parameters: {self._count_parameters()/1e6:.2f}M")

        # Build optimizers and schedulers
        with self.accelerator.main_process_first():
            self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            end = time.monotonic_ns()
            self.logger.info(
                f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
            )

        # Build helper models
        with self.accelerator.main_process_first():
            self.logger.info("Building helper models...")
            start = time.monotonic_ns()
            self._built_helper_model()
            end = time.monotonic_ns()
            self.logger.info(
                f"Building helper models done in {(end - start) / 1e6:.2f}ms"
            )

        # Accelerator preparing
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        for k in self.model:
            self.model[k] = self.accelerator.prepare(self.model[k])
        for k, v in self.optimizer.optimizers.items():
            self.optimizer.optimizers[k] = self.accelerator.prepare(
                self.optimizer.optimizers[k]
            )
            self.optimizer.schedulers[k] = self.accelerator.prepare(
                self.optimizer.schedulers[k]
            )
        end = time.monotonic_ns()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms")

        # Build criterions
        with self.accelerator.main_process_first():
            self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterions = self._build_criterion()
            end = time.monotonic_ns()
            self.logger.info(f"Building criterion done in {(end - start) / 1e6:.2f}ms")

        # Resume checkpoints
        with self.accelerator.main_process_first():
            self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
            if args.resume_type:
                self.logger.info("Resuming from checkpoint...")
                start = time.monotonic_ns()
                ckpt_path = Path(args.checkpoint)
                if self._is_valid_pattern(ckpt_path.parts[-1]):
                    ckpt_path = self._load_model(args.checkpoint, args.resume_type)
                else:
                    ckpt_path = self._load_model(
                        args.checkpoint, resume_type=args.resume_type
                    )
                end = time.monotonic_ns()
                self.logger.info(
                    f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                )
                self.checkpoints_path = json.load(
                    open(os.path.join(ckpt_path, "ckpts.json"), "r")
                )

            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # Save config
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

    def _build_dataset(self):
        return FAcodecDataset, FAcodecCollator

    def _build_criterion(self):
        criterions = dict()
        stft_criterion = MultiScaleSTFTLoss()
        mel_criterion = MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            mag_weight=0.0,
            clamp_eps=1e-5,
        )
        content_criterion = FocalLoss(gamma=2)
        l1_criterion = L1Loss()
        criterions["stft"] = stft_criterion
        criterions["mel"] = mel_criterion
        criterions["l1"] = l1_criterion
        criterions["content"] = content_criterion

        return criterions

    def _build_model(self):
        model = build_model(self.cfg.model_params)
        _ = [model[key].to(self.accelerator.device) for key in model]
        return model

    def _built_helper_model(self):
        device = self.accelerator.device
        self.pitch_extractor = load_F0_models(self.cfg.F0_path).to(device)

        # load model and processor
        self.w2v_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.w2v_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        ).to(device)
        self.w2v_model.eval()

        if nemo_asr is None:
            self.speaker_model = None
        else:
            self.speaker_model = (
                nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                    "nvidia/speakerverification_en_titanet_large"
                )
            )
            self.speaker_model = self.speaker_model.to(device)
            self.speaker_model.eval()

    def _build_optimizer(self):
        scheduler_params = {
            "warmup_steps": self.cfg.loss_params.warmup_steps,
            "base_lr": self.cfg.loss_params.base_lr,
        }
        optimizer = build_optimizer(
            {key: self.model[key] for key in self.model},
            scheduler_params_dict={key: scheduler_params.copy() for key in self.model},
            lr=float(scheduler_params["base_lr"]),
        )

        return optimizer

    def train_loop(self):
        """Training process"""
        self.accelerator.wait_for_everyone()

        # Dump config
        if self.accelerator.is_main_process:
            self._dump_cfg(self.config_save_path)
        _ = [self.model[key].train() for key in self.model]
        self.optimizer.zero_grad()

        # Sync and start training
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))

            # Train and Validate
            train_total_loss, train_losses = self._train_epoch()
            for key, loss in train_losses.items():
                self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                self.accelerator.log(
                    {"Epoch/Train {} Loss".format(key): loss},
                    step=self.epoch,
                )
            self.accelerator.log(
                {
                    "Epoch/Train Total Loss": train_total_loss,
                },
                step=self.epoch,
            )

            # Update scheduler
            self.accelerator.wait_for_everyone()

            # Check save checkpoint interval
            run_eval = False
            if self.accelerator.is_main_process:
                save_checkpoint = False
                for i, num in enumerate(self.save_checkpoint_stride):
                    if self.epoch % num == 0:
                        save_checkpoint = True
                        run_eval |= self.run_eval[i]

            # Save checkpoints
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and save_checkpoint:
                print("Saving..")
                state = {
                    "net": {key: self.model[key].state_dict() for key in self.model},
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.optimizer.scheduler_state_dict(),
                    "iters": self.step,
                    "epoch": self.epoch,
                }
                save_path = os.path.join(
                    self.checkpoint_dir,
                    "FAcodec_epoch_%05d_step_%05d.pth" % (self.epoch, self.iters),
                )
                torch.save(state, save_path)
                json.dump(
                    self.checkpoints_path,
                    open(os.path.join(self.checkpoint_dir, "ckpts.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )

            self.accelerator.wait_for_everyone()

            self.epoch += 1

        # Finish training
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            path = os.path.join(
                self.checkpoint_dir,
                "epoch-{:04d}_step-{:07d}".format(
                    self.epoch,
                    self.step,
                ),
            )
            print("Saving..")
            state = {
                "net": {key: self.model[key].state_dict() for key in self.model},
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.optimizer.scheduler_state_dict(),
                "iters": self.step,
                "epoch": self.epoch,
            }
            save_path = os.path.join(
                self.checkpoint_dir,
                "FAcodec_epoch_%05d_step_%05d.pth" % (self.epoch, self.iters),
            )
            torch.save(state, save_path)

    def _train_epoch(self):
        """Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        _ = [self.model[key].train() for key in self.model]

        epoch_losses: dict = {}
        epoch_total_loss: int = 0

        for batch in tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            # Get losses
            total_loss, losses = self._train_step(batch)
            self.batch_count += 1

            # Log info
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                self.accelerator.log(
                    {
                        "Step/Learning Rate": (
                            self.optimizer.schedulers["encoder"].get_last_lr()[0]
                            if self.step != 0
                            else 0
                        )
                    },
                    step=self.step,
                )
                for key, _ in losses.items():
                    self.accelerator.log(
                        {
                            "Step/Train {} Loss".format(key): losses[key],
                        },
                        step=self.step,
                    )

                if not epoch_losses:
                    epoch_losses = losses
                else:
                    for key, value in losses.items():
                        epoch_losses[key] += value
                epoch_total_loss += total_loss
                self.step += 1

        # Get and log total losses
        self.accelerator.wait_for_everyone()
        epoch_total_loss = (
            epoch_total_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )
        for key in epoch_losses.keys():
            epoch_losses[key] = (
                epoch_losses[key]
                / len(self.train_dataloader)
                * self.cfg.train.gradient_accumulation_step
            )
        return epoch_total_loss, epoch_losses

    def _train_step(self, data):
        """Training forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_train_epoch`` for usage.
        """
        # Init losses
        train_losses = {}
        total_loss = 0

        # Use input feature to get predictions
        data = [b.to(self.accelerator.device, non_blocking=True) for b in data]
        waves, mels, wave_lengths, mel_input_length = data

        # extract semantic latent with w2v model
        waves_16k = torchaudio.functional.resample(waves, 24000, 16000)
        w2v_input = self.w2v_processor(
            waves_16k, sampling_rate=16000, return_tensors="pt"
        ).input_values.to(self.accelerator.device)
        with torch.no_grad():
            w2v_outputs = self.w2v_model(w2v_input.squeeze(0)).logits
            predicted_ids = torch.argmax(w2v_outputs, dim=-1)
            phone_ids = (
                F.interpolate(
                    predicted_ids.unsqueeze(0).float(), mels.size(-1), mode="nearest"
                )
                .long()
                .squeeze(0)
            )

        # get clips
        mel_seg_len = min(
            [int(mel_input_length.min().item()), self.cfg.train.max_frame_len]
        )

        gt_mel_seg = []
        wav_seg = []
        w2v_seg = []

        for bib in range(len(mel_input_length)):
            mel_length = int(mel_input_length[bib].item())

            random_start = (
                np.random.randint(0, mel_length - mel_seg_len)
                if mel_length != mel_seg_len
                else 0
            )
            gt_mel_seg.append(mels[bib, :, random_start : random_start + mel_seg_len])

            # w2v_seg.append(w2v_latent[bib, :, random_start:random_start + mel_seg_len])
            w2v_seg.append(phone_ids[bib, random_start : random_start + mel_seg_len])

            y = waves[bib][random_start * 300 : (random_start + mel_seg_len) * 300]

            wav_seg.append(y.to(self.accelerator.device))

        gt_mel_seg = torch.stack(gt_mel_seg).detach()

        wav_seg = torch.stack(wav_seg).float().detach().unsqueeze(1)
        w2v_seg = torch.stack(w2v_seg).float().detach()

        with torch.no_grad():
            real_norm = log_norm(gt_mel_seg.unsqueeze(1)).squeeze(1).detach()
            F0_real, _, _ = self.pitch_extractor(gt_mel_seg.unsqueeze(1))

        # normalize f0
        # Remove unvoiced frames (replace with -1)
        gt_glob_f0s = []
        f0_targets = []
        for bib in range(len(F0_real)):
            voiced_indices = F0_real[bib] > 5.0
            f0_voiced = F0_real[bib][voiced_indices]

            if len(f0_voiced) != 0:
                # Convert to log scale
                log_f0 = f0_voiced.log2()

                # Calculate mean and standard deviation
                mean_f0 = log_f0.mean()
                std_f0 = log_f0.std()

                # Normalize the F0 sequence
                normalized_f0 = (log_f0 - mean_f0) / std_f0

                # Create the normalized F0 sequence with unvoiced frames
                normalized_sequence = torch.zeros_like(F0_real[bib])
                normalized_sequence[voiced_indices] = normalized_f0
                normalized_sequence[~voiced_indices] = (
                    -10
                )  # Assign -10 to unvoiced frames

                gt_glob_f0s.append(mean_f0)
            else:
                normalized_sequence = torch.zeros_like(F0_real[bib]) - 10.0
                gt_glob_f0s.append(torch.tensor(0.0).to(self.accelerator.device))

            # f0_targets.append(normalized_sequence[single_side_context // 200:-single_side_context // 200])
            f0_targets.append(normalized_sequence)
        f0_targets = torch.stack(f0_targets).to(self.accelerator.device)
        # fill nan with -10
        f0_targets[torch.isnan(f0_targets)] = -10.0
        # fill inf with -10
        f0_targets[torch.isinf(f0_targets)] = -10.0
        # if frame_rate not equal to 80, interpolate f0 from frame rate of 80 to target frame rate
        if self.cfg.preprocess_params.frame_rate != 80:
            f0_targets = F.interpolate(
                f0_targets.unsqueeze(1),
                mel_seg_len // 80 * self.cfg.preprocess_params.frame_rate,
                mode="nearest",
            ).squeeze(1)
            w2v_seg = F.interpolate(
                w2v_seg,
                mel_seg_len // 80 * self.cfg.preprocess_params.frame_rate,
                mode="nearest",
            )

        wav_seg_input = wav_seg
        wav_seg_target = wav_seg

        z = self.model.encoder(wav_seg_input)
        z, quantized, commitment_loss, codebook_loss, timbre = self.model.quantizer(
            z, wav_seg_input, n_c=2, full_waves=waves, wave_lens=wave_lengths
        )
        preds, rev_preds = self.model.fa_predictors(quantized, timbre)

        pred_wave = self.model.decoder(z)

        len_diff = wav_seg_target.size(-1) - pred_wave.size(-1)
        if len_diff > 0:
            wav_seg_target = wav_seg_target[..., len_diff // 2 : -len_diff // 2]

        # discriminator loss
        d_fake = self.model.discriminator(pred_wave.detach())
        d_real = self.model.discriminator(wav_seg_target)
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)

        self.optimizer.zero_grad()
        self.accelerator.backward(loss_d)
        grad_norm_d = torch.nn.utils.clip_grad_norm_(
            self.model.discriminator.parameters(), 10.0
        )
        self.optimizer.step("discriminator")
        self.optimizer.scheduler(key="discriminator")

        # generator loss
        signal = AudioSignal(wav_seg_target, sample_rate=24000)
        recons = AudioSignal(pred_wave, sample_rate=24000)
        stft_loss = self.criterions["stft"](recons, signal)
        mel_loss = self.criterions["mel"](recons, signal)
        waveform_loss = self.criterions["l1"](recons, signal)

        d_fake = self.model.discriminator(pred_wave)
        d_real = self.model.discriminator(wav_seg_target)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

        pred_f0, pred_uv = preds["f0"], preds["uv"]
        rev_pred_f0, rev_pred_uv = rev_preds["rev_f0"], rev_preds["rev_uv"]

        common_min_size = min(pred_f0.size(-2), f0_targets.size(-1))
        f0_targets = f0_targets[..., :common_min_size]
        real_norm = real_norm[..., :common_min_size]

        f0_loss = F.smooth_l1_loss(
            f0_targets, pred_f0.squeeze(-1)[..., :common_min_size]
        )
        uv_loss = F.smooth_l1_loss(
            real_norm, pred_uv.squeeze(-1)[..., :common_min_size]
        )
        rev_f0_loss = (
            F.smooth_l1_loss(f0_targets, rev_pred_f0.squeeze(-1)[..., :common_min_size])
            if rev_pred_f0 is not None
            else torch.FloatTensor([0]).to(self.accelerator.device)
        )
        rev_uv_loss = (
            F.smooth_l1_loss(real_norm, rev_pred_uv.squeeze(-1)[..., :common_min_size])
            if rev_pred_uv is not None
            else torch.FloatTensor([0]).to(self.accelerator.device)
        )

        tot_f0_loss = f0_loss + rev_f0_loss
        tot_uv_loss = uv_loss + rev_uv_loss

        pred_content = preds["content"]
        rev_pred_content = rev_preds["rev_content"]

        target_content_latents = w2v_seg[..., :common_min_size]

        content_loss = self.criterions["content"](
            pred_content.transpose(1, 2)[..., :common_min_size],
            target_content_latents.long(),
        )
        rev_content_loss = (
            self.criterions["content"](
                rev_pred_content.transpose(1, 2)[..., :common_min_size],
                target_content_latents.long(),
            )
            if rev_pred_content is not None
            else torch.FloatTensor([0]).to(self.accelerator.device)
        )

        tot_content_loss = content_loss + rev_content_loss

        if self.speaker_model is not None:
            spk_logits = torch.cat(
                [
                    self.speaker_model.infer_segment(w16.cpu()[..., :wl])[1]
                    for w16, wl in zip(waves_16k, wave_lengths)
                ],
                dim=0,
            )
            spk_labels = spk_logits.argmax(dim=-1)
        else:
            spk_labels = torch.zeros([len(waves_16k)], dtype=torch.long).to(
                self.accelerator.device
            )

        spk_pred_logits = preds["timbre"]
        spk_loss = F.cross_entropy(spk_pred_logits, spk_labels)
        x_spk_pred_logits = rev_preds["x_timbre"]

        x_spk_loss = (
            F.cross_entropy(x_spk_pred_logits, spk_labels)
            if x_spk_pred_logits is not None
            else torch.FloatTensor([0]).to(self.accelerator.device)
        )

        tot_spk_loss = spk_loss + x_spk_loss

        loss_gen_all = (
            mel_loss * 15.0
            + loss_feature * 1.0
            + loss_g * 1.0
            + commitment_loss * 0.25
            + codebook_loss * 1.0
            + tot_f0_loss * 1.0
            + tot_uv_loss * 1.0
            + tot_content_loss * 5.0
            + tot_spk_loss * 5.0
        )

        self.optimizer.zero_grad()
        self.accelerator.backward(loss_gen_all)

        with torch.no_grad():
            total_loss = loss_gen_all.item()
            train_losses["stft"] = stft_loss.item()
            train_losses["mel"] = mel_loss.item()
            train_losses["l1"] = waveform_loss.item()
            train_losses["f0"] = f0_loss.item()
            train_losses["uv"] = uv_loss.item()
            train_losses["content"] = content_loss.item()
            train_losses["speaker"] = spk_loss.item()
            train_losses["rev_f0"] = rev_f0_loss.item()
            train_losses["rev_uv"] = rev_uv_loss.item()
            train_losses["rev_content"] = rev_content_loss.item()
            train_losses["rev_speaker"] = x_spk_loss.item()

            train_losses["feature"] = loss_feature.item()
            train_losses["generator"] = loss_g.item()
            train_losses["commitment"] = commitment_loss.item()
            train_losses["codebook"] = codebook_loss.item()

            # discriminators
            train_losses["discriminator"] = loss_d.item()

        return total_loss, train_losses

    def _inference(self, eval_wave):
        """Inference during training for test audios."""
        z = self.model.encoder(
            eval_wave[None, None, ...].to(self.accelerator.device).float()
        )
        z, quantized, commitment_loss, codebook_loss, timbre = self.model.quantizer(
            z, eval_wave[None, None, ...], n_c=self.cfg.model_params.n_c_codebooks
        )
        full_pred_wave = self.model.decoder(z)
        return full_pred_wave[0]

    def _load_model(self, checkpoint_path=None, resume_type="resume"):
        """Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if resume_type == "resume":
            if checkpoint_path is None:
                available_checkpoints = glob.glob(
                    os.path.join(self.checkpoint_dir, "FAcodc_epoch_*_step_*.pth")
                )
                # find the checkpoint that has the highest step number
                latest_checkpoint = max(
                    available_checkpoints,
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )
                earliest_checkpoint = min(
                    available_checkpoints,
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )
                # delete the earliest checkpoint
                if (
                    earliest_checkpoint != latest_checkpoint
                    and self.accelerator.is_main_process
                    and len(available_checkpoints) > 4
                ):
                    os.remove(earliest_checkpoint)
                    print(f"Removed {earliest_checkpoint}")
            else:
                latest_checkpoint = checkpoint_path

            self.model, self.optimizer, self.epoch, self.step = load_checkpoint(
                self.model,
                self.optimizer,
                latest_checkpoint,
                load_only_params=False,
                ignore_modules=[],
                is_distributed=self.accelerator.num_processes > 1,
            )

        else:
            raise ValueError("Invalid resume type")
        return checkpoint_path

    def _count_parameters(self):
        total_num = sum(
            sum(p.numel() for p in self.model[key].parameters()) for key in self.model
        )
        # trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_num
