# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import json
import json5
import time
import accelerate
import random
import numpy as np
import shutil

from pathlib import Path
from tqdm import tqdm
from glob import glob
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from models.vocoders.vocoder_dataset import (
    VocoderDataset,
    VocoderCollator,
    VocoderConcatDataset,
)

from models.vocoders.gan.generator import bigvgan, hifigan, melgan, nsfhifigan, apnet
from models.vocoders.flow.waveglow import waveglow
from models.vocoders.diffusion.diffwave import diffwave
from models.vocoders.autoregressive.wavenet import wavenet
from models.vocoders.autoregressive.wavernn import wavernn

from models.vocoders.gan import gan_vocoder_inference
from models.vocoders.diffusion import diffusion_vocoder_inference

from utils.io import save_audio

_vocoders = {
    "diffwave": diffwave.DiffWave,
    "wavernn": wavernn.WaveRNN,
    "wavenet": wavenet.WaveNet,
    "waveglow": waveglow.WaveGlow,
    "nsfhifigan": nsfhifigan.NSFHiFiGAN,
    "bigvgan": bigvgan.BigVGAN,
    "hifigan": hifigan.HiFiGAN,
    "melgan": melgan.MelGAN,
    "apnet": apnet.APNet,
}

# Forward call for generalized Inferencor
_vocoder_forward_funcs = {
    # "world": world_inference.synthesis_audios,
    # "wavernn": wavernn_inference.synthesis_audios,
    # "wavenet": wavenet_inference.synthesis_audios,
    "diffwave": diffusion_vocoder_inference.vocoder_inference,
    "nsfhifigan": gan_vocoder_inference.vocoder_inference,
    "bigvgan": gan_vocoder_inference.vocoder_inference,
    "melgan": gan_vocoder_inference.vocoder_inference,
    "hifigan": gan_vocoder_inference.vocoder_inference,
    "apnet": gan_vocoder_inference.vocoder_inference,
}

# APIs for other tasks. e.g. SVC, TTS, TTA...
_vocoder_infer_funcs = {
    # "world": world_inference.synthesis_audios,
    # "wavernn": wavernn_inference.synthesis_audios,
    # "wavenet": wavenet_inference.synthesis_audios,
    "diffwave": diffusion_vocoder_inference.synthesis_audios,
    "nsfhifigan": gan_vocoder_inference.synthesis_audios,
    "bigvgan": gan_vocoder_inference.synthesis_audios,
    "melgan": gan_vocoder_inference.synthesis_audios,
    "hifigan": gan_vocoder_inference.synthesis_audios,
    "apnet": gan_vocoder_inference.synthesis_audios,
}


class VocoderInference(object):
    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        super().__init__()

        start = time.monotonic_ns()
        self.args = args
        self.cfg = cfg
        self.infer_type = infer_type

        # Init accelerator
        self.accelerator = accelerate.Accelerator()
        self.accelerator.wait_for_everyone()

        # Get logger
        with self.accelerator.main_process_first():
            self.logger = get_logger("inference", log_level=args.log_level)

        # Log some info
        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New inference process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")

        self.vocoder_dir = args.vocoder_dir
        self.logger.debug(f"Vocoder dir: {args.vocoder_dir}")

        os.makedirs(args.output_dir, exist_ok=True)
        if os.path.exists(os.path.join(args.output_dir, "pred")):
            shutil.rmtree(os.path.join(args.output_dir, "pred"))
        if os.path.exists(os.path.join(args.output_dir, "gt")):
            shutil.rmtree(os.path.join(args.output_dir, "gt"))
        os.makedirs(os.path.join(args.output_dir, "pred"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "gt"), exist_ok=True)

        # Set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            self.logger.debug(
                f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            )
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # Setup inference mode
        if self.infer_type == "infer_from_dataset":
            self.cfg.dataset = self.args.infer_datasets
        elif self.infer_type == "infer_from_feature":
            self._build_tmp_dataset_from_feature()
            self.cfg.dataset = ["tmp"]
        elif self.infer_type == "infer_from_audio":
            self._build_tmp_dataset_from_audio()
            self.cfg.dataset = ["tmp"]

        # Setup data loader
        with self.accelerator.main_process_first():
            self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.test_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

        # Build model
        with self.accelerator.main_process_first():
            self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            self.logger.info(f"Building model done in {(end - start) / 1e6:.3f}ms")

        # Init with accelerate
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        self.accelerator = accelerate.Accelerator()
        (self.model, self.test_dataloader) = self.accelerator.prepare(
            self.model, self.test_dataloader
        )
        end = time.monotonic_ns()
        self.accelerator.wait_for_everyone()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.3f}ms")

        with self.accelerator.main_process_first():
            self.logger.info("Loading checkpoint...")
            start = time.monotonic_ns()
            if os.path.isdir(args.vocoder_dir):
                if os.path.isdir(os.path.join(args.vocoder_dir, "checkpoint")):
                    self._load_model(os.path.join(args.vocoder_dir, "checkpoint"))
                else:
                    self._load_model(os.path.join(args.vocoder_dir))
            else:
                self._load_model(os.path.join(args.vocoder_dir))
            end = time.monotonic_ns()
            self.logger.info(f"Loading checkpoint done in {(end - start) / 1e6:.3f}ms")

        self.model.eval()
        self.accelerator.wait_for_everyone()

    def _build_tmp_dataset_from_feature(self):
        if os.path.exists(os.path.join(self.cfg.preprocess.processed_dir, "tmp")):
            shutil.rmtree(os.path.join(self.cfg.preprocess.processed_dir, "tmp"))

        utts = []
        mels = glob(os.path.join(self.args.feature_folder, "mels", "*.npy"))
        for i, mel in enumerate(mels):
            uid = mel.split("/")[-1].split(".")[0]
            utt = {"Dataset": "tmp", "Uid": uid, "index": i}
            utts.append(utt)

        os.makedirs(os.path.join(self.cfg.preprocess.processed_dir, "tmp"))
        with open(
            os.path.join(self.cfg.preprocess.processed_dir, "tmp", "test.json"), "w"
        ) as f:
            json.dump(utts, f)

        meta_info = {"dataset": "tmp", "test": {"size": len(utts)}}

        with open(
            os.path.join(self.cfg.preprocess.processed_dir, "tmp", "meta_info.json"),
            "w",
        ) as f:
            json.dump(meta_info, f)

        features = glob(os.path.join(self.args.feature_folder, "*"))
        for feature in features:
            feature_name = feature.split("/")[-1]
            if os.path.isfile(feature):
                continue
            shutil.copytree(
                os.path.join(self.args.feature_folder, feature_name),
                os.path.join(self.cfg.preprocess.processed_dir, "tmp", feature_name),
            )

    def _build_tmp_dataset_from_audio(self):
        if os.path.exists(os.path.join(self.cfg.preprocess.processed_dir, "tmp")):
            shutil.rmtree(os.path.join(self.cfg.preprocess.processed_dir, "tmp"))

        utts = []
        audios = glob(os.path.join(self.args.audio_folder, "*"))
        for i, audio in enumerate(audios):
            uid = audio.split("/")[-1].split(".")[0]
            utt = {"Dataset": "tmp", "Uid": uid, "index": i, "Path": audio}
            utts.append(utt)

        os.makedirs(os.path.join(self.cfg.preprocess.processed_dir, "tmp"))
        with open(
            os.path.join(self.cfg.preprocess.processed_dir, "tmp", "test.json"), "w"
        ) as f:
            json.dump(utts, f)

        meta_info = {"dataset": "tmp", "test": {"size": len(utts)}}

        with open(
            os.path.join(self.cfg.preprocess.processed_dir, "tmp", "meta_info.json"),
            "w",
        ) as f:
            json.dump(meta_info, f)

        from processors import acoustic_extractor

        acoustic_extractor.extract_utt_acoustic_features_serial(
            utts, os.path.join(self.cfg.preprocess.processed_dir, "tmp"), self.cfg
        )

    def _build_test_dataset(self):
        return VocoderDataset, VocoderCollator

    def _build_model(self):
        model = _vocoders[self.cfg.model.generator](self.cfg)
        return model

    def _build_dataloader(self):
        """Build dataloader which merges a series of datasets."""
        Dataset, Collator = self._build_test_dataset()

        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset, is_valid=True)
            datasets_list.append(subdataset)
        test_dataset = VocoderConcatDataset(datasets_list, full_audio_inference=False)
        test_collate = Collator(self.cfg)
        test_batch_size = min(self.cfg.inference.batch_size, len(test_dataset))
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=test_collate,
            num_workers=1,
            batch_size=test_batch_size,
            shuffle=False,
        )
        self.test_batch_size = test_batch_size
        self.test_dataset = test_dataset
        return test_dataloader

    def _load_model(self, checkpoint_dir, from_multi_gpu=False):
        """Load model from checkpoint. If a folder is given, it will
        load the latest checkpoint in checkpoint_dir. If a path is given
        it will load the checkpoint specified by checkpoint_path.
        **Only use this method after** ``accelerator.prepare()``.
        """
        if os.path.isdir(checkpoint_dir):
            if "epoch" in checkpoint_dir and "step" in checkpoint_dir:
                checkpoint_path = checkpoint_dir
            else:
                # Load the latest accelerator state dicts
                ls = [
                    str(i)
                    for i in Path(checkpoint_dir).glob("*")
                    if not "audio" in str(i)
                ]
                ls.sort(
                    key=lambda x: int(x.split("/")[-1].split("_")[0].split("-")[-1]),
                    reverse=True,
                )
                checkpoint_path = ls[0]
            accelerate.load_checkpoint_and_dispatch(
                self.accelerator.unwrap_model(self.model),
                os.path.join(checkpoint_path, "pytorch_model.bin"),
            )
            return str(checkpoint_path)
        else:
            # Load old .pt checkpoints
            if self.cfg.model.generator in [
                "bigvgan",
                "hifigan",
                "melgan",
                "nsfhifigan",
            ]:
                ckpt = torch.load(
                    checkpoint_dir,
                    map_location=(
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    ),
                )
                if from_multi_gpu:
                    pretrained_generator_dict = ckpt["generator_state_dict"]
                    generator_dict = self.model.state_dict()

                    new_generator_dict = {
                        k.split("module.")[-1]: v
                        for k, v in pretrained_generator_dict.items()
                        if (
                            k.split("module.")[-1] in generator_dict
                            and v.shape == generator_dict[k.split("module.")[-1]].shape
                        )
                    }

                    generator_dict.update(new_generator_dict)

                    self.model.load_state_dict(generator_dict)
                else:
                    self.model.load_state_dict(ckpt["generator_state_dict"])
            else:
                self.model.load_state_dict(torch.load(checkpoint_dir)["state_dict"])
            return str(checkpoint_dir)

    def inference(self):
        """Inference via batches"""
        for i, batch in tqdm(enumerate(self.test_dataloader)):
            if self.cfg.preprocess.use_frame_pitch:
                audio_pred = _vocoder_forward_funcs[self.cfg.model.generator](
                    self.cfg,
                    self.model,
                    batch["mel"].transpose(-1, -2),
                    f0s=batch["frame_pitch"].float(),
                    device=next(self.model.parameters()).device,
                )
            else:
                audio_pred = _vocoder_forward_funcs[self.cfg.model.generator](
                    self.cfg,
                    self.model,
                    batch["mel"].transpose(-1, -2),
                    device=next(self.model.parameters()).device,
                )
            audio_ls = audio_pred.chunk(self.test_batch_size)
            audio_gt_ls = batch["audio"].cpu().chunk(self.test_batch_size)
            length_ls = batch["target_len"].cpu().chunk(self.test_batch_size)
            j = 0
            for it, it_gt, l in zip(audio_ls, audio_gt_ls, length_ls):
                l = l.item()
                it = it.squeeze(0).squeeze(0)[: l * self.cfg.preprocess.hop_size]
                it_gt = it_gt.squeeze(0)[: l * self.cfg.preprocess.hop_size]
                uid = self.test_dataset.metadata[i * self.test_batch_size + j]["Uid"]
                save_audio(
                    os.path.join(self.args.output_dir, "pred", "{}.wav").format(uid),
                    it,
                    self.cfg.preprocess.sample_rate,
                )
                save_audio(
                    os.path.join(self.args.output_dir, "gt", "{}.wav").format(uid),
                    it_gt,
                    self.cfg.preprocess.sample_rate,
                )
                j += 1

        if os.path.exists(os.path.join(self.cfg.preprocess.processed_dir, "tmp")):
            shutil.rmtree(os.path.join(self.cfg.preprocess.processed_dir, "tmp"))

    def _set_random_seed(self, seed):
        """Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def _count_parameters(self, model):
        return sum(p.numel() for p in model.parameters())

    def _dump_cfg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json5.dump(
            self.cfg,
            open(path, "w"),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            quote_keys=True,
        )


def load_nnvocoder(
    cfg,
    vocoder_name,
    weights_file,
    from_multi_gpu=False,
):
    """Load the specified vocoder.
    cfg: the vocoder config filer.
    weights_file: a folder or a .pt path.
    from_multi_gpu: automatically remove the "module" string in state dicts if "True".
    """
    print("Loading Vocoder from Weights file: {}".format(weights_file))

    # Build model
    model = _vocoders[vocoder_name](cfg)
    if not os.path.isdir(weights_file):
        # Load from .pt file
        if vocoder_name in ["bigvgan", "hifigan", "melgan", "nsfhifigan"]:
            ckpt = torch.load(
                weights_file,
                map_location=(
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            )
            if from_multi_gpu:
                pretrained_generator_dict = ckpt["generator_state_dict"]
                generator_dict = model.state_dict()

                new_generator_dict = {
                    k.split("module.")[-1]: v
                    for k, v in pretrained_generator_dict.items()
                    if (
                        k.split("module.")[-1] in generator_dict
                        and v.shape == generator_dict[k.split("module.")[-1]].shape
                    )
                }

                generator_dict.update(new_generator_dict)

                model.load_state_dict(generator_dict)
            else:
                model.load_state_dict(ckpt["generator_state_dict"])
        else:
            model.load_state_dict(torch.load(weights_file)["state_dict"])
    else:
        # Load from accelerator state dict
        weights_file = os.path.join(weights_file, "checkpoint")
        ls = [str(i) for i in Path(weights_file).glob("*") if not "audio" in str(i)]
        ls.sort(key=lambda x: int(x.split("_")[-3].split("-")[-1]), reverse=True)
        checkpoint_path = ls[0]
        accelerator = accelerate.Accelerator()
        model = accelerator.prepare(model)
        accelerator.load_state(checkpoint_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    return model


def tensorize(data, device, n_samples):
    """
    data: a list of numpy array
    """
    assert type(data) == list
    if n_samples:
        data = data[:n_samples]
    data = [torch.as_tensor(x, device=device) for x in data]
    return data


def synthesis(
    cfg,
    vocoder_weight_file,
    n_samples,
    pred,
    f0s=None,
    batch_size=64,
    fast_inference=False,
):
    """Synthesis audios from a given vocoder and series of given features.
    cfg: vocoder config.
    vocoder_weight_file: a folder of accelerator state dict or a path to the .pt file.
    pred: a list of numpy arrays. [(seq_len1, acoustic_features_dim), (seq_len2, acoustic_features_dim), ...]
    """

    vocoder_name = cfg.model.generator

    print("Synthesis audios using {} vocoder...".format(vocoder_name))

    ###### TODO: World Vocoder Refactor ######
    # if vocoder_name == "world":
    #     world_inference.synthesis_audios(
    #         cfg, dataset_name, split, n_samples, pred, save_dir, tag
    #     )
    #     return

    # ====== Loading neural vocoder model ======
    vocoder = load_nnvocoder(
        cfg, vocoder_name, weights_file=vocoder_weight_file, from_multi_gpu=True
    )
    device = next(vocoder.parameters()).device

    # ====== Inference for predicted acoustic features ======
    # pred: (frame_len, n_mels) -> (n_mels, frame_len)
    mels_pred = tensorize([p.T for p in pred], device, n_samples)
    print("For predicted mels, #sample = {}...".format(len(mels_pred)))
    audios_pred = _vocoder_infer_funcs[vocoder_name](
        cfg,
        vocoder,
        mels_pred,
        f0s=f0s,
        batch_size=batch_size,
        fast_inference=fast_inference,
    )
    return audios_pred
