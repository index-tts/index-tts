# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable
import torch
import numpy as np
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from torch.utils.data import ConcatDataset, Dataset


class CodecDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        """
        Args:
            cfg: config
            dataset: dataset name
            is_valid: whether to use train or valid dataset
        """
        assert isinstance(dataset, str)

        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)

        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)
        self.metadata = self.get_metadata()

        self.data_root = processed_data_dir
        self.cfg = cfg

        if cfg.preprocess.use_audio:
            self.utt2audio_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2audio_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.audio_dir,
                    uid + ".npy",
                )
        elif cfg.preprocess.use_label:
            self.utt2label_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2label_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.label_dir,
                    uid + ".npy",
                )
        elif cfg.preprocess.use_one_hot:
            self.utt2one_hot_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2one_hot_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.one_hot_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_mel:
            self.utt2mel_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2mel_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.mel_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_frame_pitch:
            self.utt2frame_pitch_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2frame_pitch_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.pitch_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_uv:
            self.utt2uv_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                self.utt2uv_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.uv_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_amplitude_phase:
            self.utt2logamp_path = {}
            self.utt2pha_path = {}
            self.utt2rea_path = {}
            self.utt2imag_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                self.utt2logamp_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.log_amplitude_dir,
                    uid + ".npy",
                )
                self.utt2pha_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.phase_dir,
                    uid + ".npy",
                )
                self.utt2rea_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.real_dir,
                    uid + ".npy",
                )
                self.utt2imag_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.imaginary_dir,
                    uid + ".npy",
                )

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()

        if self.cfg.preprocess.use_mel:
            mel = np.load(self.utt2mel_path[utt])
            assert mel.shape[0] == self.cfg.preprocess.n_mel  # [n_mels, T]

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = mel.shape[1]

            single_feature["mel"] = mel

        if self.cfg.preprocess.use_frame_pitch:
            frame_pitch = np.load(self.utt2frame_pitch_path[utt])

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_pitch)

            aligned_frame_pitch = align_length(
                frame_pitch, single_feature["target_len"]
            )

            single_feature["frame_pitch"] = aligned_frame_pitch

        if self.cfg.preprocess.use_audio:
            audio = np.load(self.utt2audio_path[utt])

            single_feature["audio"] = audio

        return single_feature

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata

    def get_dataset_name(self):
        return self.metadata[0]["Dataset"]

    def __len__(self):
        return len(self.metadata)


class CodecConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset], full_audio_inference=False):
        """Concatenate a series of datasets with their random inference audio merged."""
        super().__init__(datasets)

        self.cfg = self.datasets[0].cfg

        self.metadata = []

        # Merge metadata
        for dataset in self.datasets:
            self.metadata += dataset.metadata

        # Merge random inference features
        if full_audio_inference:
            self.eval_audios = []
            self.eval_dataset_names = []
            if self.cfg.preprocess.use_mel:
                self.eval_mels = []
            if self.cfg.preprocess.use_frame_pitch:
                self.eval_pitchs = []
            for dataset in self.datasets:
                self.eval_audios.append(dataset.eval_audio)
                self.eval_dataset_names.append(dataset.get_dataset_name())
                if self.cfg.preprocess.use_mel:
                    self.eval_mels.append(dataset.eval_mel)
                if self.cfg.preprocess.use_frame_pitch:
                    self.eval_pitchs.append(dataset.eval_pitch)


class CodecCollator(object):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, n_mels, frame]
        # frame_pitch: [b, frame]
        # audios: [b, frame * hop_size]

        for key in batch[0].keys():
            if key == "target_len":
                packed_batch_features["target_len"] = torch.LongTensor(
                    [b["target_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["target_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "mel":
                values = [torch.from_numpy(b[key]).T for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
