# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random

from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)


class ScheduledSampler(Sampler):
    """A sampler that samples data from a given concat-dataset.

    Args:
        concat_dataset (ConcatDataset): a concatenated dataset consisting of all datasets
        batch_size (int): batch size
        holistic_shuffle (bool): whether to shuffle the whole dataset or not
        logger (logging.Logger): logger to print warning message

    Usage:
        For cfg.train.batch_size = 3, cfg.train.holistic_shuffle = False, cfg.train.drop_last = True:
        >>> list(ScheduledSampler(ConcatDataset([0, 1, 2], [3, 4, 5], [6, 7, 8]])))
        [3, 4, 5, 0, 1, 2, 6, 7, 8]
    """

    def __init__(
        self, concat_dataset, batch_size, holistic_shuffle, logger=None, type="train"
    ):
        if not isinstance(concat_dataset, ConcatDataset):
            raise ValueError(
                "concat_dataset must be an instance of ConcatDataset, but got {}".format(
                    type(concat_dataset)
                )
            )
        if not isinstance(batch_size, int):
            raise ValueError(
                "batch_size must be an integer, but got {}".format(type(batch_size))
            )
        if not isinstance(holistic_shuffle, bool):
            raise ValueError(
                "holistic_shuffle must be a boolean, but got {}".format(
                    type(holistic_shuffle)
                )
            )

        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        self.holistic_shuffle = holistic_shuffle

        affected_dataset_name = []
        affected_dataset_len = []
        for dataset in concat_dataset.datasets:
            dataset_len = len(dataset)
            dataset_name = dataset.get_dataset_name()
            if dataset_len < batch_size:
                affected_dataset_name.append(dataset_name)
                affected_dataset_len.append(dataset_len)

        self.type = type
        for dataset_name, dataset_len in zip(
            affected_dataset_name, affected_dataset_len
        ):
            if not type == "valid":
                logger.warning(
                    "The {} dataset {} has a length of {}, which is smaller than the batch size {}. This may cause unexpected behavior.".format(
                        type, dataset_name, dataset_len, batch_size
                    )
                )

    def __len__(self):
        # the number of batches with drop last
        num_of_batches = sum(
            [
                math.floor(len(dataset) / self.batch_size)
                for dataset in self.concat_dataset.datasets
            ]
        )
        return num_of_batches * self.batch_size

    def __iter__(self):
        iters = []
        for dataset in self.concat_dataset.datasets:
            iters.append(
                SequentialSampler(dataset).__iter__()
                if self.holistic_shuffle
                else RandomSampler(dataset).__iter__()
            )
        init_indices = [0] + self.concat_dataset.cumulative_sizes[:-1]
        output_batches = []
        for dataset_idx in range(len(self.concat_dataset.datasets)):
            cur_batch = []
            for idx in iters[dataset_idx]:
                cur_batch.append(idx + init_indices[dataset_idx])
                if len(cur_batch) == self.batch_size:
                    output_batches.append(cur_batch)
                    cur_batch = []
                if self.type == "valid" and len(cur_batch) > 0:
                    output_batches.append(cur_batch)
                    cur_batch = []
        # force drop last in training
        random.shuffle(output_batches)
        output_indices = [item for sublist in output_batches for item in sublist]
        return iter(output_indices)


def build_samplers(concat_dataset: Dataset, cfg, logger, type):
    sampler = ScheduledSampler(
        concat_dataset,
        cfg.train.batch_size,
        cfg.train.sampler.holistic_shuffle,
        logger,
        type,
    )
    batch_sampler = BatchSampler(
        sampler,
        cfg.train.batch_size,
        cfg.train.sampler.drop_last if not type == "valid" else False,
    )
    return sampler, batch_sampler
