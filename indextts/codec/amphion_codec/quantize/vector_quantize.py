# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, "n d -> n () d") - rearrange(
                means, "c d -> () c d"
            )
            dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        weight_init=False,
    ):
        super().__init__()

        self.decay = decay
        init_fn = torch.randn if not weight_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        if weight_init:
            nn.init.uniform_(embed, -1 / codebook_size, 1 / codebook_size)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer(
            "initted", torch.Tensor([not kmeans_init])
        )  # if kmeans_init is True, then initted is False; otherwise, initted is True
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def init_embed_(self, data):
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace(batch_samples, mask=expired_codes)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, "... d -> (...) d")
        embed = self.embed.t()  # (codebook_size, dim) -> (dim, codebook_size)

        if not self.initted:
            self.init_embed_(flatten)

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = (
                flatten.t() @ embed_onehot
            )  # (dim, ...) @ (..., codebook_size) -> (dim, codebook_size)
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.eps)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        return quantize, embed_ind

    def vq2emb(self, vq):
        quantize = F.embedding(vq, self.embed)
        return quantize

    def latent2dist(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, "... d -> (...) d")
        embed = self.embed.t()  # (codebook_size, dim) -> (dim, codebook_size)

        if not self.initted:
            self.init_embed_(flatten)

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        dist = dist.view(*shape[:-1], -1)

        return dist, embed_ind, quantize


class SimpleCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        use_l2_normlize=False,
    ):
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.use_l2_normlize = use_l2_normlize

        self.embed = nn.Embedding(self.codebook_size, self.dim)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, "... d -> (...) d")
        embed = self.embed.weight.t()  # (codebook_size, dim) -> (dim, codebook_size)

        if self.use_l2_normlize:
            flatten = F.normalize(flatten)
            embed = F.normalize(embed)

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        return quantize, embed_ind

    def vq2emb(self, vq):
        quantize = F.embedding(vq, self.embed.weight)
        return quantize

    def latent2dist(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, "... d -> (...) d")
        embed = self.embed.weight.t()  # (codebook_size, dim) -> (dim, codebook_size)

        if self.use_l2_normlize:
            flatten = F.normalize(flatten)
            embed = F.normalize(embed)

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        dist = dist.view(*shape[:-1], -1)

        return dist, embed_ind, quantize


class VectorQuantize(nn.Module):
    """Vector quantization and factorized vecotor quantization implementation
    Args:
        input_dim (int): Dimension of input.
        codebook_size (int): Codebook size.
        codebook_dim (int): Codebook dimension. We suggest use codebook_dim = input_dim
            if use codebook_type == "euclidean", otherwise, if you want to use
            factorized vector quantization, use codebook_dim as small number (e.g. 8 or 32).
        commitment (float): Weight for commitment loss.
        use_l2_normlize (bool): Whether to use l2 normlized codes for factorized vecotor quantization,
            we suggest use it as True if you want to use factorized vector quantization
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=0.005,
        codebook_loss_weight=1.0,
        use_l2_normlize=False,
        codebook_type="euclidean",  # "euclidean" or "simple"
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        weight_init=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.use_l2_normlize = use_l2_normlize
        self.codebook_type = codebook_type
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.weight_init = weight_init

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(
                self.codebook_dim, self.input_dim, kernel_size=1
            )

        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        if self.codebook_type == "euclidean":
            self.codebook = EuclideanCodebook(
                self.codebook_dim,
                codebook_size=self.codebook_size,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                decay=self.decay,
                eps=self.eps,
                threshold_ema_dead_code=self.threshold_ema_dead_code,
                weight_init=self.weight_init,
            )
        elif self.codebook_type == "simple":
            self.codebook = SimpleCodebook(
                self.codebook_dim,
                codebook_size=self.codebook_size,
                use_l2_normlize=self.use_l2_normlize,
            )
        else:
            raise NotImplementedError(
                f"codebook_type {self.codebook_type} is not implemented!"
            )

    def forward(self, z):
        """
        Parameters
        ----------
        z: torch.Tensor[B x D x T]

        Returns
        -------
        z_q: torch.Tensor[B x D x T]
            Quantized continuous representation of input
        commit_loss: Tensor[B]
            Commitment loss to train encoder to predict vectors closer to codebook entries
        codebook_loss: Tensor[B]
            Codebook loss to update the codebook
        indices: torch.Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        z_e: torch.Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes project input into low-dimensional space if self.input_dim != self.codebook_dim
        z_e = self.in_project(z)
        z_q, indices = self.decode_latents(z_e)

        # Compute commitment loss and codebook loss
        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
                * self.codebook_loss_weight
            )
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)
            codebook_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = z_e + (z_q - z_e).detach()

        z_q = self.out_project(z_q)

        return z_q, commit_loss, codebook_loss, indices, z_e

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> b t d")
        z_q, indices = self.codebook(encodings)
        z_q = z_q.transpose(1, 2)
        return z_q, indices

    def vq2emb(self, vq, out_proj=True):
        emb = self.codebook.vq2emb(vq)
        emb = emb.transpose(1, 2)
        if out_proj:
            emb = self.out_project(emb)
        return emb

    def latent2dist(self, latents):
        latents = rearrange(latents, "b d t -> b t d")
        dist, embed_ind, quantize = self.codebook.latent2dist(latents)
        return dist, embed_ind, quantize.transpose(1, 2)
