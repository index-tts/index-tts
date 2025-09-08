# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange
from indextts.utils.maskgct.models.tts.maskgct.llama_nar import DiffLlama


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class MaskGCT_S2A(nn.Module):
    def __init__(
        self,
        num_quantizer=12,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        codebook_size=1024,
        cfg_scale=0.15,
        mask_layer_schedule="linear",
        cond_codebook_size=1024,
        cond_dim=1024,
        predict_layer_1=True,
        cfg=None,
    ):
        super().__init__()

        num_quantizer = (
            cfg.num_quantizer
            if cfg is not None and hasattr(cfg, "num_quantizer")
            else num_quantizer
        )
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        num_layers = (
            cfg.num_layers
            if cfg is not None and hasattr(cfg, "num_layers")
            else num_layers
        )
        num_heads = (
            cfg.num_heads
            if cfg is not None and hasattr(cfg, "num_heads")
            else num_heads
        )
        codebook_size = (
            cfg.codebook_size
            if cfg is not None and hasattr(cfg, "codebook_size")
            else codebook_size
        )
        cfg_scale = (
            cfg.cfg_scale
            if cfg is not None and hasattr(cfg, "cfg_scale")
            else cfg_scale
        )
        mask_layer_schedule = (
            cfg.mask_layer_schedule
            if cfg is not None and hasattr(cfg, "mask_layer_schedule")
            else mask_layer_schedule
        )
        cond_codebook_size = (
            cfg.cond_codebook_size
            if cfg is not None and hasattr(cfg, "cond_codebook_size")
            else cond_codebook_size
        )
        cond_dim = (
            cfg.cond_dim if cfg is not None and hasattr(cfg, "cond_dim") else cond_dim
        )
        predict_layer_1 = (
            cfg.predict_layer_1
            if cfg is not None and hasattr(cfg, "predict_layer_1")
            else predict_layer_1
        )

        self.num_quantizer = num_quantizer
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.mask_layer_schedule = mask_layer_schedule
        self.cond_codebook_size = cond_codebook_size
        self.cond_dim = cond_dim
        self.predict_layer_1 = predict_layer_1

        self.layer_emb = nn.Embedding(self.num_quantizer, self.hidden_size)
        self.mask_emb = nn.Embedding(1, self.hidden_size)

        self.token_emb = torch.nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, self.hidden_size)
                for _ in range(self.num_quantizer)
            ]
        )

        self.to_logits = torch.nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.codebook_size)
                for _ in range(self.num_quantizer)
            ]
        )

        self.cond_emb = nn.Embedding(cond_codebook_size, self.hidden_size)

        self.reset_parameters()

        self.diff_estimator = DiffLlama(
            hidden_size=hidden_size,
            num_heads=self.num_heads,
            num_layers=num_layers,
        )

    def mask_prob(self, t):
        return torch.sin(t * np.pi / 2).to(t.device)

    def mask_layer(self, t):
        # print(self.predict_layer_1)
        if self.mask_layer_schedule == "uniform":
            if self.predict_layer_1:
                mask_layer = torch.randint(0, self.num_quantizer, (1,)).to(t.device)
            else:
                mask_layer = torch.randint(1, self.num_quantizer, (1,)).to(t.device)
        elif self.mask_layer_schedule == "cosine":
            if self.predict_layer_1:
                weights = torch.tensor(
                    [
                        np.cos(i / self.num_quantizer * np.pi / 2)
                        for i in range(self.num_quantizer)
                    ]
                )
            else:
                weights = torch.tensor(
                    [0]
                    + [
                        np.cos((i - 1) / self.num_quantizer * np.pi / 2)
                        for i in range(1, self.num_quantizer)
                    ]
                )
            mask_layer = torch.multinomial(weights, 1).to(t.device)
        elif self.mask_layer_schedule == "linear":
            if self.predict_layer_1:
                weights = torch.tensor(
                    [self.num_quantizer - i for i in range(self.num_quantizer)]
                )
            else:
                weights = torch.tensor(
                    [0]
                    + [
                        self.num_quantizer - (i - 1)
                        for i in range(1, self.num_quantizer)
                    ]
                )
            weights = weights / weights.sum()
            mask_layer = torch.multinomial(weights, 1).to(t.device)
        # print(mask_layer)
        new_t = t

        return mask_layer, new_t

    def forward_diffusion(self, x0, t):
        # x0: (B, T, num_quantizer)
        mask_layer, new_t = self.mask_layer(t)  # (1,)
        mask_prob = self.mask_prob(new_t)  # (B,)
        mask_token = self.mask_emb(torch.zeros_like(mask_layer))  # (1, hidden_size)

        xt = torch.zeros(x0.shape[0], x0.shape[1], self.hidden_size).to(x0.device)

        cfg_scale = self.cfg_scale

        # get prompt len
        if torch.rand(1) > cfg_scale:
            prompt_len = torch.randint(
                min(x0.shape[1] // 4, 5), x0.shape[1] // 2, (x0.shape[0],)
            ).to(
                x0.device
            )  # (B,)
        else:
            prompt_len = torch.zeros(x0.shape[0]).to(x0)  # (B,)

        # get is prompt
        is_prompt = torch.zeros_like(x0[:, :, 0])  # (B, T)
        col_indices = (
            torch.arange(is_prompt.shape[1])
            .repeat(is_prompt.shape[0], 1)
            .to(prompt_len)
        )  # (B, T)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1  # (B, T) 1 if prompt

        for idx, token_emb_idx in enumerate(self.token_emb):
            if idx < mask_layer:
                xt = xt + token_emb_idx(x0[:, :, idx])  # (B, T, hidden_size)

            elif idx == mask_layer:
                mask = torch.bernoulli(
                    torch.ones_like(x0[:, :, idx]) * mask_prob[..., None]
                )  # mask if 1, not mask if 0
                # prompt part don't need to be masked
                mask[is_prompt.bool()] = 0
                # Ensure at least one token is masked
                mask_num = mask[:,].sum(dim=1, keepdim=False)
                all_zero_mask = (mask_num == 0).bool()
                row_indices_to_modify = torch.nonzero(all_zero_mask)
                # mask the first token if all tokens are not masked (may mask pad if random indices)
                mask[row_indices_to_modify, prompt_len[row_indices_to_modify]] = 1

                mask = mask[..., None]  # (B, T, 1)
                xt = (
                    xt
                    + mask * mask_token[:, None, :]
                    + (1 - mask) * token_emb_idx(x0[:, :, idx])
                )  # (B, T, hidden_size)

            else:
                # prompt part don't need to be masked
                xt = (
                    xt
                    + token_emb_idx(x0[:, :, idx]) * is_prompt[..., None]
                    + mask_token * (1 - is_prompt[..., None])
                )

        return xt, new_t, mask_layer, mask, prompt_len, mask_prob

    def loss_t(self, x0, x_mask, t, cond=None):
        xt, new_t, mask_layer, mask, prompt_len, mask_prob = self.forward_diffusion(
            x0, t
        )
        # xt: (B, T, hidden_size)
        # new_t: (B,)
        # mask_layer: (1,)
        # mask: (B, T, 1)   mask if 1, not mask if 0
        # prompt_len: (B,)
        # mask_prob: (B,)

        mask_layer_cond = self.layer_emb(mask_layer).unsqueeze(1)  # (1, 1, hidden_size)
        cond = cond + mask_layer_cond  # (B, T, hidden_size)

        embeds = self.diff_estimator(xt, new_t, cond, x_mask)  # (B, T, hidden_size)

        logits = self.to_logits[mask_layer.item()](embeds)  # (B, T, codebook_size)

        # final mask used for loss calculation
        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        return logits, mask_layer, final_mask, x0, prompt_len, mask_prob

    def compute_loss(self, x0, x_mask, cond=None):
        # x0: (B, T, num_quantizer)
        # x_mask: (B, T) mask is 0 for padding
        t = torch.rand(x0.shape[0], device=x0.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        return self.loss_t(x0, x_mask, t, cond)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    @torch.no_grad()
    def reverse_diffusion(
        self,
        cond,
        prompt,
        x_mask=None,
        prompt_mask=None,
        temp=1.5,
        filter_thres=0.98,
        max_layer=None,
        gt_code=None,
        n_timesteps=[10, 4, 4, 4, 4, 4, 4, 4],
        cfg=1.0,
        rescale_cfg=1.0,
    ):

        assert (
            len(n_timesteps) == self.num_quantizer
        )  # each layer has a number of steps

        prompt_code = prompt  # (B, prompt_len, num_quantizer)
        prompt_len = prompt_code.shape[1]
        target_len = cond.shape[1] - prompt_len

        if x_mask == None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)  # (B, T)
        if prompt_mask == None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len).to(
                cond.device
            )  # (B, prompt_len)

        cum = torch.zeros(x_mask.shape[0], x_mask.shape[1], self.hidden_size).to(
            x_mask.device
        )  # (B, T, hidden_size)

        bsz, seq_len, _ = cum.shape

        choice_temp = 1.0
        start_temp = temp  # temperature for sampling
        start_choice_temp = choice_temp  # temperature for choicing mask tokens

        if max_layer is None:
            max_layer = self.num_quantizer

        xt = torch.LongTensor(bsz, seq_len, max_layer).to(x_mask.device)

        if gt_code is not None:
            gt_layer = gt_code.shape[-1]
            xt[:, :, :gt_layer] = gt_code
            for i in range(gt_layer):
                cum += self.token_emb[i](xt[:, :, i])
        else:
            gt_layer = 0

        for mask_layer in range(gt_layer, max_layer):
            steps = n_timesteps[mask_layer]
            to_logits = self.to_logits[mask_layer]
            token_emb = self.token_emb[mask_layer]
            mask_layer = torch.tensor(mask_layer).to(x_mask.device).long().unsqueeze(0)
            mask_layer_cond = self.layer_emb(mask_layer).unsqueeze(
                1
            )  # (1,) -> (1, 1, hidden_size)
            temp_cond = cond + mask_layer_cond  # (B, T, hidden_size)

            mask_token = self.mask_emb(torch.zeros_like(mask_layer))  # (1, hidden_size)
            mask = torch.full((bsz, seq_len, 1), True).to(x_mask.device)  # (B, T, 1)
            seq = torch.full((bsz, seq_len), 0).to(x_mask.device)

            h = 1.0 / steps

            # prompt_code: (B, prompt_len, num_quantizer)
            cur_prompt = 0
            for idx, emb in enumerate(self.token_emb):
                cur_prompt = cur_prompt + emb(
                    prompt_code[:, :, idx]
                )  # (B, prompt_len, hidden_size)

            t_list = [1.0 - i * h for i in range(steps)]
            t_list.append(0.0)
            for i in range(steps):
                t = t_list[i] * torch.ones(bsz).to(x_mask.device)
                token = token_emb(seq)  # (B, T, hidden_size)
                cur = cum + mask * mask_token[:, None, :] + (~mask) * token
                cur = cur + mask_token[:, None, :] * (max_layer - 1 - mask_layer)

                xt_input = torch.cat([cur_prompt, cur], dim=1)  # (B, T, hidden_size)
                xt_mask = torch.cat(
                    [prompt_mask, x_mask], dim=1
                )  # (B, T), mask is 0 for padding

                embeds = self.diff_estimator(xt_input, t, temp_cond, xt_mask)
                embeds = embeds[:, prompt_len:, :]

                # cfg
                if cfg > 0:
                    mask_embeds = self.diff_estimator(
                        cur, t, temp_cond[:, prompt_len:, :], x_mask
                    )
                    pos_emb_std = embeds.std()  # std(g_cond)
                    embeds = embeds + cfg * (embeds - mask_embeds)  # g_cfg
                    rescale_embeds = embeds * pos_emb_std / embeds.std()  # g_final
                    embeds = rescale_cfg * rescale_embeds + (1 - rescale_cfg) * embeds

                logits = to_logits(embeds)  # (B, T, codebook_size)
                annealing_scale = t_list[i]

                choice_temp = start_choice_temp * annealing_scale
                temp = start_temp * annealing_scale
                logits = top_k(logits, filter_thres)

                if i == steps - 1:
                    # greedy
                    if steps == 1:
                        temp = 0.2
                        sampled_ids = gumbel_sample(logits, temperature=max(temp, 1e-3))
                    else:
                        sampled_ids = logits.argmax(dim=-1)

                else:
                    # sampling
                    sampled_ids = gumbel_sample(logits, temperature=max(temp, 1e-3))

                seq = torch.where(mask.squeeze(-1), sampled_ids, seq)

                scores = logits.softmax(dim=-1)
                scores = scores.gather(2, rearrange(sampled_ids, "b n -> b n 1"))
                scores = rearrange(scores, "b n 1 -> b n")

                scores = choice_temp * gumbel_noise(scores) + scores
                scores = 1 - scores

                next_t = t_list[i + 1] * torch.ones(bsz).to(x_mask.device)

                next_mask_num = (self.mask_prob(next_t) * seq_len).long()[0].item()

                if next_mask_num == 0:
                    break
                scores = scores.masked_fill(
                    ~mask.squeeze(-1), -torch.finfo(scores.dtype).max
                )

                mask_indices = scores.topk(next_mask_num, dim=-1).indices
                mask = torch.zeros_like(scores, dtype=torch.bool).scatter(
                    1, mask_indices, True
                )
                seq = seq.masked_fill(mask, 0)

                mask = mask.unsqueeze(-1)

            cum = cum + token_emb(seq)
            xt[..., mask_layer.squeeze(0).item()] = seq

        return xt

    def forward(self, x0, x_mask, cond_code=None):
        # x0: (B, T, num_quantizer)
        # x_mask: (B, T) mask is 0 for padding
        # cond_code: semantic token (B, T)
        cond = self.cond_emb(cond_code)

        logits, mask_layer, final_mask, x0, prompt_len, mask_prob = self.compute_loss(
            x0,
            x_mask,
            cond,
        )
        return logits, mask_layer, final_mask, x0, prompt_len, mask_prob
