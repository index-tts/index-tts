import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.arch_util import AttentionBlock
from indextts.utils.typical_sampling import TypicalLogitsWarper


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class GPT2InferenceModel(GPT2PreTrainedModel):
    def __init__(
        self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=False
    ):
        super().__init__(config)
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(
                len(self.transformer.h), range(max(1, torch.cuda.device_count()))
            )
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(
                    text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
                )
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - mel_len, attention_mask.device
            )
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


class ConditioningEncoder(nn.Module):
    def __init__(
        self,
        spec_dim,
        embedding_dim,
        attn_blocks=6,
        num_attn_heads=4,
        do_checkpointing=False,
        mean=False,
    ):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h
            # return h[:, :, 0]


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(
    layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing
):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model

    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len,
        n_ctx=max_mel_seq_len + max_text_seq_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=checkpointing,
        use_cache=not checkpointing,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return (
        gpt,
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim),
        LearnedPositionEmbeddings(max_text_seq_len, model_dim),
        None,
        None,
    )


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1),
            nn.Sequential(
                *[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]
            ),
            nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(channels // 16, channels // 2),
            nn.ReLU(),
            nn.Sequential(
                *[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]
            ),
            nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(channels // 8, channels),
            nn.ReLU(),
            nn.Sequential(
                *[ResBlock(channels) for _ in range(resblocks_per_reduction)]
            ),
        )
        self.reduction = 4

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x.permute(0, 2, 1)


class UnifiedVoice(nn.Module):
    def __init__(
        self,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_conditioning_inputs=1,
        mel_length_compression=1024,
        number_text_tokens=256,
        start_text_token=0,
        stop_text_token=1,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        train_solo_embeddings=False,
        use_mel_codes_as_input=True,
        checkpointing=True,
        types=1,
        condition_num_latent=32,
        condition_type="perceiver",
        condition_module=None,
    ):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
            condition_type: perceiver, gst or default encoder
        """
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        if condition_type == "perceiver":
            self.conditioning_encoder = ConditioningEncoder(
                100, model_dim, num_attn_heads=heads
            )
            self.perceiver_encoder = PerceiverResampler(
                model_dim, dim_context=model_dim, num_latents=self.cond_num
            )
        elif (
            condition_type == "conformer_perceiver"
            or condition_type == "conformer_encoder"
        ):
            self.conditioning_encoder = ConformerEncoder(
                input_size=100,
                output_size=condition_module["output_size"],
                linear_units=condition_module["linear_units"],
                attention_heads=condition_module["attention_heads"],
                num_blocks=condition_module["num_blocks"],
                input_layer=condition_module["input_layer"],
            )
            if condition_type == "conformer_perceiver":
                self.perceiver_encoder = PerceiverResampler(
                    model_dim,
                    dim_context=condition_module["output_size"],
                    ff_mult=condition_module["perceiver_mult"],
                    heads=condition_module["attention_heads"],
                    num_latents=self.cond_num,
                )
        else:
            self.conditioning_encoder = ConditioningEncoder(
                100, model_dim, num_attn_heads=heads, mean=True
            )

        self.text_embedding = nn.Embedding(
            self.number_text_tokens * types + 1, model_dim
        )
        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)
        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            self.mel_layer_pos_embedding,
            self.text_layer_pos_embedding,
        ) = build_hf_gpt_transformer(
            layers,
            model_dim,
            heads,
            self.max_mel_tokens + 2 + self.max_conditioning_inputs,
            self.max_text_tokens + 2,
            checkpointing,
        )
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(
                torch.randn(1, 1, model_dim) * 0.02, requires_grad=True
            )
            self.text_solo_embedding = nn.Parameter(
                torch.randn(1, 1, model_dim) * 0.02, requires_grad=True
            )
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.max_mel_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model,
                mp_size=1,
                replace_with_kernel_inject=False,
                dtype=torch.float16,
            )
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model,
                mp_size=1,
                replace_with_kernel_inject=False,
                dtype=torch.float32,
            )
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        # self.inference_model = PrunedGPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
        self.gpt.wte = self.mel_embedding

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_text_padding_compiled(self, text_input_tokens, text_lengths):
        """
        使用张量操作代替条件语句的版本，用于设置文本填充
        """
        batch_size = text_input_tokens.shape[0]
        seq_length = text_input_tokens.shape[1]

        # 创建一个表示位置的张量
        positions = torch.arange(seq_length, device=text_input_tokens.device).expand(
            batch_size, seq_length
        )

        # 创建一个表示每个位置是否超过实际长度的掩码
        mask = positions >= text_lengths.unsqueeze(1)

        # 使用掩码操作替代条件语句
        text_input_tokens = text_input_tokens.masked_fill(mask, self.stop_text_token)

        return text_input_tokens

    def set_mel_padding_compiled(self, mel_input_tokens, mel_lengths):
        """
        兼容版本的 set_mel_padding。
        当 mel_input_tokens 的 batch_size 小于 mel_lengths 的长度时，
        只使用 mel_lengths 中对应 batch_size 数量的值。

        Args:
            mel_input_tokens: shape [B, seq_len]，其中 B 可能小于 len(mel_lengths)
            mel_lengths: shape [L]，其中 L >= B
        Returns:
            shape 与输入 mel_input_tokens 相同
        """
        batch_size, seq_len = mel_input_tokens.shape
        
        # 只使用需要的长度值
        effective_lengths = mel_lengths[:batch_size]
        
        # 创建位置索引 [B, seq_len]
        position_idx = torch.arange(seq_len, device=mel_input_tokens.device)
        position_idx = position_idx[None, :].expand(batch_size, -1)
        
        # 创建掩码 [B, seq_len]
        mask = position_idx >= effective_lengths[:, None]
        
        # 使用掩码更新 tensor
        mel_input_tokens = torch.where(mask, self.stop_mel_token, mel_input_tokens)
        
        return mel_input_tokens


    
    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(
        self,
        speech_conditioning_inputs,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        get_attns=False,
        return_latent=False,
    ):
        if second_inputs is not None:
            emb = torch.cat(
                [speech_conditioning_inputs, first_inputs, second_inputs], dim=1
            )
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(
            inputs_embeds=emb, return_dict=True, output_attentions=get_attns
        )
        if get_attns:
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        if self.condition_type == "perceiver":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            speech_conditioning_input = self.conditioning_encoder(
                speech_conditioning_input
            )  # (b, d, s)
            conds = self.perceiver_encoder(
                speech_conditioning_input.transpose(1, 2)
            )  # (b, 32, d)
        elif self.condition_type == "conformer_perceiver":
            speech_conditioning_input, mask = self.conditioning_encoder(
                speech_conditioning_input.transpose(1, 2), cond_mel_lengths
            )  # (b, s, d), (b, 1, s)
            if self.condition_type == "conformer_perceiver":
                # conds_mask = torch.cat([torch.ones((mask.shape[0], self.cond_num), dtype=torch.bool), mask.squeeze(1)], dim=1)
                conds_mask = self.cond_mask_pad(mask.squeeze(1))
                conds = self.perceiver_encoder(
                    speech_conditioning_input, conds_mask
                )  # (b, 32, d)
        elif self.condition_type == "gst":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            conds = self.gst_encoder(
                speech_conditioning_input.transpose(1, 2)
            )  # (b, 1, d)
        else:
            speech_conditioning_input = (
                speech_conditioning_input.unsqueeze(1)
                if len(speech_conditioning_input.shape) == 3
                else speech_conditioning_input
            )
            conds = []
            for j in range(speech_conditioning_input.shape[1]):
                conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
            conds = torch.stack(conds, dim=1)
            conds = conds.mean(dim=1)
            conds = conds.unsqueeze(1)
        return conds

    @torch.compile(
        fullgraph=True,  # 整个图编译，提供最大优化
        backend=(
            "inductor" if torch.cuda.is_available() else "aot_eager"
        ),  # 根据设备选择后端
        mode=(
            "reduce-overhead" if torch.cuda.is_available() else None
        ),  # GPU上使用减少开销模式
    )
    def optimized_forward(
        self,
        speech_conditioning_latent,
        text_inputs,
        text_lengths,
        mel_codes,
        wav_lengths,
        cond_mel_lengths,
        return_latent=True,
        clip_inputs=False,
    ):
        """优化的前向传播函数"""
        return self.forward(
            speech_conditioning_latent=speech_conditioning_latent,
            text_inputs=text_inputs,
            text_lengths=text_lengths,
            mel_codes=mel_codes,
            wav_lengths=wav_lengths,
            cond_mel_lengths=cond_mel_lengths,
            return_latent=return_latent,
            clip_inputs=clip_inputs,
        )

    def forward(
        self,
        speech_conditioning_latent,
        text_inputs,
        text_lengths,
        mel_codes,
        wav_lengths,
        cond_mel_lengths=None,
        types=None,
        text_first=True,
        raw_mels=None,
        return_attentions=False,
        return_latent=False,
        clip_inputs=False,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        speech_conditioning_input: MEL float tensor, (b,1024)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        raw_mels: MEL float tensor (b,80,s)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        If clip_inputs is True, the inputs will be clipped to the smallest input size across each input modality.
        """

        speech_conditioning_latent = self.get_conditioning(
            speech_conditioning_latent, cond_mel_lengths
        )
        # Types are expressed by expanding the text embedding space.
        if types is not None:
            text_inputs = text_inputs * (1 + types).unsqueeze(-1)

        if clip_inputs:
            # This model will receive micro-batches with a ton of padding for both the text and MELs. Ameliorate this by
            # chopping the inputs by the maximum actual length.
            max_text_len = text_lengths.max()
            text_inputs = text_inputs[:, :max_text_len]
            max_mel_len = wav_lengths.max() // self.mel_length_compression
            mel_codes = mel_codes[:, :max_mel_len]
            if raw_mels is not None:
                raw_mels = raw_mels[:, :, : max_mel_len * 4]

        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        # mel_codes_lengths = torch.div(wav_lengths, self.mel_length_compression, rounding_mode='trunc')
        mel_codes_lengths = (
            torch.ceil(wav_lengths / self.mel_length_compression).long() + 1
        )

        # mel_codes_copy = mel_codes.clone()

        # print(f"mel_codes.shape: {mel_codes.shape} mel_codes_lengths.shape: {mel_codes_lengths.shape} mel_codes_copy.shape: {mel_codes_copy.shape}")

        # mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = self.set_mel_padding_compiled(mel_codes, mel_codes_lengths)

        # mel_codes_compiled = self.set_mel_padding_compiled(mel_codes_copy, mel_codes_lengths)

        

        # print(f"mel_codes.shape: {mel_codes.shape} mel_codes_compiled.shape: {mel_codes_compiled.shape}")

        # text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = self.set_text_padding_compiled(text_inputs, text_lengths)
        # text_inputs_compiled = self.set_text_padding_compiled(text_inputs, text_lengths)

        # print(f"text_inputs.shape: {text_inputs.shape} text_inputs_compiled.shape: {text_inputs_compiled.shape}")

        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        conds = speech_conditioning_latent
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(
            text_inputs
        )
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )
        if raw_mels is not None:
            mel_inp = F.pad(raw_mels, (0, 8))
        else:
            mel_inp = mel_codes
        mel_emb = self.mel_embedding(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        if text_first:
            # print(
            #     f"conds: {conds.shape}, text_emb: {text_emb.shape}, mel_emb: {mel_emb.shape}"
            # )
            text_logits, mel_logits = self.get_logits(
                speech_conditioning_inputs=conds,
                first_inputs=text_emb,
                first_head=self.text_head,
                second_inputs=mel_emb,
                second_head=self.mel_head,
                get_attns=return_attentions,
                return_latent=return_latent,
            )
            if return_latent:
                return mel_logits[
                    :, :-2
                ]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.
        else:
            mel_logits, text_logits = self.get_logits(
                conds,
                mel_emb,
                self.mel_head,
                text_emb,
                self.text_head,
                get_attns=return_attentions,
                return_latent=return_latent,
            )
            if return_latent:
                return text_logits[
                    :, :-2
                ]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

        if return_attentions:
            return mel_logits

        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits

    # 在UnifiedVoice类中添加这个新方法
    @torch.compile(
        fullgraph=False,
        backend="inductor" if torch.cuda.is_available() else "aot_eager",
        mode="reduce-overhead" if torch.cuda.is_available() else None,
    )
    def prepare_inference_inputs(
        self, speech_conditioning_latent, text_inputs, cond_mel_lengths=None
    ):
        """
        优化的输入预处理函数，专门为torch.compile设计，处理推理过程中的嵌入计算部分

        Args:
            speech_conditioning_latent: 语音条件输入特征，通常是参考音频的mel谱图
            text_inputs: 文本输入tokens
            cond_mel_lengths: 条件mel长度 (可选)

        Returns:
            tuple: (
                嵌入向量,
                文本输入设备,
                填充的文本tokens (用于后续处理)
            )
        """
        # 文本输入处理
        text_inputs_padded = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs_aligned, _ = self.build_aligned_inputs_and_targets(
            text_inputs_padded, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(text_inputs_aligned) + self.text_pos_embedding(
            text_inputs_aligned
        )

        # 语音条件处理
        speech_conds = self.get_conditioning(
            speech_conditioning_latent, cond_mel_lengths
        )

        # 组合嵌入
        combined_emb = torch.cat([speech_conds, text_emb], dim=1)

        return combined_emb, text_inputs.device, text_inputs_aligned

    def inference_speech(
        self,
        speech_conditioning_latent,
        text_inputs,
        cond_mel_lengths=None,
        input_tokens=None,
        num_return_sequences=1,
        max_generate_length=None,
        typical_sampling=False,
        typical_mass=0.9,
        **hf_generate_kwargs,
    ):

        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs, _ = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(
            text_inputs
        )

        speech_conditioning_latent = self.get_conditioning(
            speech_conditioning_latent, cond_mel_lengths
        )
        conds = speech_conditioning_latent
        emb = torch.cat([conds, text_emb], dim=1)
        self.inference_model.store_mel_emb(emb)

        # +1 for the start_audio_token
        fake_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )

        fake_inputs[:, -1] = self.start_mel_token
        trunc_index = fake_inputs.shape[1]
        if input_tokens is None:
            inputs = fake_inputs
        else:
            assert (
                num_return_sequences % input_tokens.shape[0] == 0
            ), "The number of return sequences must be divisible by the number of input sequences"
            fake_inputs = fake_inputs.repeat(num_return_sequences, 1)
            input_tokens = input_tokens.repeat(
                num_return_sequences // input_tokens.shape[0], 1
            )
            inputs = torch.cat([fake_inputs, input_tokens], dim=1)

        logits_processor = (
            LogitsProcessorList([TypicalLogitsWarper(mass=typical_mass)])
            if typical_sampling
            else LogitsProcessorList()
        )
        max_length = (
            trunc_index + self.max_mel_tokens - 1
            if max_generate_length is None
            else trunc_index + max_generate_length
        )
        gen = self.inference_model.generate(
            inputs,
            bos_token_id=self.start_mel_token,
            pad_token_id=self.stop_mel_token,
            eos_token_id=self.stop_mel_token,
            max_length=max_length,
            logits_processor=logits_processor,
            num_return_sequences=num_return_sequences,
            **hf_generate_kwargs,
        )
        return gen[:, trunc_index:]

    def inference_speech_optimized(
        self,
        speech_conditioning_latent,
        text_inputs,
        cond_mel_lengths=None,
        input_tokens=None,
        num_return_sequences=1,
        max_generate_length=None,
        typical_sampling=False,
        typical_mass=0.9,
        **hf_generate_kwargs,
    ):
        """
        使用torch.compile优化的语音推理方法
        """
        # 使用编译后的函数处理输入
        emb, device, text_inputs_aligned = self.prepare_inference_inputs(
            speech_conditioning_latent, text_inputs, cond_mel_lengths
        )

        # 存储嵌入用于推理模型
        self.inference_model.store_mel_emb(emb)

        # 创建假输入，+1 是为了start_audio_token
        fake_inputs = torch.full(
            (emb.shape[0], emb.shape[1] + 1),
            fill_value=1,
            dtype=torch.long,
            device=device,
        )

        fake_inputs[:, -1] = self.start_mel_token
        trunc_index = fake_inputs.shape[1]
        if input_tokens is None:
            inputs = fake_inputs
        else:
            assert (
                num_return_sequences % input_tokens.shape[0] == 0
            ), "The number of return sequences must be divisible by the number of input sequences"
            fake_inputs = fake_inputs.repeat(num_return_sequences, 1)
            input_tokens = input_tokens.repeat(
                num_return_sequences // input_tokens.shape[0], 1
            )
            inputs = torch.cat([fake_inputs, input_tokens], dim=1)

        logits_processor = (
            LogitsProcessorList([TypicalLogitsWarper(mass=typical_mass)])
            if typical_sampling
            else LogitsProcessorList()
        )
        max_length = (
            trunc_index + self.max_mel_tokens - 1
            if max_generate_length is None
            else trunc_index + max_generate_length
        )
        gen = self.inference_model.generate(
            inputs,
            bos_token_id=self.start_mel_token,
            pad_token_id=self.stop_mel_token,
            eos_token_id=self.stop_mel_token,
            max_length=max_length,
            logits_processor=logits_processor,
            num_return_sequences=num_return_sequences,
            **hf_generate_kwargs,
        )
        return gen[:, trunc_index:]
