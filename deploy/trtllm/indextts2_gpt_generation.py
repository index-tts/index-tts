"""TRT-LLM generation session wrapper for IndexTTS2 GPT."""

import torch
from tensorrt_llm.runtime import ModelConfig
from tensorrt_llm.runtime import GenerationSession
from tensorrt_llm import Mapping
from tensorrt_llm.runtime.generation import (
    _tile_beam_width,
    RuntimeTensor,
)

from typing import Optional


class IndexTTS2GPTForCausalLMGenerationSession(GenerationSession):

    def __init__(
        self,
        model_config: ModelConfig,
        engine_buffer,
        mapping: Mapping,
        debug_mode=False,
        debug_tensors_to_save=None,
        cuda_graph_mode=False,
        stream: torch.cuda.Stream = None,
        latents_condition_length: int = 34,
        text_position_id_offset: int = 1818,
        latents_condition_position_id: int = 2420,
    ):
        # Use debug_mode in parent to bypass strict tensor name validation,
        # since our engine has extra output 'last_hidden_state' the parent doesn't know about.
        super().__init__(
            model_config,
            engine_buffer,
            mapping,
            debug_mode=True,
            debug_tensors_to_save=debug_tensors_to_save,
            cuda_graph_mode=cuda_graph_mode,
            stream=stream,
        )
        self.debug_mode = debug_mode

        self.latents_condition_length = latents_condition_length
        self.text_position_id_offset = text_position_id_offset
        self.latents_condition_position_id = latents_condition_position_id
        self._outputs_hidden_states = []

    @property
    def vocab_size(self):
        return self._model_config.output_vocab_size

    def setup(self, batch_size, max_context_length, max_new_tokens, **kwargs):
        result = super().setup(batch_size, max_context_length, max_new_tokens, **kwargs)

        if self.mapping.is_last_pp_rank():
            self.buffer['last_hidden_state'] = torch.empty(
                (batch_size, self.hidden_size)
                if not self.gather_context_logits else
                (batch_size, max_context_length, self.hidden_size),
                dtype=self._tensor_dtype('last_hidden_state'),
                device=self.device)

        return result

    def _build_context_position_ids(self, seq_len):
        """Build position IDs for one sample in the context phase.

        Layout: [conds (pos=2420) | text (pos=1818,1819,...) | start_mel (pos=0)]
        """
        pos_ids = torch.zeros(seq_len, dtype=torch.int32, device='cuda')
        latents_len = min(self.latents_condition_length, seq_len)
        pos_ids[:latents_len] = self.latents_condition_position_id
        if seq_len > self.latents_condition_length + 1:
            middle_len = seq_len - self.latents_condition_length - 1
            pos_ids[self.latents_condition_length:self.latents_condition_length + middle_len] = \
                self.text_position_id_offset + torch.arange(middle_len, dtype=torch.int32, device='cuda')
        if seq_len > 0:
            pos_ids[-1] = 0
        return pos_ids

    def _prepare_context_inputs(self, batch_size, context_lengths,
                                host_context_lengths, use_gpt_attention_plugin,
                                remove_input_padding, **kwargs):
        ret = super()._prepare_context_inputs(
            batch_size, context_lengths, host_context_lengths,
            use_gpt_attention_plugin, remove_input_padding, **kwargs)

        if not (self.has_position_embedding and self.has_attn_layers):
            return ret

        if use_gpt_attention_plugin:
            if remove_input_padding:
                ret['position_ids'] = torch.concat([
                    self._build_context_position_ids(host_context_lengths[i])
                    for i in range(batch_size)
                ]).contiguous()
            else:
                max_ctx = host_context_lengths.max().item()
                position_ids = torch.zeros([batch_size, max_ctx],
                                          dtype=torch.int32, device='cuda')
                for i in range(batch_size):
                    sl = host_context_lengths[i].item()
                    position_ids[i, :sl] = self._build_context_position_ids(sl)
                ret['position_ids'] = position_ids
        elif self.has_attn_layers:
            attention_mask = ret['attention_mask']
            position_ids = torch.zeros_like(attention_mask, dtype=torch.int32)
            for i in range(batch_size):
                sl = (attention_mask[i] == 1).sum().item()
                if sl > 0:
                    position_ids[i, :sl] = self._build_context_position_ids(sl)
            ret['position_ids'] = position_ids.int()

        return ret

    def _prepare_generation_inputs(self, batch_size, context_lengths,
                                   use_gpt_attention_plugin,
                                   remove_input_padding, **kwargs):
        step = kwargs['step']
        ret = super()._prepare_generation_inputs(
            batch_size, context_lengths, use_gpt_attention_plugin,
            remove_input_padding, **kwargs)

        if not (self.has_position_embedding and self.has_attn_layers):
            return ret

        if use_gpt_attention_plugin and not self.is_redrafter_mode:
            position_ids = torch.full_like(context_lengths, step + 1,
                                          dtype=torch.int32).contiguous()
            if not remove_input_padding:
                position_ids = torch.unsqueeze(position_ids, 1)
            ret['position_ids'] = position_ids
        elif not use_gpt_attention_plugin and self.has_attn_layers:
            num_beams = ret.get('attention_mask', context_lengths).shape[0] // batch_size
            ret['position_ids'] = torch.full(
                (batch_size * num_beams, 1), step + 1,
                dtype=torch.int32, device='cuda')

        return ret

    def _get_context_shape_buffer(self, *args, **kwargs):
        tensors = super()._get_context_shape_buffer(*args, **kwargs)
        if self.mapping.is_last_pp_rank():
            tensors['last_hidden_state'] = RuntimeTensor.from_torch(
                'last_hidden_state', self.buffer['last_hidden_state'])
        return tensors

    def _get_next_step_shape_buffer(self, *args, **kwargs):
        tensors = super()._get_next_step_shape_buffer(*args, **kwargs)
        if self.mapping.is_last_pp_rank():
            tensors['last_hidden_state'] = RuntimeTensor.from_torch(
                'last_hidden_state', self.buffer['last_hidden_state'])
        return tensors

    def handle_per_step(self, **kwargs):
        input_context_lengths = kwargs['context_lengths']
        result = super().handle_per_step(**kwargs)

        step = kwargs['step']
        batch_size = kwargs['batch_size']
        beam_width = kwargs['beam_width']

        if self.mapping.is_last_pp_rank():
            if step == 0 and self.gather_context_logits:
                if self.remove_input_padding:
                    last_token_ids = input_context_lengths.detach().clone()
                    if self.use_gpt_attention_plugin:
                        last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
                    self.buffer['last_hidden_state'] = self.buffer['last_hidden_state'].reshape(
                        [1, -1, self.hidden_size])
                    self.buffer['last_hidden_state'] = torch.index_select(
                        self.buffer['last_hidden_state'], 1,
                        last_token_ids - 1).view(batch_size, self.hidden_size)
                else:
                    last_token_ids = input_context_lengths.detach().clone()
                    last_token_ids = last_token_ids.reshape(batch_size, 1, 1)
                    last_token_ids_hs = last_token_ids.expand(
                        batch_size, 1, self.hidden_size) - 1
                    self.buffer['last_hidden_state'] = torch.gather(
                        self.buffer['last_hidden_state'],
                        dim=1,
                        index=last_token_ids_hs.to(dtype=torch.int64)).view(
                            batch_size, self.hidden_size)

            if step == 0 and beam_width > 1:
                self.buffer['last_hidden_state'] = _tile_beam_width(
                    self.buffer['last_hidden_state'], beam_width)

            self._outputs_hidden_states.append(
                self.buffer['last_hidden_state'].detach().clone())

        return result

    def decode_regular(self, **kwargs):
        self._outputs_hidden_states = []
        result = super().decode_regular(**kwargs)
        if isinstance(result, dict):
            result['last_hidden_state'] = self._outputs_hidden_states
        return result

    def decode_stream(self, **kwargs):
        self._outputs_hidden_states = []
        gen = super().decode_stream(**kwargs)
        for output in gen:
            if isinstance(output, dict) and self.mapping.is_last_pp_rank():
                output['last_hidden_state'] = self._outputs_hidden_states[-1]
            yield output

