"""IndexTTS2 GPT model definition for TRT-LLM.
"""

from typing import Optional, Union

from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.functional import (Tensor, is_gated_activation, non_gated_version, recv,
                           send, cp_split_plugin, allgather, view, index_select,
                           gather_last_token_logits, tanh)
from tensorrt_llm.layers import (MLP, MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, LayerNorm, MoeConfig,
                       PositionEmbeddingType)
from tensorrt_llm.lora_manager import LoraConfig, use_lora
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.module import Module
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models.modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig, Gemma2ConfigGroup, Gemma3ConfigGroup)
from tensorrt_llm.models.gpt.config import GPTConfig
from tensorrt_llm.models.gpt.model import GPTModel
from tensorrt_llm._torch.models.modeling_utils import register_auto_model
from tensorrt_llm._common import default_net

from backends.trt.trtllm.convert_gpt_weights import load_weights_from_hf_model, load_hf_weights_and_config


@register_auto_model("IndexTTS2GPTForCausalLM")
class IndexTTS2GPTForCausalLM(DecoderModelForCausalLM):
    config_class = GPTConfig

    def __init__(self, config: GPTConfig):
        transformer = GPTModel(config)

        if config.mapping.is_last_pp_rank():
            output_vocab_size_padded = pad_vocab_size(config.output_vocab_size,
                                               config.mapping.tp_size)
            lm_head = ColumnLinear(config.hidden_size,
                                   output_vocab_size_padded,
                                   bias=True,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        super().__init__(config, transformer, lm_head)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                mrope_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None,
                spec_decoding_params=None):

        attention_params = Attention.fill_attention_params(
            self, attention_params)

        if self.config.mapping.cp_size > 1:
            if len(input_ids.shape) == 1:
                input_ids, cp_join_index = cp_split_plugin(
                    input_ids,
                    attention_params.host_request_types,
                    attention_params.host_context_lengths,
                    self.config.mapping.cp_size,
                    self.config.mapping.cp_rank,
                )
            else:
                assert False, "Context parallelism with non-remove-padding is not supported yet."

        is_gemma_2_cg = self.config.has_config_group(Gemma2ConfigGroup)
        is_gemma_3_cg = self.config.has_config_group(Gemma3ConfigGroup)

        kwargs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'kv_cache_params': kv_cache_params,
            'attention_params': attention_params,
        }
        if lora_params is not None:
            kwargs['lora_params'] = lora_params
        if hidden_states is not None:
            kwargs['hidden_states'] = hidden_states
        if prompt_embedding_table is not None:
            kwargs['prompt_embedding_table'] = prompt_embedding_table
        if prompt_tasks is not None:
            kwargs['prompt_tasks'] = prompt_tasks
        if prompt_vocab_size is not None:
            kwargs['prompt_vocab_size'] = prompt_vocab_size
        if spec_decoding_params is not None:
            kwargs['spec_decoding_params'] = spec_decoding_params
        if mrope_params is not None:
            kwargs['mrope_params'] = mrope_params

        hidden_states = self.transformer.forward(**kwargs)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.config.mapping.cp_size > 1:
            if len(hidden_states.shape) == 2:
                hidden_states = allgather(hidden_states,
                                          self.config.mapping.cp_group,
                                          gather_dim=0)
                hidden_states = view(hidden_states,
                                     [-1, hidden_states.shape[-1]])
                hidden_states = index_select(hidden_states, 0, cp_join_index)
            else:
                assert False, "Context parallelism with non-remove-padding is not supported yet."

        if self.config.mapping.is_last_pp_rank():
            all_hidden_states = hidden_states
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

            lm_logits = self.lm_head(hidden_states)
            if hasattr(self.config, 'output_multiplier_scale'):
                lm_logits *= getattr(self.config, 'output_multiplier_scale', 1)
            if self.mup_width_multiplier is not None:
                lm_logits = lm_logits / self.mup_width_multiplier
            if is_gemma_2_cg or is_gemma_3_cg:
                softcap = self.config.get_config_group(
                    Gemma2ConfigGroup if not is_gemma_3_cg else
                    Gemma3ConfigGroup).final_logit_softcapping
                if softcap:
                    lm_logits = lm_logits * float(1 / softcap)
                    lm_logits = tanh(lm_logits) * float(softcap)
            lm_logits.mark_output('logits', self.config.logits_dtype)
            hidden_states.mark_output('last_hidden_state', self.config.dtype)
        else:
            hidden_states.mark_output('hidden_states_output', self.config.dtype)

        if use_cache and not default_net().plugin_config.paged_kv_cache:
            for i, present in zip(
                    self.config.mapping.pp_layers(
                        self.config.num_hidden_layers), presents):
                present.mark_output(f'present_key_value_{i}',
                                    self.config.kv_dtype)
            if self.config.mapping.is_last_pp_rank():
                return (lm_logits, presents, hidden_states)
            return (hidden_states, presents)
        else:
            if self.config.mapping.is_last_pp_rank():
                return lm_logits, hidden_states, all_hidden_states
            return hidden_states

    @classmethod
    def from_hugging_face(
            cls,
            cfg_path,
            model_dir,
            hf_model_or_dir,
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        """Load IndexTTS2 GPT weights and create a TRT-LLM model."""
        kwargs.pop('load_model_on_cpu', None)
        kwargs.pop('is_prequantized_to_fp8', None)

        hf_weights, hf_config = load_hf_weights_and_config(cfg_path, model_dir)

        config = GPTConfig.from_hugging_face(hf_config,
                                             dtype=dtype,
                                             mapping=mapping,
                                             quant_config=quant_config,
                                             **kwargs)
        config.output_vocab_size = hf_config.output_vocab_size

        weights = load_weights_from_hf_model(hf_weights, hf_config, config)
        model = cls(config)
        model.load(weights)
        return model
