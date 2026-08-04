"""TensorRT-LLM runtime wrapper for the IndexTTS2 GPT model."""

import json
import os
from pathlib import Path

import torch
import tensorrt_llm
import tensorrt_llm.runtime

from backends.trt.trtllm.indextts2_gpt_generation import IndexTTS2GPTForCausalLMGenerationSession


def _build_model_config(config):
    """Build a TRT-LLM ModelConfig from the engine config.json."""
    pretrained_config = config['pretrained_config']
    build_config = config['build_config']
    plugin_config = build_config['plugin_config']
    mapping_cfg = pretrained_config['mapping']
    quantization = pretrained_config.get('quantization', {})

    dtype = pretrained_config['dtype']
    tp_size = mapping_cfg['tp_size']
    pp_size = mapping_cfg.get('pp_size', 1)
    world_size = tp_size * pp_size

    num_heads = pretrained_config['num_attention_heads'] // tp_size
    num_kv_heads = pretrained_config['num_key_value_heads'] // tp_size
    hidden_size = pretrained_config['hidden_size'] // tp_size
    vocab_size = pretrained_config['vocab_size']
    output_vocab_size = pretrained_config['output_vocab_size']
    num_layers = pretrained_config['num_hidden_layers']
    head_size = pretrained_config.get('head_size', 64)

    use_gpt_attention_plugin = plugin_config.get('gpt_attention_plugin', 'disable') != 'disable'
    remove_input_padding = plugin_config.get('remove_input_padding', False)
    gemm_allreduce_plugin = plugin_config.get('gemm_allreduce_plugin', None)
    lora_plugin = plugin_config.get('lora_plugin', None)
    lora_plugin = lora_plugin is not None and lora_plugin != 'disable' if lora_plugin else False
    mamba_conv1d_plugin = plugin_config.get('mamba_conv1d_plugin', 'disable') != 'disable'
    paged_state = plugin_config.get('paged_state', False)

    max_batch_size = build_config.get('max_batch_size', 8)
    max_beam_width = build_config.get('max_beam_width', 3)
    max_input_len = build_config.get('max_input_len', 1024)
    max_prompt_embedding_table_size = build_config.get('max_prompt_embedding_table_size', 0)
    tokens_per_block = plugin_config.get('tokens_per_block', 32)
    gather_context_logits = build_config.get('gather_context_logits', False)
    gather_generation_logits = build_config.get('gather_generation_logits', False)

    from tensorrt_llm.bindings import KVCacheType
    kv_cache_type_str = build_config.get('kv_cache_type', 'PAGED')
    if kv_cache_type_str == 'PAGED':
        kv_cache_type = KVCacheType.PAGED
    elif kv_cache_type_str == 'CONTINUOUS':
        kv_cache_type = KVCacheType.CONTINUOUS
    else:
        kv_cache_type = KVCacheType.DISABLED

    from tensorrt_llm.quantization import QuantMode
    quant_algo = quantization.get('quant_algo', None)
    kv_cache_quant_algo = quantization.get('kv_cache_quant_algo', None)
    quant_mode = QuantMode(0)
    if quant_algo is not None:
        if 'W4A16' in str(quant_algo):
            quant_mode = QuantMode.use_weight_only(use_int4_weights=True)
        elif 'W8A16' in str(quant_algo):
            quant_mode = QuantMode.use_weight_only(use_int4_weights=False)
    if kv_cache_quant_algo is not None:
        quant_mode = quant_mode.set_int8_kv_cache()

    lora_config = build_config.get('lora_config', {})
    lora_target_modules = lora_config.get('lora_target_modules', [])
    trtllm_modules_to_hf_modules = lora_config.get('trtllm_modules_to_hf_modules', None)
    model_name = pretrained_config.get('architecture', '')

    from tensorrt_llm.layers import LanguageAdapterConfig

    model_config = tensorrt_llm.runtime.ModelConfig(
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        gpt_attention_plugin=use_gpt_attention_plugin,
        gemm_allreduce_plugin=gemm_allreduce_plugin,
        remove_input_padding=remove_input_padding,
        model_name=model_name,
        dtype=dtype,
        head_size=head_size,
        has_position_embedding=True,
        has_token_type_embedding=False,
        kv_cache_type=kv_cache_type,
        tokens_per_block=tokens_per_block,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        quant_mode=quant_mode,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        lora_plugin=lora_plugin,
        lora_target_modules=lora_target_modules,
        trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
        cross_attention=False,
        skip_cross_kv=False,
        num_medusa_heads=0,
        max_medusa_tokens=0,
        paged_state=paged_state,
        mamba_conv1d_plugin=mamba_conv1d_plugin,
        conv_kernel=0,
        layer_types=[],
        rnn_hidden_size=0,
        rnn_head_size=0,
        rnn_conv_dim_size=0,
        state_size=0,
        state_dtype="",
        gpu_weights_percent=1.0,
        redrafter_num_beams=0,
        redrafter_draft_len_per_beam=0,
        num_kv_heads_per_layer=None,
        num_kv_heads_per_cross_attn_layer=None,
        skip_cross_attn_blocks=False,
        language_adapter_config=None,
    )
    model_config.output_vocab_size = output_vocab_size

    return model_config, dtype, world_size, max_input_len


def _ptuning_setup(prompt_table, dtype, hidden_size, input_ids,
                   input_lengths, remove_input_padding):
    """Set up prompt-tuning (ptuning) kwargs for the decode call."""
    if prompt_table is not None:
        task_vocab_size = torch.tensor(
            [prompt_table.shape[1]], dtype=torch.int32, device="cuda")
        prompt_table = prompt_table.view(
            prompt_table.shape[0] * prompt_table.shape[1],
            prompt_table.shape[2])
        prompt_table = prompt_table.cuda().to(
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
    else:
        prompt_table = torch.empty([1, hidden_size]).cuda()
        task_vocab_size = torch.zeros([1]).cuda()

    num_sequences = (input_lengths.size(0) if remove_input_padding
                     else input_ids.size(0))
    tasks = torch.zeros([num_sequences], dtype=torch.int32).cuda()

    return {
        "prompt_embedding_table": prompt_table,
        "tasks": tasks,
        "prompt_vocab_size": task_vocab_size,
    }


class GPTTRTEngine:
    """TRT-LLM wrapper for IndexTTS2 GPT generation.

    Encapsulates engine loading, session creation, and the full decode pipeline.
    Call signature matches the interface expected by the FasterIndexTTS2 pipeline.
    """

    def __init__(self, engine_dir, device="cuda:0"):
        self.device = torch.device(device)

        engine_dir = Path(engine_dir)
        with open(engine_dir / 'config.json', 'r') as f:
            config = json.load(f)

        self.model_config, self.dtype, world_size, self.max_input_len = \
            _build_model_config(config)
        self.vocab_size = self.model_config.vocab_size

        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)

        engine_path = engine_dir / f'rank{runtime_rank}.engine'
        with open(engine_path, 'rb') as f:
            engine_buffer = f.read()

        self.session = IndexTTS2GPTForCausalLMGenerationSession(
            self.model_config, engine_buffer, runtime_mapping)

        # Track current beam width. dynamic_decoder locks to the beam_width
        # from its setup() call and rejects changes. We recreate it when needed.
        self._current_beam_width = None

        print(f">> GPT TRT-LLM engine loaded from: {engine_dir}")

    def _ensure_beam_width(self, num_beams):
        """Recreate the dynamic_decoder if beam width changed."""
        if self._current_beam_width != num_beams:
            import torch
            session = self.session
            session.dynamic_decoder = torch.classes.trtllm.DynamicDecodeOp(
                self.model_config.max_batch_size,
                self.model_config.max_beam_width,
                session.vocab_size,
                session.vocab_size_padded,
                session.mapping.tp_size,
                session.mapping.pp_size,
                session.decoder_logits_dtype,
            )
            self._current_beam_width = num_beams

    def _prepare_inputs(self, conds_latents, text_input_ids, text_lengths=None):
        """Prepare inputs for TRT-LLM decode, supporting variable-length batches.

        Args:
            conds_latents: (B, cond_len, H) conditioning embeddings
            text_input_ids: (B, max_text_len) padded text token IDs
            text_lengths: (B,) actual text length per sample, or None (= all same)

        Returns:
            input_ids_or_list: list[Tensor] if remove_input_padding, else (B, L) Tensor
            per_sample_lengths: (B,) per-sample total input lengths
            max_input_length: int, max across batch
            ptuning_args: dict
        """
        batch_size = conds_latents.shape[0]
        cond_len = conds_latents.shape[1]

        if text_lengths is None:
            text_lengths = torch.tensor(
                [text_input_ids.shape[1]] * batch_size,
                device=self.device, dtype=torch.int32)
        else:
            text_lengths = text_lengths.to(device=self.device, dtype=torch.int32)

        per_sample_lengths = cond_len + text_lengths

        fake_prompt_id = torch.arange(
            self.vocab_size,
            self.vocab_size + batch_size * cond_len,
            device=self.device,
        ).reshape(batch_size, cond_len)

        input_ids = torch.cat([fake_prompt_id, text_input_ids], dim=1).contiguous()
        input_ids = input_ids.to(torch.int32).to(self.device)

        max_input_length = per_sample_lengths.max().item()

        if self.session.remove_input_padding:
            input_ids_or_list = [
                input_ids[b, :per_sample_lengths[b]] for b in range(batch_size)
            ]
        else:
            input_ids_or_list = input_ids

        ptuning_args = _ptuning_setup(
            prompt_table=conds_latents,
            dtype=self.dtype,
            hidden_size=self.model_config.hidden_size,
            input_ids=input_ids,
            input_lengths=per_sample_lengths.unsqueeze(1),
            remove_input_padding=self.model_config.remove_input_padding,
        )

        return input_ids_or_list, per_sample_lengths, max_input_length, ptuning_args

    def __call__(self, conds_latents, text_input_ids,
                 stop_token_id, max_new_tokens=1500,
                 num_beams=3, top_k=30, top_p=0.8,
                 temperature=0.8, repetition_penalty=10.0,
                 streaming=False, text_lengths=None):
        """Run GPT generation via TRT-LLM.

        Args:
            conds_latents: (B, cond_len, hidden_size) conditioning embeddings
            text_input_ids: (B, L) text token IDs (padded to max length)
            stop_token_id: int, the stop mel token ID
            max_new_tokens: maximum number of mel tokens to generate
            num_beams: beam width for beam search
            top_k, top_p, temperature, repetition_penalty: sampling parameters
            streaming: if True, use streaming decode
            text_lengths: (B,) actual text length per sample, or None (= all same)

        Returns:
            codes: (B, T_mel) generated mel code IDs
            latent: (B, T_mel, hidden_size) per-token hidden states
            code_lens: (B,) valid code lengths per sample
        """
        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=stop_token_id,
            pad_id=stop_token_id,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
        )

        batch_size = conds_latents.shape[0]
        input_ids_or_list, per_sample_lengths, max_input_length, ptuning_args = \
            self._prepare_inputs(conds_latents, text_input_ids, text_lengths)

        self._ensure_beam_width(num_beams)

        with torch.no_grad():
            self.session.setup(
                batch_size=batch_size,
                max_context_length=max_input_length,
                max_new_tokens=max_new_tokens,
                beam_width=num_beams,
            )

            if self.session.remove_input_padding:
                result = self.session.decode_batch(
                    input_ids=input_ids_or_list,
                    sampling_config=sampling_config,
                    streaming=streaming,
                    return_dict=True,
                    **ptuning_args,
                )
            else:
                result = self.session.decode(
                    input_ids=input_ids_or_list,
                    context_lengths=per_sample_lengths,
                    streaming=streaming,
                    return_dict=True,
                    sampling_config=sampling_config,
                    **ptuning_args,
                )

            if streaming:
                all_hidden_states = []
                for step_dict in result:
                    if step_dict is None:
                        continue
                    step_output_ids = step_dict["output_ids"]
                    hs = step_dict.get("last_hidden_state")
                    if hs is not None:
                        all_hidden_states.append(hs.detach().clone())
                torch.cuda.synchronize()
                output_dict = {
                    "output_ids": step_output_ids,
                    "last_hidden_state": all_hidden_states,
                }
            else:
                torch.cuda.synchronize()
                output_dict = result

        output_ids = output_dict["output_ids"]
        output_ids = output_ids[:, 0, :]  # beam 0

        # Extract codes per-sample: TRT-LLM pads shorter samples to
        # max_context_length in output_ids, but the dynamic decoder writes
        # generated tokens starting at each sample's own context length.
        # We must use per_sample_lengths[b] as the offset for each sample.
        per_sample_code_lists = []
        per_sample_lens = []
        for b in range(batch_size):
            start = per_sample_lengths[b].item()
            sample_codes = output_ids[b, start:]
            stop_positions = (sample_codes == stop_token_id).nonzero(as_tuple=False)
            code_len = stop_positions[0].item() if len(stop_positions) > 0 else sample_codes.shape[0]
            per_sample_code_lists.append(sample_codes[:code_len])
            per_sample_lens.append(code_len)

        max_code_len = max(per_sample_lens) if per_sample_lens else 0
        codes = torch.full((batch_size, max_code_len), stop_token_id,
                           dtype=output_ids.dtype, device=output_ids.device)
        for b in range(batch_size):
            codes[b, :per_sample_lens[b]] = per_sample_code_lists[b]
        code_lens = torch.tensor(per_sample_lens, dtype=torch.long, device=codes.device)

        # Extract last_hidden_state: list of (B*beam, hidden) per step -> stack.
        # hidden_states_list is indexed by generation step (not output_ids
        # position), so step k corresponds to code k for ALL samples
        # regardless of their context length.
        hidden_states_list = output_dict["last_hidden_state"]
        num_beams = sampling_config.num_beams
        latent = torch.stack([
            hs.view(batch_size, num_beams, -1)[:, 0, :]
            for hs in hidden_states_list
        ])
        latent = latent.permute(1, 0, 2)[:, :max_code_len, :].float()

        return codes, latent, code_lens

    def generate_chunks(self, conds_latents, text_input_ids,
                             stop_token_id, chunk_size, overlap_size,
                             max_new_tokens=1500,
                             num_beams=3, top_k=30, top_p=0.8,
                             temperature=0.8, repetition_penalty=10.0,
                             text_lengths=None):
        """Real streaming: yields chunks from inside the TRT-LLM iteration loop
        as soon as enough tokens accumulate.

        Yields:
            chunk_codes: (B, chunk_len) mel codes for this chunk
            chunk_latent: (B, chunk_len, H) hidden states for this chunk
            is_last: bool, True if this is the final chunk
            batch_done: list[bool], per-sample completion flags
            chunk_code_lens: (B,) tensor of valid code lengths per sample
        """
        stride = chunk_size - overlap_size
        if stride <= 0:
            raise ValueError(
                f"overlap_size ({overlap_size}) must be less than chunk_size ({chunk_size}); "
                f"got stride={stride} which would cause an infinite loop."
            )

        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=stop_token_id,
            pad_id=stop_token_id,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
        )

        batch_size = conds_latents.shape[0]
        input_ids_or_list, per_sample_lengths, max_input_length, ptuning_args = \
            self._prepare_inputs(conds_latents, text_input_ids, text_lengths)

        self._ensure_beam_width(num_beams)

        with torch.no_grad():
            self.session.setup(
                batch_size=batch_size,
                max_context_length=max_input_length,
                max_new_tokens=max_new_tokens,
                beam_width=num_beams,
            )

            if self.session.remove_input_padding:
                result = self.session.decode_batch(
                    input_ids=input_ids_or_list,
                    sampling_config=sampling_config,
                    streaming=True,
                    return_dict=True,
                    **ptuning_args,
                )
            else:
                result = self.session.decode(
                    input_ids=input_ids_or_list,
                    context_lengths=per_sample_lengths,
                    streaming=True,
                    return_dict=True,
                    sampling_config=sampling_config,
                    **ptuning_args,
                )

            all_hidden_states = []
            step_count = 0
            next_chunk_at = chunk_size

            # Track per-sample done status and valid lengths
            sample_done = [False] * batch_size
            sample_len = [0] * batch_size

            # Pre-compute per-sample start offsets for code extraction.
            # TRT-LLM pads shorter samples to max_context_length in output_ids,
            # but generated tokens start at each sample's own context length.
            sample_starts = [per_sample_lengths[b].item() for b in range(batch_size)]

            # Aligned codes buffer: step k -> column k for all samples
            max_gen = max_new_tokens
            codes = torch.full((batch_size, max_gen), stop_token_id,
                               dtype=torch.int32, device=conds_latents.device)

            for step_dict in result:
                if step_dict is None:
                    continue
                step_output_ids = step_dict["output_ids"]
                hs = step_dict.get("last_hidden_state")
                if hs is not None:
                    # Beam-aware: take beam 0 for each sample
                    all_hidden_states.append(
                        hs.view(batch_size, num_beams, -1)[:, 0, :].detach().clone())

                step_count += 1
                # Extract the latest generated token per-sample from its own offset
                for b in range(batch_size):
                    tok = step_output_ids[b, 0, sample_starts[b] + step_count - 1]
                    codes[b, step_count - 1] = tok

                # Check stop token per sample
                for b in range(batch_size):
                    if not sample_done[b]:
                        if codes[b, step_count - 1].item() == stop_token_id:
                            sample_done[b] = True
                            sample_len[b] = step_count - 1
                        else:
                            sample_len[b] = step_count

                current_len = max(sample_len)

                # Yield chunks when enough tokens accumulated
                while current_len >= next_chunk_at:
                    pos = next_chunk_at - chunk_size
                    c = codes[:, pos:next_chunk_at]
                    l = torch.stack(all_hidden_states[pos:next_chunk_at]).permute(1, 0, 2).float()
                    batch_done_flags = [sample_done[b] and sample_len[b] <= next_chunk_at
                                        for b in range(batch_size)]
                    chunk_code_lens = torch.tensor(
                        [max(0, min(sample_len[b] - pos, chunk_size)) for b in range(batch_size)],
                        dtype=torch.long, device=c.device)
                    yield (c, l, False, batch_done_flags, chunk_code_lens)
                    next_chunk_at += stride

                if all(sample_done):
                    break

            torch.cuda.synchronize()

            # Final partial chunk
            current_len = max(sample_len)
            pos = next_chunk_at - chunk_size
            if pos < current_len:
                c = codes[:, pos:current_len]
                l = torch.stack(all_hidden_states[pos:current_len]).permute(1, 0, 2).float()
                batch_done_flags = [True] * batch_size
                chunk_code_lens = torch.tensor(
                    [max(0, sample_len[b] - pos) for b in range(batch_size)],
                    dtype=torch.long, device=c.device)
                yield (c, l, True, batch_done_flags, chunk_code_lens)
