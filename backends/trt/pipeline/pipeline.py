"""Faster IndexTTS-2 inference pipeline."""

import os
import time
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import SeamlessM4TFeatureExtractor

from backends.trt.pipeline.config import TTSConfig, load_config
from backends.trt.pipeline.text_processing import TextNormalizer, TextTokenizer
from backends.trt.pipeline.audio_processing import (
    mel_spectrogram,
    load_and_resample,
    resample,
    compute_kaldi_fbank,
    compute_semantic_features,
)
from backends.trt.pipeline.cfm_solver import cfm_inference
from backends.trt.pipeline.streaming import StreamingDecoder, MEL_CODE_TO_FRAME_RATIO, HOP_SIZE, SAMPLE_RATE

from backends.trt.runtime.gpt_trtllm_runtime import GPTTRTEngine
from backends.trt.runtime.generic_trt_runtime import GenericTRTEngine


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SpeakerCondition:
    """Cached per-speaker conditioning tensors."""
    spk_cond_emb: torch.Tensor       # (1, T, 1024)
    prompt_condition: torch.Tensor    # (1, T_mel, 512)
    ref_mel: torch.Tensor             # (1, 80, T_mel)
    style: torch.Tensor               # (1, 192)


@dataclass
class AudioChunk:
    sample_rate: int
    audio: np.ndarray   # int16
    is_last: bool


@dataclass
class BatchAudioChunk:
    """One yield from batched streaming: per-sample audio + completion flags."""
    sample_rate: int
    audio_list: List[Optional[np.ndarray]]  # audio_list[b] = int16 array or None
    done_list: List[bool]                    # done_list[b] = True when sample b finished


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sequence_mask(lengths, max_length=None):
    if max_length is None:
        max_length = lengths.max()
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


def _find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()
    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    return torch.argmax(similarities)


def _insert_interval_silence(wavs, sampling_rate=22050, interval_ms=200):
    if not wavs or interval_ms <= 0:
        return wavs
    channel_size = wavs[0].size(0)
    sil_dur = int(sampling_rate * interval_ms / 1000.0)
    sil_tensor = torch.zeros(channel_size, sil_dur)
    result = []
    for i, wav in enumerate(wavs):
        result.append(wav)
        if i < len(wavs) - 1:
            result.append(sil_tensor)
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class FasterIndexTTS2:
    """Standalone FasterIndexTTS2 pipeline backed by TRT / TRT-LLM engines.

    Parameters
    ----------
    config_path : str
        Path to ``checkpoints/config.yaml``.
    model_dir : str
        Directory containing ``bpe.model``, ``wav2vec2bert_stats.pt``,
        ``feat1.pt``, ``feat2.pt``.
    gpt_engine_dir : str
        Directory with TRT-LLM GPT engine (``config.json`` + ``rank*.engine``).
    speed_emb_path : str
        Path to pre-exported ``speed_emb.pt`` (2 x 1280 tensor).
    speech_semantic_encoder_engine, semantic_codec_engine, speaker_perceiver_conditioner_engine,
    emotion_perceiver_conditioner_engine, latent_projector_engine, length_regulator_engine,
    campplus_engine, dit_engine, bigvgan_engine : str
        Paths to the nine plain-TRT ``.engine`` files.
    streaming_chunk_size, streaming_overlap_size : int
        Code-level chunking for streaming mode (0 = disable).
    device : str
        CUDA device string.
    """

    def __init__(
        self,
        config_path: str,
        model_dir: str,
        gpt_engine_dir: str,
        speed_emb_path: str,
        speech_semantic_encoder_engine: str,
        semantic_codec_engine: str,
        speaker_perceiver_conditioner_engine: str,
        emotion_perceiver_conditioner_engine: str,
        latent_projector_engine: str,
        length_regulator_engine: str,
        campplus_engine: str,
        dit_engine: str,
        bigvgan_engine: str,
        streaming_chunk_size: int = 0,
        streaming_overlap_size: int = 5,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        self.cfg: TTSConfig = load_config(config_path)

        # -- TRT engines --
        print(">> Loading TRT engines ...")
        self._speech_semantic_encoder = GenericTRTEngine(speech_semantic_encoder_engine, device=device)
        self._semantic_codec = GenericTRTEngine(semantic_codec_engine, device=device)
        self._speaker_perceiver_conditioner = GenericTRTEngine(speaker_perceiver_conditioner_engine, device=device)
        self._emotion_perceiver_conditioner = GenericTRTEngine(emotion_perceiver_conditioner_engine, device=device)
        self._gpt = GPTTRTEngine(gpt_engine_dir, device=device)
        self._latent_projector = GenericTRTEngine(latent_projector_engine, device=device)
        self._length_regulator = GenericTRTEngine(length_regulator_engine, device=device)
        self._dit = GenericTRTEngine(dit_engine, device=device)
        self._dit_fn = lambda x, prompt_x, x_lens, t, style, cond: self._dit(
            x=x, prompt_x=prompt_x, x_lens=x_lens, t=t, style=style, cond=cond
        )
        self._bigvgan = GenericTRTEngine(bigvgan_engine, device=device)
        self._campplus = GenericTRTEngine(campplus_engine, device=device)
        print(">> All 10 TRT engines loaded")

        # -- Speed embedding (trivial 2x1280 weight from pre-exported file) --
        self._speed_emb_weight = torch.load(speed_emb_path, map_location=self.device, weights_only=True)

        # -- Text processing --
        bpe_path = os.path.join(model_dir, self.cfg.bpe_model)
        normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(bpe_path, normalizer=normalizer)

        # -- Semantic normalisation stats --
        stat = torch.load(os.path.join(model_dir, self.cfg.w2v_stat), map_location=self.device)
        self._semantic_mean = stat["mean"].to(self.device)
        self._semantic_std = torch.sqrt(stat["var"]).to(self.device)

        # -- Feature extractor --
        self._feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        # -- Emotion / speaker matrices --
        emo_matrix = torch.load(os.path.join(model_dir, self.cfg.emo_matrix), map_location=self.device)
        spk_matrix = torch.load(os.path.join(model_dir, self.cfg.spk_matrix), map_location=self.device)
        self._emo_matrix = torch.split(emo_matrix, self.cfg.emo_num)
        self._spk_matrix = torch.split(spk_matrix, self.cfg.emo_num)

        # -- Mel function shortcut --
        self._mel_fn_args = dict(
            n_fft=self.cfg.n_fft, num_mels=self.cfg.n_mels,
            sampling_rate=self.cfg.sample_rate, hop_size=self.cfg.hop_length,
            win_size=self.cfg.win_length, fmin=self.cfg.fmin, fmax=self.cfg.fmax,
            center=False,
        )

        # -- Streaming decoder --
        self._streamer: Optional[StreamingDecoder] = None
        if streaming_chunk_size > 0:
            self._streamer = StreamingDecoder(
                gpt_engine=self._gpt,
                codes_to_audio_fn=self._codes_to_audio_batch,
                chunk_size=streaming_chunk_size,
                overlap_size=streaming_overlap_size,
            )

        print(">> FasterIndexTTS2 ready")

    # ------------------------------------------------------------------
    # Speaker conditioning
    # ------------------------------------------------------------------

    def preload_speaker(self, audio_path: str, max_seconds: float = 15.0) -> SpeakerCondition:
        """Pre-compute all per-speaker cached tensors."""
        audio, sr = load_and_resample(audio_path, max_seconds=max_seconds)
        audio_22k = resample(audio, sr, 22050)
        audio_16k = resample(audio, sr, 16000)

        input_features, attention_mask = compute_semantic_features(
            audio_16k, self._feature_extractor, device=self.device,
        )
        feat = self._speech_semantic_encoder(
            input_features=input_features,
            attention_mask=attention_mask.float(),
        )
        spk_cond_emb = (feat - self._semantic_mean) / self._semantic_std

        result = self._semantic_codec(x=spk_cond_emb)
        S_ref = result["quantized"] if isinstance(result, dict) else result

        ref_mel = mel_spectrogram(audio_22k.to(self.device).float(), **self._mel_fn_args)
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(self.device)

        fbank = compute_kaldi_fbank(audio_16k, device=self.device)
        style = self._campplus(x=fbank)

        prompt_condition = self._length_regulator(
            x=S_ref, T_out=torch.tensor(ref_target_lengths.item(), dtype=torch.int64, device=self.device)
        )

        return SpeakerCondition(
            spk_cond_emb=spk_cond_emb,
            prompt_condition=prompt_condition,
            ref_mel=ref_mel,
            style=style,
        )

    def preload_speakers_batch(self, audio_paths: List[str], max_seconds: float = 15.0) -> List[SpeakerCondition]:
        """Batch speaker preprocessing for multiple speakers.

        Batches: speech_semantic_encoder, semantic_codec, length_regulator.
        Per-sample: campplus (unmasked global pooling), mel_spectrogram (fast PyTorch).
        """
        B = len(audio_paths)
        if B == 1:
            return [self.preload_speaker(audio_paths[0], max_seconds)]

        # Load and resample all audios
        audios_16k = []
        audios_22k = []
        for path in audio_paths:
            audio, sr = load_and_resample(path, max_seconds=max_seconds)
            audios_22k.append(resample(audio, sr, 22050))
            audios_16k.append(resample(audio, sr, 16000))

        # Batch SeamlessM4T features + speech_semantic_encoder TRT
        feat_list = []
        mask_list = []
        for a16k in audios_16k:
            inp_feat, att_mask = compute_semantic_features(a16k, self._feature_extractor, device=self.device)
            feat_list.append(inp_feat.squeeze(0))
            mask_list.append(att_mask.squeeze(0))
        max_feat_len = max(f.size(0) for f in feat_list)
        batched_feat = torch.zeros(B, max_feat_len, feat_list[0].size(-1), device=self.device)
        batched_mask = torch.zeros(B, max_feat_len, device=self.device)
        for i, (f, m) in enumerate(zip(feat_list, mask_list)):
            batched_feat[i, :f.size(0)] = f
            batched_mask[i, :m.size(0)] = m

        feat = self._speech_semantic_encoder(input_features=batched_feat, attention_mask=batched_mask.float())
        spk_cond_embs = (feat - self._semantic_mean) / self._semantic_std

        # Track per-sample valid lengths for spk_cond_emb
        emb_lens = torch.tensor([f.size(0) for f in feat_list], device=self.device, dtype=torch.long)

        # Mask before semantic_codec
        emb_mask = _sequence_mask(emb_lens, max_length=spk_cond_embs.size(1)).unsqueeze(-1).to(spk_cond_embs.dtype)
        spk_cond_embs_masked = spk_cond_embs * emb_mask

        # Batch semantic_codec TRT
        result = self._semantic_codec(x=spk_cond_embs_masked)
        S_refs = result["quantized"] if isinstance(result, dict) else result

        # Per-sample: mel_spectrogram + campplus
        ref_mels = []
        styles = []
        ref_target_lengths_list = []
        for i in range(B):
            ref_mel = mel_spectrogram(audios_22k[i].to(self.device).float(), **self._mel_fn_args)
            ref_mels.append(ref_mel)
            ref_target_lengths_list.append(ref_mel.size(2))

            fbank = compute_kaldi_fbank(audios_16k[i], device=self.device)
            style = self._campplus(x=fbank)
            styles.append(style)

        # Batch length_regulator TRT
        ref_target_lengths = torch.tensor(ref_target_lengths_list, device=self.device, dtype=torch.long)
        max_ref_target = ref_target_lengths.max().item()

        # Mask S_refs before length_regulator
        S_refs_masked = S_refs * emb_mask
        prompt_conditions = self._length_regulator(
            x=S_refs_masked, T_out=torch.tensor(max_ref_target, dtype=torch.int64, device=self.device)
        )
        pc_mask = _sequence_mask(ref_target_lengths, max_length=prompt_conditions.size(1)).unsqueeze(-1).to(prompt_conditions.dtype)
        prompt_conditions = prompt_conditions * pc_mask

        # Build per-sample SpeakerCondition (unpadded)
        results = []
        for i in range(B):
            el = emb_lens[i].item()
            rtl = ref_target_lengths[i].item()
            results.append(SpeakerCondition(
                spk_cond_emb=spk_cond_embs[i:i+1, :el, :],
                prompt_condition=prompt_conditions[i:i+1, :rtl, :],
                ref_mel=ref_mels[i],
                style=styles[i],
            ))
        return results

    # ------------------------------------------------------------------
    # Conditioning latent construction
    # ------------------------------------------------------------------

    def _build_conds_latent(
        self,
        spk_cond_emb: torch.Tensor,
        emo_cond_emb: torch.Tensor,
        emo_alpha: float,
        emo_vector: Optional[List[float]],
        style: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Build (B, 34, 1280) conditioning latent."""
        spk_lengths = torch.tensor([spk_cond_emb.shape[1]], device=self.device, dtype=torch.int64)
        emo_lengths = torch.tensor([emo_cond_emb.shape[1]], device=self.device, dtype=torch.int64)

        speech_conditioning_latent = self._speaker_perceiver_conditioner(
            speech_input=spk_cond_emb, lengths=spk_lengths,
        )

        base_vec = self._emotion_perceiver_conditioner(speech_input=spk_cond_emb, lengths=spk_lengths)
        emo_vec = self._emotion_perceiver_conditioner(speech_input=emo_cond_emb, lengths=emo_lengths)
        emovec = base_vec + emo_alpha * (emo_vec - base_vec)

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            random_index = [
                _find_most_similar_cosine(style, tmp) for tmp in self._spk_matrix
            ]
            emo_mat = [tmp[idx].unsqueeze(0) for idx, tmp in zip(random_index, self._emo_matrix)]
            emo_mat = torch.cat(emo_mat, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_mat
            emovec_mat = torch.sum(emovec_mat, 0).unsqueeze(0)
            emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

        duration_emb = self._speed_emb_weight[0].unsqueeze(0).expand(batch_size, -1)
        duration_emb_half = self._speed_emb_weight[1].unsqueeze(0).expand(batch_size, -1)

        if speech_conditioning_latent.size(0) < batch_size:
            speech_conditioning_latent = speech_conditioning_latent.expand(batch_size, -1, -1).contiguous()
        if emovec.dim() == 2 and emovec.size(0) < batch_size:
            emovec = emovec.expand(batch_size, -1).contiguous()

        conds_latent = torch.cat([
            speech_conditioning_latent + emovec.unsqueeze(1),
            duration_emb_half.unsqueeze(1),
            duration_emb.unsqueeze(1),
        ], dim=1)

        return conds_latent

    def _build_conds_latent_batch(
        self,
        speakers: List[SpeakerCondition],
        emo_speakers: List[Optional[SpeakerCondition]],
        emo_alphas: List[float],
        emo_vectors: List[Optional[List[float]]],
    ) -> torch.Tensor:
        """Build (B, 34, 1280) conditioning latent for per-sample speakers.

        Batches speaker_perceiver_conditioner and emotion_perceiver_conditioner TRT engines by padding
        variable-length spk_cond_emb to (B, max_T, 1024) with lengths.
        """
        B = len(speakers)
        if B == 1:
            emo_emb = emo_speakers[0].spk_cond_emb if emo_speakers[0] is not None else speakers[0].spk_cond_emb
            return self._build_conds_latent(
                speakers[0].spk_cond_emb, emo_emb,
                emo_alphas[0], emo_vectors[0], speakers[0].style, 1,
            )

        # Pad spk_cond_emb to (B, max_T, 1024)
        spk_lens = torch.tensor([s.spk_cond_emb.size(1) for s in speakers], device=self.device, dtype=torch.int64)
        max_spk_len = spk_lens.max().item()
        padded_spk = torch.zeros(B, max_spk_len, 1024, device=self.device, dtype=speakers[0].spk_cond_emb.dtype)
        for i, s in enumerate(speakers):
            padded_spk[i, :s.spk_cond_emb.size(1)] = s.spk_cond_emb.squeeze(0)

        # Pad emo_cond_emb
        emo_embs = []
        for i in range(B):
            if emo_speakers[i] is not None:
                emo_embs.append(emo_speakers[i].spk_cond_emb)
            else:
                emo_embs.append(speakers[i].spk_cond_emb)
        emo_lens = torch.tensor([e.size(1) for e in emo_embs], device=self.device, dtype=torch.int64)
        max_emo_len = emo_lens.max().item()
        padded_emo = torch.zeros(B, max_emo_len, 1024, device=self.device, dtype=padded_spk.dtype)
        for i, e in enumerate(emo_embs):
            padded_emo[i, :e.size(1)] = e.squeeze(0)

        # Batched speaker_perceiver_conditioner TRT
        speech_conditioning_latent = self._speaker_perceiver_conditioner(
            speech_input=padded_spk, lengths=spk_lens,
        )

        # Batched emotion_perceiver_conditioner TRT
        base_vec = self._emotion_perceiver_conditioner(speech_input=padded_spk, lengths=spk_lens)
        emo_vec = self._emotion_perceiver_conditioner(speech_input=padded_emo, lengths=emo_lens)

        # Per-sample emo_alpha blending
        alphas = torch.tensor(emo_alphas, device=self.device, dtype=base_vec.dtype).unsqueeze(1)
        emovec = base_vec + alphas * (emo_vec - base_vec)

        # Per-sample emo_vector mixing
        for i in range(B):
            if emo_vectors[i] is not None:
                weight_vector = torch.tensor(emo_vectors[i], device=self.device)
                random_index = [
                    _find_most_similar_cosine(speakers[i].style, tmp) for tmp in self._spk_matrix
                ]
                emo_mat = [tmp[idx].unsqueeze(0) for idx, tmp in zip(random_index, self._emo_matrix)]
                emo_mat = torch.cat(emo_mat, 0)
                emovec_mat = weight_vector.unsqueeze(1) * emo_mat
                emovec_mat = torch.sum(emovec_mat, 0).unsqueeze(0)
                emovec[i] = emovec_mat.squeeze(0) + (1 - torch.sum(weight_vector)) * emovec[i]

        # Speed embeddings
        duration_emb = self._speed_emb_weight[0].unsqueeze(0).expand(B, -1)
        duration_emb_half = self._speed_emb_weight[1].unsqueeze(0).expand(B, -1)

        conds_latent = torch.cat([
            speech_conditioning_latent + emovec.unsqueeze(1),
            duration_emb_half.unsqueeze(1),
            duration_emb.unsqueeze(1),
        ], dim=1)

        return conds_latent

    # ------------------------------------------------------------------
    # s2mel pipeline: codes -> audio
    # ------------------------------------------------------------------

    def _codes_to_audio_batch(
        self,
        codes: torch.Tensor,
        latent: torch.Tensor,
        code_lens: torch.Tensor,
        prompt_condition: torch.Tensor,
        ref_mel: torch.Tensor,
        style: torch.Tensor,
        prompt_lens: torch.Tensor = None,
    ) -> List[np.ndarray]:
        """latent projector -> length_regulator -> CFM(DiT) -> BigVGAN.

        All engines run in batched mode.  Per-sample variable lengths are
        handled via masking (length_regulator, CFM) and per-sample audio
        cropping (BigVGAN).

        Args:
            prompt_lens: (B,) per-sample prompt lengths. If None, uses
                prompt_condition.size(1) for all samples (same-speaker batch).

        Returns per-sample float32 numpy arrays.
        """
        batch_size = codes.shape[0]
        device = codes.device

        with torch.no_grad():
            S_infer = self._latent_projector(latent=latent, codes=codes)

            if batch_size > 1:
                code_mask = _sequence_mask(code_lens, max_length=S_infer.size(1))
                S_infer = S_infer * code_mask.unsqueeze(-1).to(S_infer.dtype)

            target_lengths = (code_lens * MEL_CODE_TO_FRAME_RATIO).long()

            cond = self._length_regulator(
                x=S_infer, T_out=torch.tensor(target_lengths.max().item(), dtype=torch.int64, device=self.device)
            )
            mask = _sequence_mask(target_lengths, max_length=cond.size(1)).unsqueeze(-1).to(cond.dtype)
            cond = cond * mask

            # Expand single-speaker tensors to batch if needed
            if prompt_condition.size(0) < batch_size:
                prompt_condition = prompt_condition.expand(batch_size, -1, -1)
            if ref_mel.size(0) < batch_size:
                ref_mel = ref_mel.expand(batch_size, -1, -1)
            if style.size(0) < batch_size:
                style = style.expand(batch_size, -1)

            if prompt_lens is None:
                prompt_lens = torch.full(
                    (batch_size,), prompt_condition.size(1),
                    dtype=torch.long, device=device,
                )

            # Build cat_condition with per-sample contiguous placement:
            # [prompt(pl), cond(tl)] with no zero gap for shorter prompts
            x_lens = (prompt_lens + target_lengths).long()
            max_total = x_lens.max().item()
            cond_dim = cond.size(2)
            cat_condition = torch.zeros(batch_size, max_total, cond_dim, device=device, dtype=cond.dtype)
            for b in range(batch_size):
                pl = prompt_lens[b].item()
                tl = target_lengths[b].item()
                cat_condition[b, :pl] = prompt_condition[b, :pl]
                cat_condition[b, pl:pl + tl] = cond[b, :tl]

            vc_target = cfm_inference(
                self._dit_fn, cat_condition, x_lens, ref_mel, style,
                n_timesteps=25, inference_cfg_rate=0.7,
                zero_prompt_speech_token=self.cfg.zero_prompt_speech_token,
                prompt_lens=prompt_lens,
            )

            # Per-sample crop: remove each sample's prompt region
            max_gen_len = target_lengths.max().item()
            cropped = torch.zeros(batch_size, vc_target.size(1), max_gen_len, device=device, dtype=vc_target.dtype)
            for b in range(batch_size):
                pl = prompt_lens[b].item()
                tl = target_lengths[b].item()
                cropped[b, :, :tl] = vc_target[b, :, pl:pl + tl]
            vc_target = cropped

            wav_batch = self._bigvgan(mel=vc_target.float())

        # Per-sample audio crop
        audio_lengths = target_lengths * HOP_SIZE
        results = []
        if wav_batch.dim() == 1:
            results.append(wav_batch[:audio_lengths[0].item()].cpu().float().numpy())
        elif wav_batch.dim() == 2:
            for b in range(batch_size):
                results.append(wav_batch[b, :audio_lengths[b].item()].cpu().float().numpy())
        else:
            for b in range(batch_size):
                results.append(wav_batch[b, 0, :audio_lengths[b].item()].cpu().float().numpy())
        return results

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------

    def _prepare_trt_text_ids(
        self,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Offset BPE IDs into the GPT shared vocab space and wrap with
        start/stop tokens.

        Returns (trt_text_ids, trt_text_lengths).
        """
        nmc = self.cfg.number_mel_codes
        start_text = self.cfg.start_text_token + nmc
        stop_text = self.cfg.stop_text_token + nmc

        trt_text_ids = text_tokens.clone() + nmc
        trt_text_ids = F.pad(trt_text_ids, (1, 0), value=start_text)
        trt_text_ids = F.pad(trt_text_ids, (0, 1), value=stop_text)
        start_mel = torch.tensor(
            [[self.cfg.start_mel_token]], device=trt_text_ids.device, dtype=trt_text_ids.dtype,
        )
        trt_text_ids = torch.cat([trt_text_ids, start_mel.expand(trt_text_ids.shape[0], -1)], dim=1)
        trt_text_lengths = torch.tensor(
            [trt_text_ids.shape[1]] * trt_text_ids.shape[0],
            device=trt_text_ids.device, dtype=torch.int32,
        )
        return trt_text_ids, trt_text_lengths

    # ------------------------------------------------------------------
    # Batched text preparation
    # ------------------------------------------------------------------

    def _prepare_batched_trt_text(
        self,
        text_tokens_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrap each sample with start/stop/start_mel tokens individually,
        then pad the batch.

        This ensures every sample has the correct token layout
        ``[start_text, text+nmc..., stop_text, start_mel]``
        regardless of length differences across the batch.

        Returns (trt_text_ids, trt_text_lengths) with correct per-sample lengths.
        """
        per_sample_ids = []
        for t in text_tokens_list:
            ids, _ = self._prepare_trt_text_ids(t)   # (1, L_i + 3)
            per_sample_ids.append(ids.squeeze(0))     # (L_i + 3,)

        lengths = [ids.shape[0] for ids in per_sample_ids]
        max_len = max(lengths)

        # Pad to max length; pad value doesn't matter because text_lengths
        # tells the GPT exactly how many tokens are valid per sample
        batched = torch.full(
            (len(per_sample_ids), max_len), 0, dtype=torch.int32, device=self.device,
        )
        for i, ids in enumerate(per_sample_ids):
            batched[i, :ids.shape[0]] = ids

        trt_text_lengths = torch.tensor(lengths, device=self.device, dtype=torch.int32)
        return batched, trt_text_lengths

    # ------------------------------------------------------------------
    # Core per-segment generation (batched)
    # ------------------------------------------------------------------

    def _generate_segment_batch(
        self,
        text_tokens_list: List[torch.Tensor],
        conds_latent: torch.Tensor,
        prompt_condition: torch.Tensor,
        ref_mel: torch.Tensor,
        style: torch.Tensor,
        prompt_lens: torch.Tensor = None,
        max_mel_tokens: int = 1500,
        num_beams: int = 3,
        top_k: int = 30,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 10.0,
    ) -> List[torch.Tensor]:
        """Generate audio for a batch of text segments.

        Supports per-sample conditioning (different speakers) when
        prompt_lens is provided.

        Returns list of (1, T_audio) wav tensors (int16-scale).
        """
        batch_size = len(text_tokens_list)
        trt_text_ids, trt_text_lengths = self._prepare_batched_trt_text(text_tokens_list)

        if conds_latent.size(0) < batch_size:
            conds_latent = conds_latent.expand(batch_size, -1, -1).contiguous()
        if prompt_condition.size(0) < batch_size:
            prompt_condition = prompt_condition.expand(batch_size, -1, -1).contiguous()
        if ref_mel.size(0) < batch_size:
            ref_mel = ref_mel.expand(batch_size, -1, -1).contiguous()
        if style.size(0) < batch_size:
            style = style.expand(batch_size, -1).contiguous()

        codes, latent, code_lens = self._gpt(
            conds_latents=conds_latent,
            text_input_ids=trt_text_ids,
            stop_token_id=self.cfg.stop_mel_token,
            max_new_tokens=max_mel_tokens,
            num_beams=num_beams,
            top_k=top_k, top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            text_lengths=trt_text_lengths,
        )

        audios_np = self._codes_to_audio_batch(
            codes, latent, code_lens, prompt_condition, ref_mel, style,
            prompt_lens=prompt_lens,
        )

        wav_tensors = []
        for audio_np in audios_np:
            wav = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wav_tensors.append(wav)
        return wav_tensors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _prepare_batch_conditioning(
        self,
        speakers: List[SpeakerCondition],
        emo_speakers: List[Optional[SpeakerCondition]],
        emo_alphas: List[float],
        emo_vectors: List[Optional[List[float]]],
    ):
        """Build all batched conditioning tensors for per-sample speakers.

        Returns (conds_latent, prompt_condition, ref_mel, style, prompt_lens).
        """
        B = len(speakers)

        # Batched conds_latent
        conds_latent = self._build_conds_latent_batch(
            speakers, emo_speakers, emo_alphas, emo_vectors,
        )

        # Pad prompt_condition to (B, max_T_prompt, 512)
        prompt_lens = torch.tensor(
            [s.prompt_condition.size(1) for s in speakers],
            device=self.device, dtype=torch.long,
        )
        max_prompt_len = prompt_lens.max().item()
        prompt_condition = torch.zeros(
            B, max_prompt_len, speakers[0].prompt_condition.size(2),
            device=self.device, dtype=speakers[0].prompt_condition.dtype,
        )
        for i, s in enumerate(speakers):
            prompt_condition[i, :s.prompt_condition.size(1)] = s.prompt_condition.squeeze(0)

        # Pad ref_mel to (B, 80, max_T_prompt)
        ref_mel = torch.zeros(
            B, speakers[0].ref_mel.size(1), max_prompt_len,
            device=self.device, dtype=speakers[0].ref_mel.dtype,
        )
        for i, s in enumerate(speakers):
            ref_mel[i, :, :s.ref_mel.size(2)] = s.ref_mel.squeeze(0)

        # Stack style to (B, 192)
        style = torch.cat([s.style for s in speakers], dim=0)

        return conds_latent, prompt_condition, ref_mel, style, prompt_lens

    def generate(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, SpeakerCondition, List[SpeakerCondition]],
        emo_speaker: Union[str, SpeakerCondition, List[Optional[SpeakerCondition]], None] = None,
        emo_alpha: Union[float, List[float]] = 1.0,
        emo_vector: Union[Optional[List[float]], List[Optional[List[float]]]] = None,
        max_text_tokens_per_segment: int = 120,
        interval_silence_ms: int = 200,
        stream: bool = False,
        output_path: Optional[str] = None,
        max_mel_tokens: int = 1500,
        num_beams: int = 3,
        top_k: int = 30,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 10.0,
    ) -> Union[Tuple[int, np.ndarray], Generator]:
        """Generate speech from text.

        Parameters
        ----------
        text : str or list[str]
            Single string or list of strings for batched generation.
        speaker : str, SpeakerCondition, or list[SpeakerCondition]
            Reference speaker(s). A single value is broadcast to all items.
        emo_speaker : str, SpeakerCondition, list, or None
            Emotion reference(s). If None, uses speaker.
        emo_alpha : float or list[float]
            Emotion blending weight(s).
        emo_vector : list[float], list[list[float]], or None
            8-D emotion vector(s).
        stream : bool
            If True, returns a generator yielding AudioChunk / BatchAudioChunk.
        """
        start_time = time.perf_counter()

        texts = [text] if isinstance(text, str) else text
        batch_size = len(texts)

        # Normalize speaker to list
        if isinstance(speaker, str):
            speaker = self.preload_speaker(speaker)
        if isinstance(speaker, SpeakerCondition):
            speakers = [speaker] * batch_size
        else:
            speakers = speaker

        # Normalize emo params to lists
        if emo_speaker is None:
            emo_speakers = [None] * batch_size
        elif isinstance(emo_speaker, (str, SpeakerCondition)):
            if isinstance(emo_speaker, str):
                emo_speaker = self.preload_speaker(emo_speaker)
            emo_speakers = [emo_speaker] * batch_size
        else:
            emo_speakers = emo_speaker

        if isinstance(emo_alpha, (int, float)):
            emo_alphas = [float(emo_alpha)] * batch_size
        else:
            emo_alphas = [float(a) for a in emo_alpha]

        if emo_vector is None or (isinstance(emo_vector, list) and len(emo_vector) > 0 and isinstance(emo_vector[0], (int, float))):
            emo_vectors = [emo_vector] * batch_size
        else:
            emo_vectors = emo_vector

        # Default emo_alpha=1.0 when no separate emo reference
        for i in range(batch_size):
            if emo_speakers[i] is None:
                emo_speakers[i] = speakers[i]
                if emo_vectors[i] is None:
                    emo_alphas[i] = 1.0

        # Use fast single-conditioning path only when ALL conditioning is identical
        all_same_conditioning = (
            all(s is speakers[0] for s in speakers)
            and all(e is emo_speakers[0] for e in emo_speakers)
            and all(a == emo_alphas[0] for a in emo_alphas)
            and all(v == emo_vectors[0] for v in emo_vectors)
        )

        if all_same_conditioning:
            spk = speakers[0]
            conds_latent = self._build_conds_latent(
                spk.spk_cond_emb, emo_speakers[0].spk_cond_emb,
                emo_alphas[0], emo_vectors[0], spk.style, batch_size=1,
            )
            prompt_condition = spk.prompt_condition
            ref_mel = spk.ref_mel
            style = spk.style
            prompt_lens = None
        else:
            conds_latent, prompt_condition, ref_mel, style, prompt_lens = \
                self._prepare_batch_conditioning(speakers, emo_speakers, emo_alphas, emo_vectors)

        all_segments = []
        for t in texts:
            tokens_str = self.tokenizer.tokenize(t)
            segments = self.tokenizer.split_segments(
                tokens_str, max_text_tokens_per_segment=max_text_tokens_per_segment,
            )
            all_segments.append(segments)

        # Streaming path
        if stream and self._streamer is not None:
            return self._generate_stream(
                all_segments, conds_latent,
                prompt_condition, ref_mel, style, prompt_lens,
                max_mel_tokens=max_mel_tokens, num_beams=num_beams,
                top_k=top_k, top_p=top_p, temperature=temperature,
                repetition_penalty=repetition_penalty,
                interval_silence_ms=interval_silence_ms,
                output_path=output_path,
            )

        # Non-streaming path
        all_wavs: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        max_num_segments = max(len(segs) for segs in all_segments)
        for seg_idx in range(max_num_segments):
            seg_tokens_list = []
            active_indices = []
            for b in range(batch_size):
                if seg_idx < len(all_segments[b]):
                    seg = all_segments[b][seg_idx]
                    ids = self.tokenizer.convert_tokens_to_ids(seg)
                    t = torch.tensor([ids], dtype=torch.int32, device=self.device)
                    seg_tokens_list.append(t)
                    active_indices.append(b)

            if not seg_tokens_list:
                continue

            wavs = self._generate_segment_batch(
                seg_tokens_list, conds_latent,
                prompt_condition, ref_mel, style,
                prompt_lens=prompt_lens,
                max_mel_tokens=max_mel_tokens, num_beams=num_beams,
                top_k=top_k, top_p=top_p, temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            for j, b in enumerate(active_indices):
                all_wavs[b].append(wavs[j])

        results = []
        for b in range(batch_size):
            wavs_with_silence = _insert_interval_silence(
                all_wavs[b], sampling_rate=SAMPLE_RATE, interval_ms=interval_silence_ms,
            )
            full_wav = torch.cat(wavs_with_silence, dim=1)
            results.append(full_wav)

        elapsed = time.perf_counter() - start_time

        if batch_size == 1:
            wav = results[0]
            wav_duration = wav.shape[-1] / SAMPLE_RATE
            print(f">> Total inference: {elapsed:.2f}s, audio: {wav_duration:.2f}s, RTF: {elapsed / wav_duration:.4f}")
            audio_int16 = wav.cpu().to(torch.int16).numpy().T
            if output_path:
                self._save_wav(wav, output_path)
            return (SAMPLE_RATE, audio_int16)

        out = []
        for b in range(batch_size):
            wav = results[b]
            audio_int16 = wav.cpu().to(torch.int16).numpy().T
            out.append((SAMPLE_RATE, audio_int16))
        return out

    # ------------------------------------------------------------------
    # Streaming generation (single and batched)
    # ------------------------------------------------------------------

    def _generate_stream(
        self,
        all_segments,
        conds_latent,
        prompt_condition,
        ref_mel,
        style,
        prompt_lens=None,
        max_mel_tokens: int = 1500,
        num_beams: int = 3,
        top_k: int = 30,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 10.0,
        interval_silence_ms: int = 200,
        output_path: Optional[str] = None,
        audio_ws=None,
    ) -> Generator:
        """Streaming generator supporting both single and batched text.

        Accepts per-sample conditioning (prompt_condition, ref_mel, style
        already padded/stacked, with optional prompt_lens for per-sample
        prompt lengths).
        """
        batch_size = len(all_segments)
        max_num_segments = max(len(segs) for segs in all_segments)
        is_single = (batch_size == 1)

        per_sample_chunks: List[List[np.ndarray]] = [[] for _ in range(batch_size)]

        for seg_idx in range(max_num_segments):
            seg_tokens_list = []
            active_indices = []
            for b in range(batch_size):
                if seg_idx < len(all_segments[b]):
                    seg = all_segments[b][seg_idx]
                    ids = self.tokenizer.convert_tokens_to_ids(seg)
                    t = torch.tensor([ids], dtype=torch.int32, device=self.device)
                    seg_tokens_list.append(t)
                    active_indices.append(b)

            if not seg_tokens_list:
                continue

            active_batch = len(seg_tokens_list)
            trt_text_ids, trt_text_lengths = self._prepare_batched_trt_text(seg_tokens_list)

            seg_conds = conds_latent.expand(active_batch, -1, -1).contiguous() if conds_latent.size(0) < active_batch else conds_latent
            seg_prompt = prompt_condition.expand(active_batch, -1, -1).contiguous() if prompt_condition.size(0) < active_batch else prompt_condition
            seg_ref_mel = ref_mel.expand(active_batch, -1, -1).contiguous() if ref_mel.size(0) < active_batch else ref_mel
            seg_style = style.expand(active_batch, -1).contiguous() if style.size(0) < active_batch else style

            for _sr, audio_list, done_list in self._streamer.generate(
                conds_latent=seg_conds,
                trt_text_ids=trt_text_ids,
                stop_token_id=self.cfg.stop_mel_token,
                prompt_condition=seg_prompt,
                ref_mel=seg_ref_mel,
                style=seg_style,
                max_new_tokens=max_mel_tokens,
                num_beams=num_beams,
                top_k=top_k, top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                audio_ws=audio_ws,
                text_lengths=trt_text_lengths,
                prompt_lens=prompt_lens,
            ):
                for j, b in enumerate(active_indices):
                    if audio_list[j] is not None:
                        per_sample_chunks[b].append(audio_list[j])

                if is_single:
                    audio = audio_list[0]
                    if audio is not None:
                        is_last = done_list[0] and seg_idx == max_num_segments - 1
                        yield AudioChunk(
                            sample_rate=SAMPLE_RATE, audio=audio, is_last=is_last,
                        )
                else:
                    full_audio_list: List[Optional[np.ndarray]] = [None] * batch_size
                    full_done_list = [False] * batch_size
                    for j, b in enumerate(active_indices):
                        full_audio_list[b] = audio_list[j]
                        full_done_list[b] = (
                            done_list[j] and seg_idx >= len(all_segments[b]) - 1
                        )
                    yield BatchAudioChunk(
                        sample_rate=SAMPLE_RATE,
                        audio_list=full_audio_list,
                        done_list=full_done_list,
                    )

        if output_path and is_single and per_sample_chunks[0]:
            full_audio = np.concatenate(per_sample_chunks[0])
            wav = torch.tensor(full_audio, dtype=torch.float32).unsqueeze(0)
            self._save_wav(wav, output_path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _save_wav(wav: torch.Tensor, path: str):
        if os.path.isfile(path):
            os.remove(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torchaudio.save(path, wav.cpu().to(torch.int16), SAMPLE_RATE)
        print(f">> WAV saved to {path}")
