"""CPU call-chain regressions using real tensors and injected model components.

No checkpoints are loaded. The public wrappers, generators, allocation, streaming
and waveform assembly run unchanged; only model outputs and text splitting are
replaced by small deterministic doubles.
"""
import importlib
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

SAMPLE_RATE = 22050
HOP = 256


@pytest.fixture(params=["indextts.infer_v2", "indextts.infer_v2_5"])
def engine(request):
    module = importlib.import_module(request.param)
    tts = module.IndexTTS2.__new__(module.IndexTTS2)
    tts.device = "cpu"
    tts.dtype = None
    tts.low_vram = False
    tts.cfg = SimpleNamespace(s2mel=SimpleNamespace(
        preprocess_params=SimpleNamespace(spect_params=SimpleNamespace(hop_length=HOP))))
    tts._set_gr_progress = Mock()
    tts.cache_spk_audio_prompt = tts.cache_emo_audio_prompt = "reference.wav"
    tts.cache_spk_cond = tts.cache_emo_cond = torch.zeros(1, 4, 3)
    tts.cache_s2mel_style = torch.zeros(1, 4)
    tts.cache_s2mel_prompt = torch.zeros(1, 10, 4)
    tts.cache_mel = torch.zeros(1, 4, 10)
    tts.stop_mel_token = 999
    tts.gpt = Mock()
    tts.gpt.merge_emovec.return_value = torch.zeros(1, 4)
    tts.gpt.inference_speech.side_effect = lambda *args, **kwargs: (
        torch.tensor([[2, 3, 999]]), torch.zeros(1, 2, 4))
    tts.gpt.return_value = torch.zeros(1, 2, 4)
    tts.semantic_codec = SimpleNamespace(
        quantizer=SimpleNamespace(vq2emb=lambda codes: torch.zeros(1, 4, 2)),
        decode=lambda codes: torch.zeros(1, 2, 4))
    tts.target_lengths = []

    def regulate(semantic, ylens, **kwargs):
        frames = ylens.item()
        tts.target_lengths.append(frames)
        return (torch.zeros(1, frames, 4),)

    estimator = SimpleNamespace(
        input_pos=torch.arange(16384),
        transformer=SimpleNamespace(freqs_cis=torch.empty(16384, 1, 2)),
        style_as_token=False, time_as_token=False)
    cfm = SimpleNamespace(estimator=estimator)

    def diffuse(condition, lengths, reference, *args, **kwargs):
        assert lengths.item() == condition.shape[1]
        assert condition.shape[1] <= min(
            estimator.input_pos.shape[0], estimator.transformer.freqs_cis.shape[0])
        return torch.zeros(1, 4, condition.shape[1])

    cfm.inference = Mock(side_effect=diffuse)
    tts.s2mel = SimpleNamespace(models={
        "cfm": cfm, "length_regulator": Mock(side_effect=regulate),
        "gpt_layer": lambda latent: latent})
    tts.bigvgan = Mock(side_effect=lambda mel: torch.full(
        (1, 1, mel.shape[-1] * HOP), 0.25))
    tts.segments = ["ab", "abcd"]
    tts.tokenizer = SimpleNamespace(
        tokenize=lambda text: list(text),
        split_segments=lambda *args, **kwargs: [list(s) for s in tts.segments],
        convert_tokens_to_ids=lambda tokens: [2] * len(tokens),
        unk_token_id=-1, encode=lambda text, **kwargs: [2] * len(text))
    tts.text_process = SimpleNamespace(clean_pattern=re.compile(r"(?!)"), char_rep_map={})
    tts.split_text_by_tokens = lambda *args: tts.segments
    tts.split_text_by_punctuation = lambda *args, **kwargs: tts.segments
    tts.module = module
    return tts


def call(tts, method="infer", **kwargs):
    if tts.module.__name__.endswith("v2_5"):
        kwargs.update(lang="en", text_normalization=False)
    return getattr(tts, method)("reference.wav", "some text", None, **kwargs)


@pytest.mark.parametrize("method,stream", [("infer", False), ("infer", True), ("infer_generator", True)])
def test_rejects_200_seconds_before_generation(engine, method, stream):
    engine.segments = ["a"]
    with pytest.raises(ValueError, match=r"DiT position capacity.*17227.*16374"):
        result = call(engine, method, target_duration=200, stream_return=stream)
        if stream:
            next(result)
    engine.gpt.inference_speech.assert_not_called()
    engine.s2mel.models["length_regulator"].assert_not_called()
    engine.s2mel.models["cfm"].inference.assert_not_called()
    engine.bigvgan.assert_not_called()


def test_rejects_later_segment_before_streaming_any_audio(engine):
    engine.segments = ["a", "b" * 100]
    with pytest.raises(ValueError, match="segment 2"):
        next(call(engine, target_duration=200, stream_return=True))
    engine.gpt.inference_speech.assert_not_called()


def test_reference_length_changes_accepted_boundary(engine):
    engine.segments = ["a"]
    # A tiny injected position table keeps this actual call-chain test cheap.
    estimator = engine.s2mel.models["cfm"].estimator
    estimator.input_pos = torch.arange(16)
    estimator.transformer.freqs_cis = torch.empty(16, 1, 2)
    duration = 6 * HOP / SAMPLE_RATE
    rate, data = call(engine, target_duration=duration)
    assert rate == SAMPLE_RATE
    assert data.shape == (6 * HOP, 1)
    assert engine.target_lengths == [6]

    engine.cache_s2mel_prompt = torch.zeros(1, 11, 4)
    engine.cache_mel = torch.zeros(1, 4, 11)
    engine.gpt.inference_speech.reset_mock()
    with pytest.raises(ValueError, match=r"only 5 are available"):
        call(engine, target_duration=duration)
    engine.gpt.inference_speech.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_public_output_has_exact_duration_and_pauses(engine, stream):
    duration = 0.123  # A non-frame-aligned sample budget exercises final fitting.
    result = call(engine, target_duration=duration, duration_factor=999,
                  interval_silence=10, stream_return=stream)
    if stream:
        chunks = list(result)
        assert len(chunks) == 3  # speech, pause, speech; no trailing pause
        assert chunks[1].shape[-1] == int(SAMPLE_RATE * 0.01)
        assert not torch.count_nonzero(chunks[1])
        assert sum(chunk.shape[-1] for chunk in chunks) == round(duration * SAMPLE_RATE)
    else:
        rate, data = result
        assert rate == SAMPLE_RATE
        assert data.shape == (round(duration * SAMPLE_RATE), 1)
        assert data[-1, 0] == 0  # Upstream fade follows final sample fitting.
    assert engine.target_lengths == [4, 6]
    assert engine.gpt.inference_speech.call_count == 2


@pytest.mark.parametrize("stream", [False, True])
def test_automatic_duration_still_uses_duration_factor(engine, stream):
    result = call(engine, duration_factor=2, stream_return=stream)
    if stream:
        list(result)
    assert engine.target_lengths == [6, 6]


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_reaches_public_validation(engine, duration):
    with pytest.raises(ValueError, match="positive number"):
        call(engine, target_duration=duration)
    engine.gpt.inference_speech.assert_not_called()


def test_low_vram_rejects_later_chunk_before_first_generation(engine):
    if engine.module.__name__.endswith("infer_v2"):
        pytest.skip("Low-VRAM wrapper is specific to v2.5")
    engine.low_vram = True
    engine.segments = ["a", "b" * 39]
    with pytest.raises(ValueError, match="segment 2"):
        engine.infer("reference.wav", "x" * 41, None, "en", target_duration=200,
                     text_normalization=False)
    engine.gpt.inference_speech.assert_not_called()
    engine.s2mel.models["cfm"].inference.assert_not_called()


def test_low_vram_output_and_forwarding(engine):
    if engine.module.__name__.endswith("infer_v2"):
        pytest.skip("Low-VRAM wrapper is specific to v2.5")
    engine.low_vram = True
    engine.segments = ["a", "bbb"]
    # Each outer chunk gets tokenized independently by the real generator.
    engine.split_text_by_tokens = lambda text, *args: [text]
    result = engine.infer("reference.wav", "x" * 41, None, "en",
                          target_duration=0.123, duration_factor=999,
                          interval_silence=10, text_normalization=False)
    rate, data = result
    assert rate == SAMPLE_RATE
    assert data.shape == (round(0.123 * SAMPLE_RATE), 1)
    assert engine.target_lengths == [3, 7]
    assert engine.gpt.inference_speech.call_count == 2


def test_low_vram_prepares_and_reuses_uncached_reference(engine, monkeypatch):
    if engine.module.__name__.endswith("infer_v2"):
        pytest.skip("Low-VRAM wrapper is specific to v2.5")
    engine.low_vram = True
    engine.cache_spk_cond = None
    engine.segments = ["a", "bbb"]
    engine.split_text_by_tokens = lambda text, *args: [text]
    engine._load_and_cut_audio = Mock(return_value=(torch.zeros(1, 100), SAMPLE_RATE))
    engine.extract_features = Mock(return_value={
        "input_features": torch.zeros(1, 4, 3), "attention_mask": torch.ones(1, 3)})
    engine.get_emb = Mock(return_value=torch.zeros(1, 4, 3))
    engine.mel_fn = Mock(return_value=torch.zeros(1, 4, 10))
    engine.campplus_model = Mock(return_value=torch.zeros(1, 4))
    monkeypatch.setattr(engine.module.torchaudio.transforms, "Resample", lambda *args: lambda wav: wav)
    monkeypatch.setattr(engine.module.torchaudio.compliance.kaldi, "fbank", lambda *args, **kwargs: torch.zeros(3, 4))
    rate, data = engine.infer("reference.wav", "x" * 41, None, "en",
                            target_duration=0.123, interval_silence=10,
                            text_normalization=False)
    assert rate == SAMPLE_RATE
    assert data.shape == (round(0.123 * SAMPLE_RATE), 1)
    engine.mel_fn.assert_called_once()
    engine.get_emb.assert_called_once()
    engine._load_and_cut_audio.assert_called_once()
    engine.campplus_model.assert_called_once()
    assert engine.target_lengths == [10, 3, 7]  # reference prepared once, then speech


@pytest.mark.parametrize("style_token,time_token", [(False, False), (True, False), (False, True), (True, True)])
def test_position_budget_reserves_optional_conditioning_tokens(engine, style_token, time_token):
    engine.segments = ["a"]
    estimator = engine.s2mel.models["cfm"].estimator
    estimator.input_pos = torch.arange(16)
    estimator.transformer.freqs_cis = torch.empty(16, 1, 2)
    estimator.style_as_token = style_token
    estimator.time_as_token = time_token
    available = 6 - int(style_token) - int(time_token)
    call(engine, target_duration=available * HOP / SAMPLE_RATE)
    engine.gpt.inference_speech.reset_mock()
    with pytest.raises(ValueError, match="DiT position capacity"):
        call(engine, target_duration=(available + 1) * HOP / SAMPLE_RATE)
    engine.gpt.inference_speech.assert_not_called()


@pytest.mark.parametrize("short_buffer", ["input_pos", "freqs_cis"])
def test_capacity_uses_the_smaller_actual_position_buffer(engine, short_buffer):
    engine.segments = ["a"]
    estimator = engine.s2mel.models["cfm"].estimator
    if short_buffer == "input_pos":
        estimator.input_pos = torch.arange(15)
    else:
        estimator.transformer.freqs_cis = torch.empty(15, 1, 2)
    with pytest.raises(ValueError, match="only 5 are available"):
        call(engine, target_duration=6 * HOP / SAMPLE_RATE)
    engine.gpt.inference_speech.assert_not_called()


def test_actual_rotary_function_at_and_above_default_position_capacity():
    from indextts.s2mel.modules.gpt_fast.model import apply_rotary_emb, precompute_freqs_cis

    freqs = precompute_freqs_cis(16384, 2, 10000)
    output = apply_rotary_emb(torch.zeros(1, 16384, 1, 2), freqs)
    assert output.shape == (1, 16384, 1, 2)
    for length in (16385, 17227):
        with pytest.raises(RuntimeError, match="shape"):
            apply_rotary_emb(torch.zeros(1, length, 1, 2), freqs)


@pytest.mark.parametrize("stream", [False, True])
def test_public_rejects_duration_shorter_than_pauses(engine, stream):
    with pytest.raises(ValueError, match="too short"):
        result = call(engine, target_duration=0.1, interval_silence=200,
                      stream_return=stream)
        if stream:
            next(result)
    engine.gpt.inference_speech.assert_not_called()


def test_public_preserves_explicit_sampling_choice(engine):
    call(engine, target_duration=0.123, interval_silence=10, do_sample=False)
    assert all(not args.kwargs["do_sample"] for args in engine.gpt.inference_speech.call_args_list)
