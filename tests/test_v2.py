"""
IndexTTS v2 tests.

Run with:
    uv run --extra test pytest tests/test_v2.py -v

CI only (no GPU):
    uv run --extra test pytest tests/test_v2.py -v -m "not gpu"
"""
import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

CHECKPOINTS_DIR = Path("checkpoints")
CONFIG_PATH = CHECKPOINTS_DIR / "config.yaml"

CHECKPOINTS_25_DIR = Path("checkpoints_25")
CONFIG_25_PATH = CHECKPOINTS_25_DIR / "config.yaml"


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture(scope="module")
def tts_model():
    pytest.importorskip("torch")
    if not CONFIG_PATH.exists():
        pytest.skip(f"Checkpoints not found at {CHECKPOINTS_DIR}")

    from indextts.infer_v2 import IndexTTS2
    return IndexTTS2(
        cfg_path=str(CONFIG_PATH),
        model_dir=str(CHECKPOINTS_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
    )


@pytest.fixture(scope="module")
def prompt_wav():
    from indextts.utils.examples_downloader import ensure_test_sample_available
    return ensure_test_sample_available()


@pytest.fixture(scope="module")
def tts_model_25():
    pytest.importorskip("torch")
    if not CONFIG_25_PATH.exists():
        pytest.skip(f"Checkpoints not found at {CHECKPOINTS_25_DIR}")

    from indextts.infer_v2_5 import IndexTTS2
    return IndexTTS2(
        cfg_path=str(CONFIG_25_PATH),
        model_dir=str(CHECKPOINTS_25_DIR),
        use_bf16=True,
    )


# -- Download URL checks (no GPU) ---------------------------------------------

# Each auxiliary model: (test_id, repo_id, probe_file)
# probe_file: a small file in the repo to verify download works end-to-end.
_MODEL_PROBES = [
    ("bigvgan", "nvidia/bigvgan_v2_22khz_80band_256x", "config.json"),
    ("w2v-bert-2.0", "facebook/w2v-bert-2.0", "config.json"),
    ("campplus", "funasr/campplus", "campplus_cn_common.bin"),
    ("MaskGCT", "amphion/MaskGCT", "README.md"),
]


@pytest.mark.parametrize("name,repo_id,filename", _MODEL_PROBES, ids=[m[0] for m in _MODEL_PROBES])
def test_model_download_reachable(name, repo_id, filename, tmp_path):
    """Each auxiliary model must be downloadable via the real download path."""
    from indextts.utils.examples_downloader import _download_file
    from indextts.utils.network_detection import need_proxy

    base_url = "https://hf-mirror.com" if need_proxy() else "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/main/{filename}"
    dest = tmp_path / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download_file(url, str(dest), max_bytes=8192)
    assert dest.exists() and dest.stat().st_size > 0


def test_example_download_reachable(tmp_path, monkeypatch):
    """Example audio must be downloadable via the real download path."""
    from indextts.utils import examples_downloader

    monkeypatch.setattr(examples_downloader, "_TESTS_DIR", str(tmp_path))
    path = examples_downloader.download_test_sample(force=True)
    assert Path(path).exists() and Path(path).stat().st_size > 0


# -- Model download logic (no GPU) --------------------------------------------

def test_legacy_cache_compatibility(tmp_path, monkeypatch):
    """ensure_models_available preserves existing cache and skips re-download."""
    from indextts.utils import model_download
    from indextts.utils.model_download import ensure_models_available

    download_calls = []
    original_download = model_download._download_single_file

    def mock_download(*args, **kwargs):
        download_calls.append(args)
        return original_download(*args, **kwargs)

    monkeypatch.setattr(model_download, "_download_single_file", mock_download)

    model_dir = tmp_path / "checkpoints"
    cache_dir = model_dir / "hf_cache"
    cache_dir.mkdir(parents=True)

    w2v_dir = cache_dir / "w2v-bert-2.0"
    w2v_dir.mkdir()
    (w2v_dir / "config.json").write_text('{"test": true}')

    bigvgan_dir = cache_dir / "bigvgan"
    bigvgan_dir.mkdir()
    (bigvgan_dir / "config.json").write_text('{"test": true}')
    (bigvgan_dir / "bigvgan_generator.pt").write_bytes(b"fake")

    campplus = cache_dir / "campplus_cn_common.bin"
    campplus.write_bytes(b"fake_campplus")

    semantic = cache_dir / "semantic_codec_model.safetensors"
    semantic.write_bytes(b"fake_semantic")

    paths = ensure_models_available(str(model_dir))

    # Files preserved
    assert (w2v_dir / "config.json").exists()
    assert campplus.exists()
    assert semantic.exists()
    assert (bigvgan_dir / "bigvgan_generator.pt").exists()

    # No download triggered
    assert len(download_calls) == 0, f"Unexpected downloads: {download_calls}"


def test_modelscope_single_file_download_matches_local_path(tmp_path, monkeypatch):
    """ModelScope single-file download must produce the exact requested local_path."""
    from indextts.utils import model_download

    local_path = tmp_path / "hf_cache" / "semantic_codec_model.safetensors"
    expected_bytes = b"fake_semantic"

    def fake_model_file_download(model_id, file_path, local_dir):
        downloaded = Path(local_dir) / file_path
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(expected_bytes)
        return str(downloaded)

    fake_modelscope = types.ModuleType("modelscope")
    fake_hub = types.ModuleType("modelscope.hub")
    fake_file_download = types.ModuleType("modelscope.hub.file_download")
    fake_file_download.model_file_download = fake_model_file_download
    fake_hub.file_download = fake_file_download
    fake_modelscope.hub = fake_hub

    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", fake_hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", fake_file_download)
    monkeypatch.setattr(model_download, "_get_using_modelscope", lambda: True)

    got = model_download._download_single_file(
        repo_id="amphion/MaskGCT",
        filename="semantic_codec/model.safetensors",
        local_path=str(local_path),
    )

    assert got == str(local_path)
    assert local_path.exists()
    assert local_path.read_bytes() == expected_bytes


# -- Text segmentation (no GPU) -----------------------------------------------

def _splitter_stub():
    """A 2.5 instance with only what split_text_by_tokens touches.

    Bypasses __init__ so the test needs no checkpoints; the tokenizer stub counts
    one token per character, which is enough to drive segmentation.
    """
    import types

    from indextts.infer_v2_5 import IndexTTS2

    obj = IndexTTS2.__new__(IndexTTS2)
    obj.tokenizer = types.SimpleNamespace(encode=lambda text, **kwargs: list(text))
    obj.gpt = types.SimpleNamespace(
        text_pos_embedding=types.SimpleNamespace(
            emb=types.SimpleNamespace(num_embeddings=602)
        )
    )
    return obj


def _annotated_long_text():
    """Long enough to segment, with annotations spread across the boundaries."""
    filler = "各种应用层出不穷，而语音合成作为人机交互的重要一环，其自然度和实时性同样关键。"
    unit = filler + "他要<going|G OW1 . IH0 NG>去，银<行|XING2>也开着。"
    return unit * 5


def test_split_keeps_pronunciation_annotations_paired():
    """Segment boundaries must not land inside a <|SPECIAL_TOKEN_n|> pair.

    apply_pronunciation_annotations runs before segmentation and emits payloads
    that can contain a period -- one of the splitter's break characters -- so an
    unguarded splitter leaves halves with an unpaired marker.
    """
    from indextts.infer_v2_5 import apply_pronunciation_annotations

    splitter = _splitter_stub()
    text = apply_pronunciation_annotations(_annotated_long_text().lower())
    expected_pairs = text.count("<|SPECIAL_TOKEN_1|>") // 2 + text.count("<|SPECIAL_TOKEN_2|>") // 2
    assert expected_pairs == 10, f"fixture should carry 10 annotations, got {expected_pairs}"

    segments = splitter.split_text_by_tokens(text, 120, "<|zh|> ")
    assert len(segments) > 1, "fixture must be long enough to segment"

    unpaired = [
        s for s in segments
        if s.count("<|SPECIAL_TOKEN_1|>") % 2 or s.count("<|SPECIAL_TOKEN_2|>") % 2
    ]
    assert not unpaired, f"annotation split across segments: {unpaired}"

    kept = sum(
        s.count("<|SPECIAL_TOKEN_1|>") // 2 + s.count("<|SPECIAL_TOKEN_2|>") // 2
        for s in segments
    )
    assert kept == expected_pairs, f"lost annotations: {expected_pairs} -> {kept}"


def test_split_stays_within_position_embedding_capacity():
    """Every segment must fit text_pos_embedding, whose overrun is a CUDA assert."""
    splitter = _splitter_stub()
    capacity = splitter.gpt.text_pos_embedding.emb.num_embeddings
    prefix = "<|zh|> "

    text = "各种应用层出不穷，而语音合成作为人机交互的重要一环。" * 40
    segments = splitter.split_text_by_tokens(text, 120, prefix)
    assert len(segments) > 1

    for s in segments:
        used = splitter._token_len(prefix + s) + 1  # +1 for the trailing pad token
        assert used <= capacity, f"segment needs {used} positions, capacity is {capacity}"


def test_split_leaves_short_text_alone():
    splitter = _splitter_stub()
    text = "今天天气不错。"
    assert splitter.split_text_by_tokens(text, 120, "<|zh|> ") == [text]


def _load_qwen_emotion_module(module_name, monkeypatch):
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    def register(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    torch = register("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.xpu = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.Tensor = object
    torch.no_grad = lambda: (lambda func: func)

    torch_nn = register("torch.nn")
    torch_nn_functional = register("torch.nn.functional")
    torch_nn_utils = register("torch.nn.utils")
    torch_nn_utils_rnn = register("torch.nn.utils.rnn", pad_sequence=lambda *args, **kwargs: None)
    torch.nn = torch_nn
    torch_nn.functional = torch_nn_functional
    torch_nn.utils = torch_nn_utils
    torch_nn_utils.rnn = torch_nn_utils_rnn

    register("torchaudio")
    register("librosa")
    register("safetensors")
    register("transformers", AutoTokenizer=_Dummy, SeamlessM4TFeatureExtractor=_Dummy, Wav2Vec2BertModel=_Dummy)
    register("modelscope", AutoModelForCausalLM=_Dummy)
    register("omegaconf", OmegaConf=types.SimpleNamespace(load=lambda *args, **kwargs: None))
    register("indextts.gpt.model_v2", UnifiedVoice=_Dummy)
    register("indextts.codec.maskgct_codec", build_semantic_codec=lambda *args, **kwargs: None)
    register("indextts.codec.models", EnhancedCodec=_Dummy)
    register("indextts.utils.checkpoint", load_checkpoint=lambda *args, **kwargs: None)
    register("indextts.utils.front", TextNormalizer=_Dummy, TextTokenizer=_Dummy)
    register("indextts.utils.tokenizer", get_tokenizer=lambda *args, **kwargs: None, lang_to_token={})
    register("indextts.utils.ja_g2p", JapaneseG2PProcessor=_Dummy)
    register("indextts.utils.nemo_tn", normalize_text=lambda text: text)
    register("indextts.s2mel.modules.commons", load_checkpoint2=lambda *args, **kwargs: None, MyModel=_Dummy)
    register("indextts.s2mel.modules.bigvgan", bigvgan=object())
    register("indextts.s2mel.modules.campplus.DTDNN", CAMPPlus=_Dummy)
    register("indextts.s2mel.modules.audio", mel_spectrogram=lambda *args, **kwargs: None)

    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", ["indextts.infer_v2", "indextts.infer_v2_5"])
def test_qwen_emotion_convert_accepts_label_values(module_name, monkeypatch):
    module = _load_qwen_emotion_module(module_name, monkeypatch)

    emo = module.QwenEmotion.__new__(module.QwenEmotion)
    emo.cn_key_to_en = {
        "高兴": "happy",
        "愤怒": "angry",
        "悲伤": "sad",
        "恐惧": "afraid",
        "反感": "disgusted",
        "低落": "melancholic",
        "惊讶": "surprised",
        "自然": "calm",
    }
    emo.desired_vector_order = list(emo.cn_key_to_en)
    emo.max_score = 1.2
    emo.min_score = 0.0

    assert emo.convert({"自然": "自然"}) == {
        "happy": 0.0,
        "angry": 0.0,
        "sad": 0.0,
        "afraid": 0.0,
        "disgusted": 0.0,
        "melancholic": 0.0,
        "surprised": 0.0,
        "calm": 1.0,
    }


@pytest.mark.parametrize("module_name", ["indextts.infer_v2", "indextts.infer_v2_5"])
def test_qwen_emotion_convert_accepts_label_only_payload(module_name, monkeypatch):
    module = _load_qwen_emotion_module(module_name, monkeypatch)

    emo = module.QwenEmotion.__new__(module.QwenEmotion)
    emo.cn_key_to_en = {
        "高兴": "happy",
        "愤怒": "angry",
        "悲伤": "sad",
        "恐惧": "afraid",
        "反感": "disgusted",
        "低落": "melancholic",
        "惊讶": "surprised",
        "自然": "calm",
    }
    emo.desired_vector_order = list(emo.cn_key_to_en)
    emo.max_score = 1.2
    emo.min_score = 0.0

    assert emo.convert({"emotion": "自然"})["calm"] == 1.0


@pytest.mark.parametrize("module_name", ["indextts.infer_v2", "indextts.infer_v2_5"])
def test_qwen_emotion_convert_redirects_cross_key_labels(module_name, monkeypatch):
    module = _load_qwen_emotion_module(module_name, monkeypatch)

    emo = module.QwenEmotion.__new__(module.QwenEmotion)
    emo.cn_key_to_en = {
        "高兴": "happy",
        "愤怒": "angry",
        "悲伤": "sad",
        "恐惧": "afraid",
        "反感": "disgusted",
        "低落": "melancholic",
        "惊讶": "surprised",
        "自然": "calm",
    }
    emo.desired_vector_order = list(emo.cn_key_to_en)
    emo.max_score = 1.2
    emo.min_score = 0.0

    emotion_dict = emo.convert({"高兴": "自然"})
    assert emotion_dict["happy"] == 0.0
    assert emotion_dict["calm"] == 1.0


# -- IndexTTS-2.5 default emotion conditioning reuse (no GPU) -----------------

def _new_v25_reuse_stub(monkeypatch, enabled=True):
    module = _load_qwen_emotion_module("indextts.infer_v2_5", monkeypatch)
    tts = module.IndexTTS2.__new__(module.IndexTTS2)
    tts.reuse_spk_cond_for_emo = enabled
    return module, tts


def test_25_reuse_spk_cond_constructor_option_defaults_off(monkeypatch):
    module, _ = _new_v25_reuse_stub(monkeypatch)
    parameter = inspect.signature(module.IndexTTS2.__init__).parameters["reuse_spk_cond_for_emo"]

    assert parameter.default is False


@pytest.mark.parametrize(
    "enabled,emo_audio_prompt,emo_vector,use_emo_text,expected",
    [
        (False, None, None, False, False),
        (True, None, None, False, True),
        (True, "speaker.wav", None, False, False),
        (True, None, [0.0] * 8, False, False),
        (True, None, None, True, False),
    ],
)
def test_25_reuse_spk_cond_only_applies_to_implicit_default_emotion(
        monkeypatch, enabled, emo_audio_prompt, emo_vector, use_emo_text, expected):
    _, tts = _new_v25_reuse_stub(monkeypatch, enabled=enabled)

    assert tts._should_reuse_spk_cond_for_emo(
        emo_audio_prompt,
        emo_vector,
        use_emo_text,
    ) is expected


def test_25_reuse_spk_cond_aliases_current_speaker_without_polluting_emo_cache(monkeypatch):
    _, tts = _new_v25_reuse_stub(monkeypatch)
    explicit_emo_cond = object()
    tts.cache_emo_cond = explicit_emo_cond
    tts.cache_emo_audio_prompt = "emotion.wav"
    tts._load_and_cut_audio = lambda *args, **kwargs: pytest.fail("emotion audio must not be loaded")

    first_spk_cond = object()
    second_spk_cond = object()
    assert tts._get_emo_cond_emb("speaker-a.wav", first_spk_cond, True, False) is first_spk_cond
    assert tts._get_emo_cond_emb("speaker-b.wav", second_spk_cond, True, False) is second_spk_cond
    assert tts.cache_emo_cond is explicit_emo_cond
    assert tts.cache_emo_audio_prompt == "emotion.wav"


def test_25_reuse_spk_cond_preserves_explicit_emo_audio_encoder_for_same_path(monkeypatch):
    _, tts = _new_v25_reuse_stub(monkeypatch)
    calls = []

    class FakeTensor:
        def to(self, device):
            calls.append(("to", device))
            return self

    input_features = FakeTensor()
    attention_mask = FakeTensor()
    encoded_emo = object()
    tts.device = "cpu"
    tts.cache_emo_cond = None
    tts.cache_emo_audio_prompt = None

    def load_audio(*args, **kwargs):
        calls.append(("load", args, kwargs))
        return object(), 16000

    tts._load_and_cut_audio = load_audio
    tts.extract_features = lambda *args, **kwargs: {
        "input_features": input_features,
        "attention_mask": attention_mask,
    }
    tts.get_emb = lambda features, mask: calls.append(("encode", features, mask)) or encoded_emo

    reuse_spk_cond = tts._should_reuse_spk_cond_for_emo("speaker.wav", None, False)
    result = tts._get_emo_cond_emb("speaker.wav", object(), reuse_spk_cond, False)

    assert reuse_spk_cond is False
    assert result is encoded_emo
    assert [call[0] for call in calls].count("load") == 1
    assert [call[0] for call in calls].count("encode") == 1
    assert tts.cache_emo_cond is encoded_emo
    assert tts.cache_emo_audio_prompt == "speaker.wav"

    assert tts._get_emo_cond_emb("speaker.wav", object(), False, False) is encoded_emo
    assert [call[0] for call in calls].count("load") == 1
    assert [call[0] for call in calls].count("encode") == 1


@pytest.mark.parametrize("enabled", [False, True], ids=["default-off", "reuse-enabled"])
def test_25_explicit_emotion_cache_survives_default_speaker_sequence(monkeypatch, enabled):
    """Exercise E -> default A -> default B -> E through reference preparation.

    Dependency doubles stop inference before synthesis: this validates cache
    routing and encoding counts, not checkpoint output or acoustic equivalence.
    """
    module, tts = _new_v25_reuse_stub(monkeypatch, enabled=enabled)
    load_calls = []
    encode_calls = []
    resample_calls = []
    emotion_results = []

    class FakeTensor:
        device = "cpu"

        def __init__(self, name):
            self.name = name

        def to(self, device):
            assert device == "cpu"
            return self

        def float(self):
            return self

        def size(self, dim):
            return 2

        def mean(self, **kwargs):
            return self

        def __sub__(self, other):
            return self

        def unsqueeze(self, dim):
            return self

    class ConditioningPrepared(Exception):
        pass

    def set_progress(value, description):
        if value == 0.1:
            raise ConditioningPrepared

    def load_audio(path, *args, **kwargs):
        load_calls.append((path, kwargs.get("sr")))
        return FakeTensor(path), kwargs.get("sr", 24000)

    def resample(source_rate, target_rate):
        def convert(audio):
            resample_calls.append((audio.name, source_rate, target_rate))
            return FakeTensor(audio.name)
        return convert

    def encode(features, mask):
        encoded = FakeTensor(features.name)
        encode_calls.append(encoded)
        return encoded

    tts.device = "cpu"
    tts.cache_spk_cond = None
    tts.cache_spk_audio_prompt = None
    tts.cache_emo_cond = None
    tts.cache_emo_audio_prompt = None
    tts._set_gr_progress = set_progress
    tts._load_and_cut_audio = load_audio
    tts.extract_features = lambda audio, **kwargs: {
        "input_features": audio,
        "attention_mask": FakeTensor("mask"),
    }
    tts.get_emb = encode
    tts.mel_fn = lambda audio: audio
    tts.campplus_model = lambda features: FakeTensor("style")
    tts.s2mel = types.SimpleNamespace(
        models={"length_regulator": lambda cond, **kwargs: (cond,)},
    )
    module.torch.cuda.empty_cache = lambda: None
    module.torch.LongTensor = lambda values: FakeTensor("lengths")
    module.torchaudio.transforms = types.SimpleNamespace(Resample=resample)
    module.torchaudio.compliance = types.SimpleNamespace(
        kaldi=types.SimpleNamespace(fbank=lambda audio, **kwargs: audio),
    )
    get_emo_cond_emb = tts._get_emo_cond_emb

    def record_emotion(*args):
        result = get_emo_cond_emb(*args)
        emotion_results.append(result)
        return result

    tts._get_emo_cond_emb = record_emotion

    def prepare(speaker, emotion=None):
        with pytest.raises(ConditioningPrepared):
            next(tts.infer_generator(
                spk_audio_prompt=speaker, emo_audio_prompt=emotion,
                text="hello", output_path=None, lang="en",
            ))

    prepare("speaker-a.wav", "emotion-e.wav")
    explicit_e = emotion_results[-1]
    assert tts.cache_emo_cond is explicit_e
    assert tts.cache_emo_audio_prompt == "emotion-e.wav"

    for speaker in ("speaker-a.wav", "speaker-b.wav"):
        prepare(speaker)
        if enabled:
            assert emotion_results[-1] is tts.cache_spk_cond
            assert tts.cache_emo_cond is explicit_e
            assert tts.cache_emo_audio_prompt == "emotion-e.wav"
        else:
            assert emotion_results[-1] is not tts.cache_spk_cond
            assert tts.cache_emo_audio_prompt == speaker

    prepare("speaker-b.wav", "emotion-e.wav")
    assert tts.cache_emo_audio_prompt == "emotion-e.wav"
    assert emotion_results[-1] is tts.cache_emo_cond
    assert (emotion_results[-1] is explicit_e) is enabled

    expected_loads = [("speaker-a.wav", None), ("emotion-e.wav", 16000)]
    if not enabled:
        expected_loads.append(("speaker-a.wav", 16000))
    expected_loads.append(("speaker-b.wav", None))
    if not enabled:
        expected_loads.extend([("speaker-b.wav", 16000), ("emotion-e.wav", 16000)])
    assert load_calls == expected_loads
    assert len(encode_calls) == len(expected_loads)
    assert resample_calls == [
        (speaker, 24000, rate)
        for speaker in ("speaker-a.wav", "speaker-b.wav")
        for rate in (22050, 16000)
    ]


def test_25_reuse_spk_cond_fast_path_uses_single_emovec_encoding(monkeypatch):
    _, tts = _new_v25_reuse_stub(monkeypatch)

    class FakeGpt:
        def __init__(self):
            self.get_calls = []
            self.merge_calls = []

        def get_emovec(self, cond, lengths):
            self.get_calls.append((cond, lengths))
            return "reused-emovec"

        def merge_emovec(self, *args, **kwargs):
            self.merge_calls.append((args, kwargs))
            return "merged-emovec"

    tts.gpt = FakeGpt()
    spk_cond = object()
    emo_cond = object()
    spk_lengths = object()
    emo_lengths = object()

    assert tts._get_emovec(spk_cond, spk_cond, spk_lengths, spk_lengths, 1.0, True) == "reused-emovec"
    assert len(tts.gpt.get_calls) == 1
    assert tts.gpt.merge_calls == []

    assert tts._get_emovec(spk_cond, emo_cond, spk_lengths, emo_lengths, 0.5, False) == "merged-emovec"
    assert len(tts.gpt.get_calls) == 1
    assert len(tts.gpt.merge_calls) == 1
    assert tts.gpt.merge_calls[0][1] == {"alpha": 0.5}


# -- Inference (GPU required) --------------------------------------------------

INFER_TEXTS = [
    "大家好，这是一段测试语音。",
    "There is a vehicle arriving in dock number 7?",
    "Joseph Gordon-Levitt is an American actor.",
]


@pytest.mark.gpu
@pytest.mark.parametrize("text", INFER_TEXTS, ids=lambda t: t[:20])
def test_infer(tts_model, prompt_wav, text, tmp_path):
    out = tmp_path / "out.wav"
    tts_model.infer(spk_audio_prompt=prompt_wav, text=text, output_path=str(out))
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_infer_with_emotion_vector(tts_model, prompt_wav, tmp_path):
    """infer() with explicit emotion vector."""
    emo_vec = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    out = tmp_path / "emo.wav"
    tts_model.infer(
        spk_audio_prompt=prompt_wav,
        text="今天天气真好，心情特别愉快！",
        output_path=str(out),
        emo_vector=emo_vec,
    )
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_infer_with_emo_text(tts_model, prompt_wav, tmp_path):
    """infer() with use_emo_text auto-detection."""
    out = tmp_path / "emo_text.wav"
    tts_model.infer(
        spk_audio_prompt=prompt_wav,
        text="这件事让我非常生气！",
        output_path=str(out),
        use_emo_text=True,
    )
    assert out.exists() and out.stat().st_size > 1000


# -- Long text (GPU required) -------------------------------------------------

@pytest.mark.gpu
def test_infer_long_text(tts_model, prompt_wav, tmp_path):
    text = (
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗诺兰执导并编剧，"
        "莱昂纳多迪卡普里奥、玛丽昂歌迪亚、约瑟夫高登莱维特、艾利奥特佩吉、"
        "汤姆哈迪等联袂主演，2010年7月16日在美国上映。"
        "影片剧情游走于梦境与现实之间，讲述了由莱昂纳多扮演的造梦师，"
        "带领特工团队进入他人梦境，从他人的潜意识中盗取机密的故事。"
    )
    out = tmp_path / "long.wav"
    tts_model.infer(spk_audio_prompt=prompt_wav, text=text, output_path=str(out))
    assert out.exists() and out.stat().st_size > 5000


# -- v2.5 Inference (GPU required) ---------------------------------------------

INFER_25_TEXTS = [
    ("zh", "大家好，这是一段测试语音。"),
    ("en", "There is a vehicle arriving in dock number 7?"),
    ("en", "Joseph Gordon-Levitt is an American actor."),
]


@pytest.mark.gpu
@pytest.mark.parametrize("lang,text", INFER_25_TEXTS, ids=lambda p: p[1][:20])
def test_25_infer(tts_model_25, prompt_wav, lang, text, tmp_path):
    out = tmp_path / "out.wav"
    tts_model_25.infer(spk_audio_prompt=prompt_wav, text=text, lang=lang, output_path=str(out))
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_25_infer_with_emotion_vector(tts_model_25, prompt_wav, tmp_path):
    emo_vec = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    out = tmp_path / "emo.wav"
    tts_model_25.infer(
        spk_audio_prompt=prompt_wav,
        text="今天天气真好，心情特别愉快！",
        lang="zh",
        output_path=str(out),
        emo_vector=emo_vec,
    )
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_25_infer_long_text(tts_model_25, prompt_wav, tmp_path):
    text = (
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗诺兰执导并编剧，"
        "莱昂纳多迪卡普里奥、玛丽昂歌迪亚、约瑟夫高登莱维特、艾利奥特佩吉、"
        "汤姆哈迪等联袂主演，2010年7月16日在美国上映。"
        "影片剧情游走于梦境与现实之间，讲述了由莱昂纳多扮演的造梦师，"
        "带领特工团队进入他人梦境，从他人的潜意识中盗取机密的故事。"
    )
    out = tmp_path / "long.wav"
    tts_model_25.infer(spk_audio_prompt=prompt_wav, text=text, lang="zh", output_path=str(out))
    assert out.exists() and out.stat().st_size > 5000


@pytest.mark.gpu
def test_25_infer_with_g2p_annotation(tts_model_25, prompt_wav, tmp_path):
    out = tmp_path / "g2p.wav"
    tts_model_25.infer(
        spk_audio_prompt=prompt_wav,
        text="他在银<行|XING2>里<行|HANG2>走了半天。",
        lang="zh",
        output_path=str(out),
    )
    assert out.exists() and out.stat().st_size > 1000
