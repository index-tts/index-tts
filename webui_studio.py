import os
import sys
import argparse
import threading
import time
import json
import glob
import warnings
import shutil
import re
import numpy as np
import tempfile
import yaml
import torch

try:
    import librosa
    import soundfile as sf
    from num2words import num2words
    from pydub import AudioSegment
    from PIL import Image
except ImportError:
    print(
        "⚠️  MISSING LIBRARIES! Please run: uv pip install librosa soundfile num2words pydub"
    )
    sys.exit(1)

HAS_FFMPEG = shutil.which("ffmpeg") is not None
if not HAS_FFMPEG:
    print(
        "⚠️  FFmpeg not found! MP3 conversion will be disabled in the UI. (Only WAV available)"
    )

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser(description="IndexTTS WebUI Pro")
parser.add_argument(
    "--verbose", action="store_true", default=False, help="Enable verbose mode"
)
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument(
    "--host", type=str, default="127.0.0.1", help="Host to run the web UI on"
)
parser.add_argument(
    "--model_dir", type=str, default="checkpoints", help="Model checkpoints directory"
)
parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
parser.add_argument(
    "--hf_mirror",
    type=str,
    default=None,
    help="Optional Hugging Face mirror endpoint (e.g., https://hf-mirror.com). Uses the official server by default.",
)
parser.add_argument(
    "--use_torch_compile", action="store_true", help="Enable torch.compile"
)
parser.add_argument(
    "--no_streaming", action="store_true", help="Disable streaming backend"
)
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist.")
    sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

if cmd_args.hf_mirror:
    print(f">> Setting Hugging Face endpoint to mirror: https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
else:
    print(">> Using default Hugging Face endpoint.")


hf_cache_dir = os.path.join(cmd_args.model_dir, "hf_cache")
torch_cache_dir = os.path.join(cmd_args.model_dir, "torch_cache")
os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")
os.environ.setdefault("HF_HOME", hf_cache_dir)
os.environ.setdefault("HF_HUB_CACHE", hf_cache_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
os.environ.setdefault("TORCH_HOME", torch_cache_dir)
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(torch_cache_dir, exist_ok=True)

print(">> Checking for required Hugging Face models...")
try:
    from huggingface_hub import hf_hub_download

    SPECIFIC_FILES = {
        "facebook/w2v-bert-2.0": {
            "name": "Semantic Model",
            "files": ["config.json", "preprocessor_config.json", "model.safetensors"],
        },
        "amphion/MaskGCT": {
            "name": "Semantic Codec",
            "files": ["semantic_codec/model.safetensors"],
        },
        "funasr/campplus": {
            "name": "Speaker Encoder (CAM++)",
            "files": ["campplus_cn_common.bin"],
        },
    }

    for repo_id, data in SPECIFIC_FILES.items():
        name = data["name"]
        print(f"   -> Verifying {name} ({repo_id})...")
        for file in data["files"]:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                cache_dir=hf_cache_dir,
                resume_download=True,
            )

    print(">> ✅ All core models are downloaded and ready.")

except ImportError:
    print("   ⚠️ huggingface_hub not found. Please run: pip install huggingface_hub")
    print("   Skipping automatic model download check.")
except Exception as e:
    print(f"\n   ❌ An error occurred during model download: {e}")
    print(
        "   Please check your internet connection. The application cannot start without these models."
    )
    sys.exit(1)

print(">> Loading libraries...")
import gradio as gr
from indextts.infer_studio import IndexTTS2

try:
    from studio_guide import GUIDE_MD
except ImportError:
    GUIDE_MD = "### Parameter guide file missing."

print(">> Initializing Model...")
tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    is_fp16=cmd_args.is_fp16,
    use_cuda_kernel=False,
    use_torch_compile=cmd_args.use_torch_compile,
)

EMO_CHOICES = [
    "Match prompt audio",
    "Use emotion reference audio",
    "Use emotion vector",
    "Use emotion text description",
]
OUTPUT_DIR = "outputs"
VOICE_DIR = "voices"
PRESETS_FILE = "presets.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)

DEFAULT_ICON_PATH = os.path.join(VOICE_DIR, "_default_user.png")
GUIDE_ICON_PATH = os.path.join(VOICE_DIR, "_guide_empty.png")

PRESETS = {
    "Neutral/Calm": [0, 0, 0, 0, 0, 0, 0, 1.0],
    "Happy": [1.0, 0, 0, 0, 0, 0, 0, 0],
    "Angry": [0, 1.0, 0, 0, 0, 0, 0, 0],
    "Sad": [0, 0, 1.0, 0, 0, 0, 0, 0],
    "Scared": [0, 0, 0, 1.0, 0, 0, 0, 0],
    "Surprised": [0, 0, 0, 0, 0, 0, 1.0, 0],
}


def get_voice_list():
    files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.WAV", "*.MP3"]:
        files.extend(glob.glob(os.path.join(VOICE_DIR, ext)))
    return sorted(list(set(files)))


def ensure_assets_exist():
    os.makedirs(VOICE_DIR, exist_ok=True)

    if not os.path.exists(DEFAULT_ICON_PATH):
        try:
            if Image:
                img = Image.new("RGB", (512, 512), color="#2563eb")
                img.save(DEFAULT_ICON_PATH)
        except Exception as e:
            print(f"Warning: Could not create default icon: {e}")
    if not os.path.exists(GUIDE_ICON_PATH):
        try:
            if Image:
                from PIL import ImageDraw

                img = Image.new("RGB", (512, 512), color="#1f2937")
                d = ImageDraw.Draw(img)

                msg = (
                    "\n"
                    "      ⚠️ LIBRARY IS EMPTY\n"
                    "   _________________________\n\n"
                    "   ✅ AUDIO SUPPORT:\n"
                    "      .wav / .WAV, .mp3 / .MP3, .flac\n\n"
                    "   ✅ COVER IMAGES:\n"
                    "      .png, .jpg, .jpeg, .webp\n\n"
                    "   ℹ️ INSTRUCTIONS:\n"
                    "   1. Paste files into 'voices'\n"
                    "   2. Match filenames:\n"
                    "      (voice.wav + voice.png)\n"
                    "   3. Click Refresh Button"
                )

                d.text((40, 40), msg, fill="white", spacing=12)
                img.save(GUIDE_ICON_PATH)
        except Exception as e:
            print(f"Warning: Could not create guide icon: {e}")


def get_voice_gallery_data():
    ensure_assets_exist()

    audio_files = get_voice_list()

    if not audio_files:
        if GUIDE_ICON_PATH and os.path.exists(GUIDE_ICON_PATH):
            return [(GUIDE_ICON_PATH, "Instructions")]
        return []

    gallery_items = []
    for audio_path in audio_files:
        if "_guide_empty" in audio_path or "_default_user" in audio_path:
            continue

        base_name = os.path.splitext(audio_path)[0]
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".webp"]:
            potential_path = base_name + ext
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        label = os.path.basename(audio_path)
        if img_path:
            gallery_items.append((img_path, label))
        elif DEFAULT_ICON_PATH and os.path.exists(DEFAULT_ICON_PATH):
            gallery_items.append((DEFAULT_ICON_PATH, label))
        else:
            gallery_items.append((None, label))
    return gallery_items


def normalize_text_english(text):
    text = re.sub(r"\$(\d+)", lambda m: num2words(m.group(1)) + " dollars", text)
    text = re.sub(r"\b(19|20)(\d{2})\b", lambda m: num2words(m.group(0)), text)
    text = re.sub(r"\d+", lambda m: num2words(m.group(0)), text)
    return text


def normalize_audio_loudness(audio_path):
    try:
        data, rate = sf.read(audio_path)
        peak = np.max(np.abs(data))
        target_peak = 0.95  # Target 95% volume (-0.5 dB)

        if peak <= 0:
            print(">> ⚠️ Normalizer: Audio is completely silent. Skipped.")
            return False

        change_ratio = target_peak / peak

        if 0.99 <= change_ratio <= 1.01:
            print(
                f">> ✅ Normalizer: Audio is already optimal (Peak: {peak:.2f}). No change needed."
            )
            return True

        new_data = data * change_ratio
        sf.write(audio_path, new_data, rate)

        if change_ratio > 1.0:
            percentage = (change_ratio - 1.0) * 100
            print(
                f">> 🔊 Normalizer: Too quiet. Boosted volume by {percentage:.1f}% (Peak: {peak:.2f} -> {target_peak})"
            )
        else:
            percentage = (1.0 - change_ratio) * 100
            print(
                f">> 🔉 Normalizer: Too loud. Reduced volume by {percentage:.1f}% (Peak: {peak:.2f} -> {target_peak})"
            )

        return True
    except Exception as e:
        print(f"Normalization error: {e}")
        return False


def trim_audio_if_needed(file_path):
    if not file_path:
        return None
    try:
        duration = librosa.get_duration(filename=file_path)
        if duration > 25.0:
            print(
                f">> Audio too long ({duration:.1f}s). Trimming to 25s for stability."
            )
            y, sr = librosa.load(file_path, sr=None, duration=25.0)
            base, ext = os.path.splitext(file_path)
            new_path = f"{base}_trimmed{ext}"
            sf.write(new_path, y, sr)
            return new_path
    except Exception as e:
        print(f"Error trimming audio: {e}")
    return file_path


def clean_reference_audio(audio_path):
    if not audio_path:
        return audio_path
    try:
        y, sr = librosa.load(audio_path, sr=None)
        non_silent_intervals = librosa.effects.split(y, top_db=30)

        if len(non_silent_intervals) > 0:
            start = non_silent_intervals[0][0]
            end = len(y)
            y = y[start:end]

        max_val = np.max(np.abs(y))
        if max_val > 0 and max_val < 0.6:  # Only boost if really quiet
            y = y / max_val * 0.8  # Boost to 80%

        base, ext = os.path.splitext(audio_path)
        new_path = f"{base}_clean{ext}"
        sf.write(new_path, y, sr)
        print(f">> 🧹 Cleaned reference audio (Gentle): {new_path}")
        return new_path
    except Exception as e:
        print(f"Cleaning failed, using original: {e}")
        return audio_path


def apply_audio_effects(audio_path, speed, pitch):
    if speed == 1.0 and pitch == 0:
        return audio_path
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if pitch != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(pitch))
        if speed != 1.0:
            y = librosa.effects.time_stretch(y, rate=float(speed))
        sf.write(audio_path, y, sr)
        print(f">> Applied effects: Speed={speed}x, Pitch={pitch}")
        return audio_path
    except Exception as e:
        print(f"Error applying effects: {e}")
        return audio_path


def convert_to_mp3(audio_path):
    if not HAS_FFMPEG:
        return audio_path
    try:
        mp3_path = os.path.splitext(audio_path)[0] + ".mp3"
        audio = AudioSegment.from_wav(audio_path)
        audio.export(mp3_path, format="mp3")
        print(f">> Converted to MP3: {mp3_path}")
        return mp3_path
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        return audio_path


def save_voice_to_lib(audio_path, name):
    if not audio_path:
        return gr.update(), "⚠️ No audio generated yet!"
    if not name.strip():
        return gr.update(), "⚠️ Enter a name first!"
    clean_name = "".join(
        [c for c in name if c.isalpha() or c.isdigit() or c in (" ", "-", "_")]
    ).strip()
    ext = os.path.splitext(audio_path)[1]
    target_file = os.path.join(VOICE_DIR, f"{clean_name}{ext}")
    try:
        shutil.copy(audio_path, target_file)
        srt_source = os.path.splitext(audio_path)[0] + ".srt"
        if os.path.exists(srt_source):
            shutil.copy(srt_source, os.path.join(VOICE_DIR, f"{clean_name}.srt"))
        new_list = get_voice_gallery_data()
        return (gr.update(value=new_list), f"✅ Saved: {clean_name}{ext}")
    except Exception as e:
        return gr.update(), f"❌ Error: {e}"


def cleanup_gradio_temp():
    try:
        sys_temp = tempfile.gettempdir()
        gradio_temp = os.path.join(sys_temp, "gradio")
        if not os.path.exists(gradio_temp):
            return "ℹ️ Gradio temp folder not found (Clean)."
        shutil.rmtree(gradio_temp, ignore_errors=True)
        return "🧹 Gradio temp folder cleaned."
    except Exception as e:
        return f"❌ Error: {e}"


# PRESET SYSTEM
def load_presets_file():
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_presets_file(data):
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_presets_table_data():
    data = load_presets_file()
    rows = []
    for name, p in data.items():
        rows.append(
            [
                name,
                int(p.get("diff_steps", 25)),
                float(p.get("inf_cfg", 1.0)),
                float(p.get("effect_speed", 1.0)),
                int(p.get("effect_pitch", 0)),
                float(p.get("temp", 1.0)),
                float(p.get("top_p", 0.95)),
                int(p.get("top_k", 50)),
            ]
        )
    return rows


def add_preset(name, *args):
    if not name.strip():
        return gr.update(), "⚠️ Preset name cannot be empty."
    data = load_presets_file()
    keys = [
        "diff_steps",
        "inf_cfg",
        "max_tokens",
        "effect_speed",
        "effect_pitch",
        "do_sample",
        "temp",
        "top_p",
        "top_k",
        "rep_pen",
        "max_mel",
        "normalize_txt",
        "normalize_vol",
        "split_text",
        "clean_ref_btn",
        "interval_silence",
    ]
    preset_data = {k: v for k, v in zip(keys, args)}
    data[name.strip()] = preset_data
    save_presets_file(data)
    return gr.update(value=get_presets_table_data()), f"✅ Saved preset: {name}"


def delete_preset(name):
    if not name.strip():
        return gr.update(), "⚠️ Enter name to delete."
    data = load_presets_file()
    if name.strip() in data:
        del data[name.strip()]
        save_presets_file(data)
        return gr.update(value=get_presets_table_data()), f"🗑️ Deleted: {name}"
    return gr.update(), "⚠️ Preset not found."


def apply_preset(evt: gr.SelectData):
    table_data = get_presets_table_data()
    row_idx = evt.index[0]
    if row_idx < len(table_data):
        preset_name = table_data[row_idx][0]
        all_data = load_presets_file()
        p = all_data.get(preset_name)
        if p:
            return (
                int(p.get("diff_steps", 25)),
                float(p.get("inf_cfg", 1.0)),
                int(p.get("max_tokens", 120)),
                float(p.get("effect_speed", 1.0)),
                int(p.get("effect_pitch", 0)),
                p.get("do_sample", True),
                float(p.get("temp", 1.0)),
                float(p.get("top_p", 1.0)),
                int(p.get("top_k", 50)),
                float(p.get("rep_pen", 10.0)),
                int(p.get("max_mel", 1500)),
                p.get("normalize_txt", True),
                p.get("normalize_vol", True),
                p.get("split_text", True),
                p.get("clean_ref_btn", False),
                int(p.get("interval_silence", 200)),
                preset_name,
            )
    return [gr.update()] * 16


# GLOSSARY HELPERS
def load_glossary_data():
    if os.path.exists(tts.glossary_path):
        with open(tts.glossary_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_glossary_data(data):
    with open(tts.glossary_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    tts.normalizer.load_glossary_from_yaml(tts.glossary_path)


def get_glossary_table_data():
    data = load_glossary_data()
    rows = []
    for k, v in data.items():
        zh = v.get("zh", "") if isinstance(v, dict) else v
        en = v.get("en", "") if isinstance(v, dict) else ""
        rows.append([k, zh, en])
    return rows


def add_glossary_item(term, zh, en):
    data = load_glossary_data()
    if term:
        data[term.strip()] = {"zh": zh.strip(), "en": en.strip()}
        save_glossary_data(data)
    return gr.update(value=get_glossary_table_data()), "✅ Updated"


def del_glossary_item(term):
    data = load_glossary_data()
    if term in data:
        del data[term]
        save_glossary_data(data)
    return gr.update(value=get_glossary_table_data()), "🗑️ Deleted"


# Core Logic
def generate_outputs(
    num_outputs,
    prompt,
    text,
    filename,
    output_fmt,
    emo_mode_idx,
    emo_ref,
    emo_weight,
    emo_random,
    emo_text,
    is_random_seed,
    seed_val,
    diff_steps,
    inf_cfg,
    vec_joy,
    vec_anger,
    vec_sad,
    vec_fear,
    vec_dis,
    vec_low,
    vec_sur,
    vec_calm,
    max_tokens,
    do_sample,
    top_p,
    top_k,
    temp,
    len_pen,
    beams,
    rep_pen,
    max_mel,
    normalize_txt,
    normalize_vol,
    split_text,
    effect_speed,
    effect_pitch,
    clean_ref,
    interval_silence,
    progress=gr.Progress(),
):
    vec = [vec_joy, vec_anger, vec_sad, vec_fear, vec_dis, vec_low, vec_sur, vec_calm]
    num_outputs = int(num_outputs)
    results = []

    if not prompt:
        raise gr.Error("Please select or upload a Prompt Voice!")
    if not text:
        raise gr.Error("Please enter text!")

    if normalize_txt:
        text = normalize_text_english(text)

    if isinstance(emo_mode_idx, str):
        mode_idx = EMO_CHOICES.index(emo_mode_idx) if emo_mode_idx in EMO_CHOICES else 0
    else:
        mode_idx = int(emo_mode_idx)

    if clean_ref:
        safe_prompt = clean_reference_audio(prompt)
        if mode_idx == 1 and emo_ref:
            emo_ref = clean_reference_audio(emo_ref)
    else:
        safe_prompt = trim_audio_if_needed(prompt)

    used_vec = vec if mode_idx == 2 else None
    if mode_idx == 0:
        emo_ref = None

    should_stream = (tts.device != "cpu") and (not cmd_args.no_streaming)

    for i in range(num_outputs):
        if is_random_seed:
            current_seed = -1
        else:
            current_seed = int(seed_val) + i

        if filename is None:
            filename = ""
        if filename.strip():
            safe_name = "".join(
                [c for c in filename if c.isalnum() or c in " -_"]
            ).strip()
            f_name = f"{safe_name}_v{i+1}.wav"
        else:
            f_name = f"studio_{int(time.time())}_{i+1}.wav"
        output_path = os.path.join(OUTPUT_DIR, f_name)

        generator = tts.infer(
            spk_audio_prompt=safe_prompt,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_ref,
            emo_alpha=float(emo_weight),
            emo_vector=used_vec,
            use_emo_text=(mode_idx == 3),
            emo_text=emo_text,
            use_random=emo_random,
            seed=current_seed,
            diffusion_steps=int(diff_steps),
            inference_cfg_rate=float(inf_cfg),
            verbose=cmd_args.verbose,
            max_text_tokens_per_sentence=int(max_tokens),
            do_sample=do_sample,
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temp),
            length_penalty=float(len_pen),
            num_beams=int(beams),
            repetition_penalty=float(rep_pen),
            max_mel_tokens=int(max_mel),
            interval_silence=int(interval_silence),
            split_text=split_text,
        )

        generated_audio = None
        used_seed_out = -1
        generated_srt = None

        for item in generator:
            if isinstance(item, tuple):
                if len(item) == 2 and isinstance(item[0], float):
                    total_progress = (i + item[0]) / num_outputs
                    progress(
                        total_progress,
                        desc=f"Generating {i+1}/{num_outputs}: {item[1]}",
                    )
                elif len(item) == 3 and torch.is_tensor(item[0]):
                    chunk, _, seg_text = item
                    if should_stream:
                        progress(
                            (i + 0.5) / num_outputs, desc=f"Streaming: {seg_text}..."
                        )
                elif len(item) == 3:
                    generated_audio, used_seed_out, generated_srt = item
                else:
                    generated_audio, used_seed_out = item

        # DUAL FILE LOGIC START
        final_path = None
        raw_path = None

        if generated_audio and os.path.exists(generated_audio):
            raw_path = generated_audio  # Raw AI output

            # Check if processing is needed
            needs_processing = (
                normalize_vol
                or effect_speed != 1.0
                or effect_pitch != 0
                or output_fmt == "mp3"
            )

            if needs_processing:
                base, ext = os.path.splitext(generated_audio)
                final_path = f"{base}_final{ext}"
                shutil.copy(raw_path, final_path)

                if normalize_vol:
                    normalize_audio_loudness(final_path)

                final_path = apply_audio_effects(final_path, effect_speed, effect_pitch)

                if output_fmt == "mp3":
                    final_path = convert_to_mp3(final_path)
            else:
                final_path = raw_path

            results.append((final_path, raw_path, used_seed_out, generated_srt))
        else:
            results.append((None, None, used_seed_out, None))
        # DUAL FILE LOGIC END

    final_updates = []
    for i in range(4):
        if i < len(results):
            final_path, raw_path, seed_val, srt_path = results[i]
            final_updates.extend(
                [
                    gr.update(visible=True, open=True),
                    gr.update(value=final_path, visible=True),
                    gr.update(value=raw_path, visible=True),
                    gr.update(value=srt_path, visible=bool(srt_path)),
                    gr.update(value=f"**Used Seed:** {seed_val}", visible=True),
                ]
            )
        else:
            final_updates.extend(
                [
                    gr.update(visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                ]
            )

    yield tuple(final_updates)


def on_gallery_select(evt: gr.SelectData):
    all_data = get_voice_gallery_data()
    if evt.index < len(all_data):
        filename = all_data[evt.index][1]
        full_path = os.path.join(VOICE_DIR, filename)
        if os.path.exists(full_path):
            return full_path
    return gr.update()


# UI Layout
css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600&family=Outfit:wght@500;700&display=swap');

/* Apply Fonts Globally */
body, .gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 11px;
}

/* Headers and Labels - Modern AI Look */
h1, h2, h3, h4, .block-label, .form-label, span.svelte-1gfkn6j {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
}

/* Text Areas and Inputs */
textarea, input, .gr-text-input {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important;
}

/* The Voice Gallery */
#voice_gallery_container {
    height: 380px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    display: block !important;
}
#voice_gallery_container .grid-wrap { max-height: none !important; overflow: visible !important; }
.gallery-item { border-radius: 8px !important; overflow: hidden; transition: transform 0.2s; }
.gallery-item:hover { transform: scale(1.02); }

"""

theme = gr.themes.Ocean(
    font=[
        gr.themes.GoogleFont("Plus Jakarta Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set()

with gr.Blocks(title="IndexTTS2 Studio", theme=theme, css=css) as demo:
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 10px;">
            <h1 style="font-family: 'Outfit', sans-serif; font-size: 2.5em; margin-bottom: 5px;">
                IndexTTS2 <span style="font-weight: 300; opacity: 0.7;">Studio</span>
            </h1>
            <p style="font-family: 'Plus Jakarta Sans', sans-serif; opacity: 0.8;">
                Emotionally Expressive Zero-Shot Text-to-Speech
            </p>
            <p align="center" style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.9em; opacity: 0.6;">
                <a href='https://github.com/nabil-aba' target='_blank' style="color: #60a5fa; text-decoration: none;">Nabil Aba</a> Studio Version Web Demo.
            </p>
        </div>
    """
    )

    # WORKFLOW & SETTINGS
    with gr.Row():
        # Voice Source
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 1. Voice Source")
                prompt_audio = gr.Audio(
                    label="Current Reference Voice (Drop File Here)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                with gr.Accordion("📚 Browse Voice Library", open=True):
                    with gr.Column(elem_id="voice_gallery_container"):
                        voice_gallery = gr.Gallery(
                            value=get_voice_gallery_data,
                            label="Click a voice to load it",
                            columns=3,
                            rows=None,
                            object_fit="cover",
                            allow_preview=False,
                            show_label=False,
                            container=False,
                            elem_id="inner_gallery",
                        )
                    refresh_lib = gr.Button("🔄 Refresh Library", size="sm")
                clean_ref_btn = gr.Checkbox(
                    label="✨ Auto-Clean Reference",
                    value=True,
                    info="High-pass filter. Disable if voice sounds too thin.",
                )
                refresh_lib.click(
                    lambda: gr.update(value=get_voice_gallery_data()),
                    outputs=voice_gallery,
                )
                voice_gallery.select(on_gallery_select, outputs=prompt_audio)

        # Text & Result
        with gr.Column(scale=2):
            # Text Input Group
            with gr.Group():
                gr.Markdown("### 2. Text Input")
                input_text = gr.TextArea(
                    label="Text", placeholder="Type your text here...", lines=3
                )
                with gr.Row():
                    normalize_txt = gr.Checkbox(
                        label="🧮 Convert Numbers",
                        value=True,
                        info="100 -> one hundred",
                    )
                    split_text = gr.Checkbox(
                        label="✂️ Split by (.!?)",
                        value=True,
                        info="Faster & Low RAM.",
                    )
                with gr.Row():
                    normalize_vol = gr.Checkbox(
                        label="🔊 Normalize Volume",
                        value=True,
                        info="Safe Peak Norm (Fix quiet audio)",
                    )
                    num_outputs_slider = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                        label="⚖️ Variations",
                    )
                output_fmt = (
                    gr.Radio(
                        choices=["wav", "mp3"],
                        value="wav",
                        label="Output Format",
                        interactive=True,
                    )
                    if HAS_FFMPEG
                    else gr.Radio(
                        choices=["wav"],
                        value="wav",
                        label="Output Format (MP3 Disabled)",
                        interactive=False,
                    )
                )
                gen_btn = gr.Button("🚀 Generate Audio", variant="primary", scale=2)

            # Result & Management Group
            with gr.Group():
                gr.Markdown("### 3. Result & Management")
                output_ui_flat_list = []
                for i in range(4):
                    with gr.Accordion(
                        f"Result {i+1}", open=(i == 0), visible=(i == 0)
                    ) as result_accordion:

                        with gr.Row():
                            output_audio_final = gr.Audio(
                                label="📢 Final (Norm + FX)",
                                interactive=False,
                                show_download_button=True,
                                elem_id=f"final_audio_{i}",
                            )
                            output_audio_raw = gr.Audio(
                                label="🔈 Original (Raw)",
                                interactive=False,
                                show_download_button=True,
                                elem_id=f"raw_audio_{i}",
                            )

                        output_srt = gr.File(
                            label="Download Subtitle (.srt)", visible=False
                        )
                        output_seed = gr.Markdown()

                        with gr.Group():
                            save_name = gr.Textbox(
                                placeholder=f"Name for Result {i+1}",
                                label="Save to Library",
                            )
                            with gr.Row():
                                save_btn = gr.Button("💾 Save Final", variant="primary")
                                recycle_btn = gr.Button(
                                    "♻️ Use Final as Ref", variant="secondary"
                                )
                            save_status = gr.Markdown("")

                        recycle_btn.click(
                            fn=lambda p=output_audio_final: p,
                            inputs=[output_audio_final],
                            outputs=[prompt_audio],
                        )
                        save_btn.click(
                            fn=save_voice_to_lib,
                            inputs=[output_audio_final, save_name],
                            outputs=[voice_gallery, save_status],
                        )

                        output_ui_flat_list.extend(
                            [
                                result_accordion,
                                output_audio_final,
                                output_audio_raw,
                                output_srt,
                                output_seed,
                            ]
                        )

        # Settings Tabs
        with gr.Column(scale=1):
            with gr.Tabs():
                # General
                with gr.Tab("🎛️ General"):
                    gr.Markdown("Flow & Expressiveness")
                    effect_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed Rate",
                    )
                    effect_pitch = gr.Slider(
                        minimum=-12, maximum=12, value=0, step=1, label="Pitch Shift"
                    )
                    interval_silence = gr.Slider(
                        minimum=0,
                        maximum=2000,
                        value=200,
                        step=50,
                        label="Silence (ms)",
                        info="Gap between sentences. 0 = Crossfade.",
                    )
                    temp = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature (Creativity)",
                        info="Higher = More emotional but unstable.",
                    )
                    inf_cfg = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="CFG (Similarity)",
                        info="Similarity to reference voice.",
                    )

                    gr.Markdown("---")
                    clean_temp_btn = gr.Button(
                        "🗑️ Clear Temp Files", size="sm", variant="secondary"
                    )
                    clean_status = gr.Markdown("")
                    clean_temp_btn.click(
                        cleanup_gradio_temp, inputs=[], outputs=[clean_status]
                    )

                # Quality
                with gr.Tab("🛠️ Quality"):
                    gr.Markdown("Generation Quality")
                    diff_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1,
                        label="Diffusion Steps",
                        info="Higher = Better Audio Quality.",
                    )
                    seed_check = gr.Checkbox(label="Random Seed", value=True)
                    seed_val = gr.Number(label="Seed", value=-1, interactive=False)
                    seed_check.change(
                        lambda x: gr.update(interactive=not x),
                        inputs=seed_check,
                        outputs=seed_val,
                    )
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=120,
                        step=10,
                        label="Max Text Tokens",
                        info="Chunk size. Keep ~150.",
                    )
                    max_mel = gr.Slider(
                        minimum=50,
                        maximum=2500,
                        value=1500,
                        step=50,
                        label="Max Audio Length",
                        info="Max length per segment.",
                    )

                # Extra
                with gr.Tab("🧠 Extra"):
                    gr.Markdown("Sampling Math")
                    do_sample = gr.Checkbox(
                        label="Do Sample",
                        value=True,
                        info="Uncheck for robotic/stable.",
                    )
                    top_p = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.95,
                        step=0.01,
                        label="Top P (Focus)",
                        info="Filters garbage data.",
                    )
                    top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top K",
                        info="Limits choices.",
                    )
                    rep_pen = gr.Number(
                        10.0, label="Repetition Penalty", info="Prevents loops."
                    )
                    len_pen = gr.Number(1.0, visible=False)
                    beams = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        label="Beam Search",
                        info="1 = Emotional/Fast (Recommended). >1 = Stable/Robotic (No Cut-off).",
                    )
                    custom_filename = gr.Textbox(
                        label="Custom Output Filename",
                        placeholder="Leave empty for auto-timestamp",
                    )

    gr.Markdown("---")

    # EMOTION SECTION
    with gr.Group():
        gr.Markdown("### 🎭 Emotion Settings")
        emo_mode = gr.Radio(
            choices=EMO_CHOICES,
            value=EMO_CHOICES[0],
            type="index",
            label="Emotion Source",
        )

        # Ref Audio
        with gr.Group(visible=False) as grp_ref:
            emo_ref_upload = gr.Audio(label="Emotion Audio Ref", type="filepath")
            emo_weight = gr.Slider(
                minimum=0.0, maximum=1.6, value=0.8, step=0.01, label="Strength"
            )

        # Vectors (4x4 Grid)
        with gr.Group(visible=False) as grp_vec:
            preset_dropdown = gr.Dropdown(
                label="⚡ Quick Presets",
                choices=list(PRESETS.keys()),
                value="Neutral/Calm",
            )
            emo_random = gr.Checkbox(label="Randomize Vector")
            with gr.Row():
                with gr.Column():
                    vec_joy = gr.Slider(0, 1.5, 0, label="Joy")
                    vec_anger = gr.Slider(0, 1.5, 0, label="Anger")
                    vec_sad = gr.Slider(0, 1.5, 0, label="Sad")
                    vec_fear = gr.Slider(0, 1.5, 0, label="Fear")
                with gr.Column():
                    vec_dis = gr.Slider(0, 1.5, 0, label="Disgust")
                    vec_low = gr.Slider(0, 1.5, 0, label="Low")
                    vec_sur = gr.Slider(0, 1.5, 0, label="Surprise")
                    vec_calm = gr.Slider(0, 1.5, 0, label="Calm")

            vec_comps = [
                vec_joy,
                vec_anger,
                vec_sad,
                vec_fear,
                vec_dis,
                vec_low,
                vec_sur,
                vec_calm,
            ]
            preset_dropdown.change(
                lambda x: PRESETS[x], inputs=preset_dropdown, outputs=vec_comps
            )

        # Text
        with gr.Group(visible=False) as grp_text:
            emo_text = gr.Textbox(
                label="Emotion Description", info="Requires Qwen model (Heavy!)"
            )

    gr.Markdown("---")

    # PRESETS SECTION
    with gr.Accordion("💾 Parameter Presets", open=True):
        with gr.Row():
            with gr.Column(scale=3):
                preset_table = gr.Dataframe(
                    headers=[
                        "Name",
                        "Steps",
                        "CFG",
                        "Speed",
                        "Pitch",
                        "Temp",
                        "TopP",
                        "TopK",
                    ],
                    datatype=[
                        "str",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                    ],
                    value=get_presets_table_data,
                    interactive=False,
                    label="Saved Presets (Click row to load settings)",
                    elem_id="preset_table",
                )
            with gr.Column(scale=1):
                gr.Markdown("#### Manage Presets")
                preset_name_in = gr.Textbox(
                    label="Preset Name", placeholder="e.g. My High Quality"
                )
                with gr.Row():
                    save_preset_btn = gr.Button("💾 Save", variant="primary")
                    del_preset_btn = gr.Button("🗑️ Delete", variant="secondary")
                refresh_preset_btn = gr.Button("🔄 Refresh Table")
                preset_msg = gr.Markdown("")

    # GLOSSARY SECTION
    with gr.Accordion("📖 Glossary Dictionary", open=False):
        with gr.Row():
            with gr.Column(scale=3):
                glossary_table = gr.Dataframe(
                    headers=["Term", "Chinese Reading", "English Reading"],
                    datatype=["str", "str", "str"],
                    value=get_glossary_table_data,
                    interactive=False,
                    label="Pronunciation Corrections",
                )
            with gr.Column(scale=1):
                gr.Markdown("#### Add Term")
                g_term_in = gr.Textbox(label="Term (e.g. IndexTTS)")
                g_zh_in = gr.Textbox(label="CN Reading (Optional)")
                g_en_in = gr.Textbox(label="EN Reading (e.g. Index T T S)")
                with gr.Row():
                    g_add_btn = gr.Button("Add/Update", variant="primary")
                    g_del_btn = gr.Button("Delete", variant="secondary")
                g_msg = gr.Markdown("")

    with gr.Accordion("ℹ️ IndexTTS2 Studio Guide", open=False):
        gr.Markdown(GUIDE_MD)

    def on_mode_change(mode):
        return [
            gr.update(visible=(mode == 1)),
            gr.update(visible=(mode == 2)),
            gr.update(visible=(mode == 3)),
        ]

    emo_mode.change(
        on_mode_change, inputs=[emo_mode], outputs=[grp_ref, grp_vec, grp_text]
    )

    all_inputs = [
        num_outputs_slider,
        prompt_audio,
        input_text,
        custom_filename,
        output_fmt,
        emo_mode,
        emo_ref_upload,
        emo_weight,
        emo_random,
        emo_text,
        seed_check,
        seed_val,
        diff_steps,
        inf_cfg,
        *vec_comps,
        max_tokens,
        do_sample,
        top_p,
        top_k,
        temp,
        len_pen,
        beams,
        rep_pen,
        max_mel,
        normalize_txt,
        normalize_vol,
        split_text,
        effect_speed,
        effect_pitch,
        clean_ref_btn,
        interval_silence,
    ]
    gen_btn.click(generate_outputs, inputs=all_inputs, outputs=output_ui_flat_list)

    preset_params = [
        diff_steps,
        inf_cfg,
        max_tokens,
        effect_speed,
        effect_pitch,
        do_sample,
        temp,
        top_p,
        top_k,
        rep_pen,
        max_mel,
        normalize_txt,
        normalize_vol,
        clean_ref_btn,
        interval_silence,
        split_text,
    ]
    save_preset_btn.click(
        add_preset,
        inputs=[preset_name_in] + preset_params,
        outputs=[preset_table, preset_msg],
    )
    del_preset_btn.click(
        delete_preset, inputs=[preset_name_in], outputs=[preset_table, preset_msg]
    )
    refresh_preset_btn.click(
        lambda: gr.update(value=get_presets_table_data()), outputs=preset_table
    )
    preset_table.select(
        apply_preset, inputs=[], outputs=preset_params + [preset_name_in]
    )

    # Glossary Logic
    g_add_btn.click(
        add_glossary_item,
        inputs=[g_term_in, g_zh_in, g_en_in],
        outputs=[glossary_table, g_msg],
    )
    g_del_btn.click(
        del_glossary_item, inputs=[g_term_in], outputs=[glossary_table, g_msg]
    )


def background_warmup():
    time.sleep(2)
    print(">> Starting background warmup...")
    tts.warmup()


if __name__ == "__main__":
    threading.Thread(target=background_warmup, daemon=True).start()
    print(f">> Launching on http://{cmd_args.host}:{cmd_args.port}")
    demo.queue(20).launch(
        server_name=cmd_args.host, server_port=cmd_args.port, inbrowser=True
    )
