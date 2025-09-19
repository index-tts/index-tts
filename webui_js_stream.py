# webui_js_stream.py (FINAL VERSION 3.2 - UI Layout final)

import os
import sys
import time
import numpy as np
import warnings
import nltk
import soundfile as sf
import gradio as gr
import ffmpeg

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure NLTK tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.NLTKDownloadError:
    nltk.download("punkt")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
from indextts.infer_v2 import IndexTTS2

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--model_dir", type=str, default="./checkpoints")
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--deepspeed", action="store_true", default=False)
parser.add_argument("--cuda_kernel", action="store_true", default=False)
cmd_args = parser.parse_args()

# Validate model_dir
if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} missing.")
    sys.exit(1)

for f in ["bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]:
    if not os.path.exists(os.path.join(cmd_args.model_dir, f)):
        print(f"Missing required file: {f}")
        sys.exit(1)

tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
)

EMO_CHOICES_ALL = ["Same as Timbre Prompt", "Use Emotion Prompt", "Use Emotion Vectors", "Use Emotion Text"]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("outputs/assembled", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

def assemble_audio(file_paths):
    if not file_paths:
        return None
    
    output_path = os.path.join("outputs/assembled", f"full_audio_{int(time.time())}.wav")
    
    print(f"🔊 Assembling {len(file_paths)} segments into {output_path}...")
    
    try:
        input_streams = [ffmpeg.input(f) for f in file_paths]
        ffmpeg.concat(*input_streams, v=0, a=1).output(output_path).run(overwrite_output=True, quiet=True)
        print("✅ Assembly complete.")
        return output_path
    except ffmpeg.Error as e:
        print("❌ FFmpeg error:", e.stderr.decode() if e.stderr else "Unknown error")
        return None
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please ensure it is installed and in your system's PATH.")
        return None

def gen_stream_js(emo_control_method, prompt, text,
                  emo_ref_path, emo_weight,
                  vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                  emo_text, emo_random,
                  *args):
    # --- 1. Placeholder (silent) ---
    samplerate = 22050
    silent_audio = np.zeros(int(0.1 * samplerate), dtype=np.float32)
    placeholder_path = os.path.join("outputs", "silent_placeholder.wav")
    sf.write(placeholder_path, silent_audio, samplerate)
    yield placeholder_path, None

    # --- 2. Setup ---
    generated_files = []
    do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample), "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = None
    vec = None
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    if emo_text == "":
        emo_text = None
    
    # --- 3. Generate per sentence ---
    sentences = nltk.sent_tokenize(text)
    total = len(sentences)
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        print(f"📝 Generating {i+1}/{total}: {sentence[:60]}...")

        output_path = os.path.join("outputs", f"stream_{int(time.time())}_{i}.wav")
        tts.infer(
            spk_audio_prompt=prompt, text=sentence, output_path=output_path,
            emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight, emo_vector=vec,
            use_emo_text=(emo_control_method == 3), emo_text=emo_text,
            use_random=emo_random, verbose=cmd_args.verbose, **kwargs
        )
        generated_files.append(output_path)
        yield output_path, None
        
    # --- 4. Assemble the final audio ---
    final_audio_path = assemble_audio(generated_files)
    
    yield gr.update(value=None), final_audio_path

js_code = """
() => {
    function initPlayer() {
        let fileWrapper = document.getElementById('hidden_file');
        let audioPlayer = document.getElementById('custom_player');
        let progressBox = document.getElementById('progress_log').querySelector('textarea');

        if (!fileWrapper || !audioPlayer || !progressBox) {
            setTimeout(initPlayer, 500);
            return;
        }

        let audioQueue = [];
        let isPlaying = false;

        function log(msg) {
            console.log("[IndexTTS2]", msg);
            if (progressBox) {
                progressBox.value += msg + "\\n";
                progressBox.scrollTop = progressBox.scrollHeight;
            }
        }

        function playNext() {
            if (audioQueue.length > 0 && !isPlaying) {
                isPlaying = true;
                let src = audioQueue.shift();
                log("▶️ Playing: " + src.split('/').pop());
                audioPlayer.src = src;
                audioPlayer.play().catch(err => {
                    log("⚠️ Playback error: " + err);
                    isPlaying = false;
                    playNext();
                });
            }
        }

        audioPlayer.addEventListener('ended', () => {
            log("✅ Finished");
            isPlaying = false;
            playNext();
        });

        const observer = new MutationObserver(() => {
            let fileLink = fileWrapper.querySelector('a');
            if (fileLink) {
                const newSrc = fileLink.href;
                if (newSrc && !audioQueue.includes(newSrc) && !newSrc.endsWith('silent_placeholder.wav')) {
                    audioQueue.push(newSrc);
                    log("📥 Queued: " + newSrc.split('/').pop());
                    playNext();
                }
            }
        });

        observer.observe(fileWrapper, { childList: true, subtree: true });
        log("✅ Player initialized.");
    }

    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', initPlayer);
    } else {
        initPlayer();
    }
}
"""

css_code = """
.hidden-component {
    display: none !important;
}
"""

with gr.Blocks(js=js_code, css=css_code) as demo:
    gr.HTML("<h2><center>IndexTTS2: Continuous Playback Demo</center></h2>")

    with gr.Tab("Audio Generation"):
        with gr.Row():
            prompt_audio = gr.Audio(label="Timbre Prompt Audio", type="filepath")
            with gr.Column():
                input_text_single = gr.TextArea(label="Text", placeholder="Enter a paragraph of text...")
                gen_button = gr.Button("Generate and Play Stream", variant="primary")

        output_file = gr.File(
            label="Internal stream files",
            file_types=[".wav"],
            type="filepath",
            elem_id="hidden_file",
            elem_classes=["hidden-component"]
        )
        gr.HTML('<audio id="custom_player" controls autoplay></audio>')
        
        # --- UI LAYOUT CHANGE ---
        # The order of these two components is now swapped.
        final_audio = gr.Audio(label="Assembled Full Audio", type="filepath")
        progress_log = gr.Textbox(label="Progress Log", elem_id="progress_log", interactive=False, lines=14)
        
        with gr.Accordion("Advanced Settings", open=False):
            emo_control_method = gr.Radio(choices=EMO_CHOICES_OFFICIAL, type="index",
                                          value=EMO_CHOICES_OFFICIAL[0], label="Emotion Control Method")
            emo_ref_path = gr.Audio(label="Emotion Prompt Audio", type="filepath")
            emo_weight = gr.Slider(label="Emotion Weight", minimum=0.0, maximum=1.0, value=0.65, step=0.01)
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8 = [gr.Slider(visible=False)] * 8
            emo_text = gr.Textbox(visible=False)
            emo_random = gr.Checkbox(visible=False)
            advanced_params = [
                gr.Checkbox(label="do_sample", value=True),
                gr.Slider(label="top_p", value=0.8),
                gr.Slider(label="top_k", value=30),
                gr.Slider(label="temperature", value=0.8),
                gr.Number(label="length_penalty", value=0.0),
                gr.Slider(label="num_beams", value=3),
                gr.Number(label="repetition_penalty", value=10.0),
                gr.Slider(label="max_mel_tokens", minimum=50,
                          maximum=tts.cfg.gpt.max_mel_tokens, value=1500, step=10)
            ]

    gen_button.click(
        fn=gen_stream_js,
        inputs=[emo_control_method, prompt_audio, input_text_single,
                emo_ref_path, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, *advanced_params],
        outputs=[output_file, final_audio]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)