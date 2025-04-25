import os
import shutil
import sys
import threading
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import gradio as gr
from indextts.utils.webui_utils import next_page, prev_page

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")
MODE = 'local'

REQUIRED_FILES=[
    "bigvgan_discriminator.pth",
    "bigvgan_generator.pth",
    "bpe.model",
    "gpt.pth"
]



tts = None
def init_model(**model_args):
    global tts
    tts = IndexTTS(**model_args)


def download_files_from_hf_hub(model_dir, files):
    from huggingface_hub import hf_hub_download
    REPO_ID = os.getenv("HF_REPO_ID", "IndexTeam/Index-TTS")
    import multiprocessing
    with multiprocessing.Pool(processes=len(files)) as pool:
        pool.map(
            lambda file: hf_hub_download(
                repo_id=REPO_ID,
                filename=file,
                local_dir=model_dir,
            ),
            files,
        )


os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)


def gen_single(prompt, text, infer_mode, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    if infer_mode == "普通推理":
        output = tts.infer(prompt, text, output_path) # 普通推理
    else:
        output = tts.infer_fast(prompt, text, output_path) # 批次推理
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


with gr.Blocks() as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
    <h2><center>(一款工业级可控且高效的零样本文本转语音系统)</h2>

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
    ''')
    with gr.Tab("音频生成"):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label="请上传参考音频",key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label="请输入目标文本",key="input_text_single")
                infer_mode = gr.Radio(choices=["普通推理", "批次推理"], label="选择推理模式（批次推理：更适合长句，性能翻倍）",value="普通推理")
                gen_button = gr.Button("生成语音",key="gen_button",interactive=True)
            output_audio = gr.Audio(label="生成结果", visible=True,key="output_audio")

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[prompt_audio, input_text_single, infer_mode],
                     outputs=[output_audio])


def is_docker_env():
    if os.path.exists("/.dockerenv"):
        return True
    cgroup_path = "/proc/1/cgroup"
    if os.path.exists(cgroup_path):
        with open(cgroup_path, "r") as f:
            content = f.read()
            if "docker" in content or "kubepods" in content:
                return True
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IndexTTS WebUI")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0" if is_docker_env() else None,
        help="Host address for the web UI (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the web UI (default: 7860)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints",
        help="Path to the model directory (default: checkpoints)",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to the model config file (default: checkpoints/config.yaml)",
    )
    args = parser.parse_args()
    # Check if required files exist
    model_dir = args.model_dir
    cfg_path = args.cfg_path
    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' does not exist.")
        sys.exit(1)
    missed_files = []
    for file in REQUIRED_FILES:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            missed_files.append(file)
    if len(missed_files) > 0:
        print("Downloading files from Hugging Face Hub...")
        download_files_from_hf_hub(model_dir, missed_files)
    init_model(
        model_dir=model_dir,
        cfg_path=cfg_path,
    )
    demo.queue(20)
    demo.launch(server_name=args.host, server_port=args.port, share=False)
