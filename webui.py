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
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")

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


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name="127.0.0.1")
