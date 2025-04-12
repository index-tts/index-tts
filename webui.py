import os
import shutil
import sys
import threading
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import gradio as gr
from utils.webui_utils import next_page, prev_page

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

def infer(voice, text,output_path=None):
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    tts.infer(voice, text, output_path)
    return output_path

def gen_single(prompt, text):
    output_path = infer(prompt, text)
    return gr.update(value=output_path,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


with gr.Blocks() as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>

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
            input_text_single = gr.Textbox(label="请输入目标文本",key="input_text_single")
            gen_button = gr.Button("生成语音",key="gen_button",interactive=True)
            output_audio = gr.Audio(label="生成结果", visible=False,key="output_audio")

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[prompt_audio, input_text_single],
                     outputs=[output_audio])


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name="127.0.0.1")
