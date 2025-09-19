import html
import json
import os
import sys
import threading
import time

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None

        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES_ALL[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases  # show every example

    # exclude emotion control mode 3 (emotion from text description)
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label=i18n("音色参考音频"),key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("文本"),key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                gen_button = gr.Button(i18n("生成语音"), key="gen_button",interactive=True)
            output_audio = gr.Audio(label=i18n("生成结果"), visible=True,key="output_audio")

        experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)

        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0],label=i18n("情感控制方式"))
                # we MUST have an extra, INVISIBLE list of *all* emotion control
                # methods so that gr.Dataset() can fetch ALL control mode labels!
                # otherwise, the gr.Dataset()'s experimental labels would be empty!
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                    visible=False)  # do not render
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                      placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                      value="",
                                      info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        # we must use `gr.Dataset` to support dynamic UI rewrites, since `gr.Examples`
        # binds tightly to UI and always restores the initial state of all components,
        # such as the list of available choices in emo_control_method.
        example_table = gr.Dataset(label="Examples",
            samples_per_page=20,
            samples=get_example_cases(include_experimental=False),
            type="values",
            # these components are NOT "connected". it just reads the column labels/available
            # states from them, so we MUST link to the "all options" versions of all components,
            # such as `emo_control_method_all` (to be able to see EXPERIMENTAL text labels)!
            components=[prompt_audio,
                        emo_control_method_all,  # important: support all mode labels!
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        )

    def on_example_click(example):
        print(f"Example clicked: ({len(example)} values) = {example!r}")
        return (
            gr.update(value=example[0]),
            gr.update(value=example[1]),
            gr.update(value=example[2]),
            gr.update(value=example[3]),
            gr.update(value=example[4]),
            gr.update(value=example[5]),
            gr.update(value=example[6]),
            gr.update(value=example[7]),
            gr.update(value=example[8]),
            gr.update(value=example[9]),
            gr.update(value=example[10]),
            gr.update(value=example[11]),
            gr.update(value=example[12]),
            gr.update(value=example[13]),
        )

    # click() event works on both desktop and mobile UI
    example_table.click(on_example_click,
                        inputs=[example_table],
                        outputs=[prompt_audio,
                                 emo_control_method,
                                 input_text_single,
                                 emo_upload,
                                 emo_weight,
                                 emo_text,
                                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

        return (
            gr.update(choices=new_choices, value=new_choices[new_index]),
            gr.update(samples=get_example_cases(include_experimental=is_experimental)),
        )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method, example_table]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
