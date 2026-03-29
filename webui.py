import html
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse

import gradio as gr

from indextts.infer_v2 import IndexTTS2
from indextts.webui_reference_inputs import (
    collect_emotion_references as collect_emotion_references_payload,
    collect_speaker_references as collect_speaker_references_payload,
    list_speaker_reference_targets,
)
from tools.i18n.i18n import I18nAuto


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

for file_name in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
]:
    file_path = os.path.join(cmd_args.model_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

i18n = I18nAuto(language="Auto")
tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
)

MAX_SPEAKER_ROWS = 8
MAX_EMOTION_ROWS = 8
EMOTION_VECTOR_SIZE = 8
EMOTION_TYPE_SPEAKER = "speaker"
EMOTION_TYPE_AUDIO = "audio"
EMOTION_TYPE_VECTOR = "vector"
EMOTION_TYPE_TEXT = "text"
EMOTION_TYPE_CHOICES = [
    (i18n("与音色参考相同"), EMOTION_TYPE_SPEAKER),
    (i18n("情感参考音频"), EMOTION_TYPE_AUDIO),
    (i18n("情感向量"), EMOTION_TYPE_VECTOR),
    (i18n("情感描述文本"), EMOTION_TYPE_TEXT),
]
SPEAKER_TARGET_LABEL = i18n("对应音色参考")
SPEAKER_TARGET_INFO = i18n("当存在多个音色参考时，请选择该情感行对应的音色参考。")
EMOTION_VECTOR_LABELS = [
    i18n("喜"),
    i18n("怒"),
    i18n("哀"),
    i18n("惧"),
    i18n("厌恶"),
    i18n("低落"),
    i18n("惊喜"),
    i18n("平静"),
]

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("logs", exist_ok=True)
WEBUI_LOG_PATH = os.path.join("logs", "webui_runtime.log")


def format_glossary_markdown():
    if not tts.normalizer.term_glossary:
        return i18n("暂无术语")

    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
    lines.append("|---|---|---|")

    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")

    return "\n".join(lines)


def create_warning_message(warning_text):
    return gr.HTML(
        f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">"
        f"{html.escape(warning_text)}</div>"
    )


def create_error_message(error_text):
    return (
        "<div style=\"padding: 0.75em 0.9em; border-radius: 0.5em; "
        "background: #ffe3e3; color: #8f1d1d; font-weight: 600; white-space: pre-wrap;\">"
        f"{html.escape(i18n('生成失败'))}\n{html.escape(error_text)}</div>"
    )


def create_info_message(info_text):
    return (
        "<div style=\"padding: 0.75em 0.9em; border-radius: 0.5em; "
        "background: #e7f1ff; color: #0f407a; font-weight: 600; white-space: pre-wrap;\">"
        f"{html.escape(info_text)}</div>"
    )


def clamp_weight(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def finalize_weights(weights, active_indices, anchor_index=None):
    finalized = [0.0] * len(weights)
    for index in active_indices:
        finalized[index] = round(weights[index], 2)

    target_sum = 1.0
    current_sum = round(sum(finalized[index] for index in active_indices), 2)
    delta = round(target_sum - current_sum, 2)
    if abs(delta) < 1e-9:
        return finalized

    candidate_indices = []
    if anchor_index is not None and anchor_index in active_indices:
        candidate_indices.append(anchor_index)
    candidate_indices.extend(index for index in reversed(active_indices) if index not in candidate_indices)

    for index in candidate_indices:
        updated = round(finalized[index] + delta, 2)
        if 0.0 <= updated <= 1.0:
            finalized[index] = updated
            return finalized

    fallback_index = active_indices[-1]
    finalized[fallback_index] = round(max(0.0, min(1.0, finalized[fallback_index] + delta)), 2)
    return finalized


def rebalance_weights(active_flags, weights, anchor_index=None):
    active_indices = [index for index, active in enumerate(active_flags) if bool(active)]
    normalized = [0.0] * len(weights)
    if not active_indices:
        return normalized

    if len(active_indices) == 1:
        normalized[active_indices[0]] = 1.0
        return normalized

    raw_weights = [clamp_weight(weight) for weight in weights]
    if anchor_index is not None and anchor_index in active_indices:
        anchor_value = raw_weights[anchor_index]
        remaining = max(0.0, 1.0 - anchor_value)
        other_indices = [index for index in active_indices if index != anchor_index]
        other_total = sum(raw_weights[index] for index in other_indices)
        normalized[anchor_index] = anchor_value
        if other_total > 0:
            for index in other_indices:
                normalized[index] = raw_weights[index] / other_total * remaining
        else:
            equal_share = remaining / len(other_indices)
            for index in other_indices:
                normalized[index] = equal_share
    else:
        total = sum(raw_weights[index] for index in active_indices)
        if total > 0:
            for index in active_indices:
                normalized[index] = raw_weights[index] / total
        else:
            equal_share = 1.0 / len(active_indices)
            for index in active_indices:
                normalized[index] = equal_share

    return finalize_weights(normalized, active_indices, anchor_index=anchor_index)


def make_weight_updates(weights):
    return [gr.update(value=value) for value in weights]


def update_weights_for_slider(anchor_index, *values):
    row_count = len(values) // 2
    active_flags = list(values[:row_count])
    weights = list(values[row_count:])
    return make_weight_updates(rebalance_weights(active_flags, weights, anchor_index=anchor_index))


def activate_next_row(*values):
    row_count = len(values) // 2
    active_flags = [bool(value) for value in values[:row_count]]
    weights = list(values[row_count:])
    for index, active in enumerate(active_flags):
        if not active:
            active_flags[index] = True
            break
    normalized = rebalance_weights(active_flags, weights)
    updates = [gr.update(value=active) for active in active_flags]
    updates.extend(gr.update(visible=active) for active in active_flags)
    updates.extend(make_weight_updates(normalized))
    return updates


def remove_row(row_index, minimum_active, *values):
    row_count = len(values) // 2
    active_flags = [bool(value) for value in values[:row_count]]
    weights = list(values[row_count:])
    active_count = sum(active_flags)
    if row_index < 0 or row_index >= row_count or not active_flags[row_index] or active_count <= minimum_active:
        normalized = rebalance_weights(active_flags, weights)
    else:
        active_flags[row_index] = False
        weights[row_index] = 0.0
        normalized = rebalance_weights(active_flags, weights)

    updates = [gr.update(value=active) for active in active_flags]
    updates.extend(gr.update(visible=active) for active in active_flags)
    updates.extend(make_weight_updates(normalized))
    return active_flags, updates


def emotion_type_visibility(row_type):
    row_type = row_type or EMOTION_TYPE_SPEAKER
    return (
        gr.update(visible=row_type == EMOTION_TYPE_SPEAKER),
        gr.update(visible=row_type == EMOTION_TYPE_AUDIO),
        gr.update(visible=row_type == EMOTION_TYPE_VECTOR),
        gr.update(visible=row_type == EMOTION_TYPE_TEXT),
    )


def reset_emotion_row(row_index, *values):
    row_count = len(values) // 2
    active_flags, updates = remove_row(row_index, 1, *values[: row_count * 2])
    default_type = EMOTION_TYPE_SPEAKER if row_index == 0 else EMOTION_TYPE_AUDIO
    row_defaults = [
        gr.update(value=default_type),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=""),
    ]
    row_defaults.extend(gr.update(value=0.0) for _ in range(EMOTION_VECTOR_SIZE))
    row_defaults.extend(
        [
            gr.update(visible=default_type == EMOTION_TYPE_SPEAKER),
            gr.update(visible=default_type == EMOTION_TYPE_AUDIO),
            gr.update(visible=False),
            gr.update(visible=False),
        ]
    )
    return updates + row_defaults




def collect_speaker_references(values):
    return collect_speaker_references_payload(values)


def collect_emotion_references(values, synthesis_text):
    return collect_emotion_references_payload(
        values,
        synthesis_text,
        tts.normalize_emo_vec,
        EMOTION_TYPE_SPEAKER,
        EMOTION_TYPE_AUDIO,
        EMOTION_TYPE_VECTOR,
        EMOTION_TYPE_TEXT,
    )


def build_speaker_target_choices(values):
    targets = list_speaker_reference_targets(values)
    choices = []
    for index, path in targets:
        label = f"{i18n('音色参考')} {index + 1}"
        filename = os.path.basename(path)
        if filename:
            label = f"{label}: {filename}"
        choices.append((label, index))
    return targets, choices


def refresh_speaker_target_selectors(*values):
    offset = 0
    speaker_values = []
    for _ in range(MAX_SPEAKER_ROWS):
        speaker_values.append(
            {
                "active": values[offset],
                "audio": values[offset + 1],
            }
        )
        offset += 2

    emotion_types = list(values[offset : offset + MAX_EMOTION_ROWS])
    offset += MAX_EMOTION_ROWS
    current_indices = list(values[offset : offset + MAX_EMOTION_ROWS])

    targets, choices = build_speaker_target_choices(speaker_values)
    valid_indices = {index for index, _ in targets}
    default_value = targets[0][0] if targets else None
    updates = []

    for row_type, current_index in zip(emotion_types, current_indices):
        selected = None
        if current_index not in (None, ""):
            try:
                candidate = int(current_index)
            except (TypeError, ValueError):
                candidate = None
            if candidate in valid_indices:
                selected = candidate

        if selected is None and row_type == EMOTION_TYPE_SPEAKER and default_value is not None:
            selected = default_value

        updates.append(
            gr.update(
                choices=choices,
                value=selected,
                interactive=bool(targets),
            )
        )

    return updates


def parse_generation_inputs(raw_values):
    offset = 0
    text = raw_values[offset]
    offset += 1
    emo_random = raw_values[offset]
    offset += 1
    max_text_tokens_per_segment = raw_values[offset]
    offset += 1

    do_sample = raw_values[offset]
    top_p = raw_values[offset + 1]
    top_k = raw_values[offset + 2]
    temperature = raw_values[offset + 3]
    length_penalty = raw_values[offset + 4]
    num_beams = raw_values[offset + 5]
    repetition_penalty = raw_values[offset + 6]
    max_mel_tokens = raw_values[offset + 7]
    offset += 8

    speaker_values = []
    for _ in range(MAX_SPEAKER_ROWS):
        speaker_values.append(
            {
                "active": raw_values[offset],
                "audio": raw_values[offset + 1],
                "weight": raw_values[offset + 2],
            }
        )
        offset += 3

    emotion_values = []
    for _ in range(MAX_EMOTION_ROWS):
        emotion_values.append(
            {
                "active": raw_values[offset],
                "type": raw_values[offset + 1],
                "speaker_index": raw_values[offset + 2],
                "audio": raw_values[offset + 3],
                "text": raw_values[offset + 4],
                "weight": raw_values[offset + 5],
                "vector": list(raw_values[offset + 6 : offset + 6 + EMOTION_VECTOR_SIZE]),
            }
        )
        offset += 6 + EMOTION_VECTOR_SIZE

    generation_kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }

    return {
        "text": text,
        "emo_random": emo_random,
        "max_text_tokens_per_segment": int(max_text_tokens_per_segment),
        "speaker_values": speaker_values,
        "emotion_values": emotion_values,
        "generation_kwargs": generation_kwargs,
    }


def gen_single(*raw_values, progress=gr.Progress()):
    try:
        parsed = parse_generation_inputs(raw_values)
        text = parsed["text"]
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
        tts.gr_progress = progress

        speaker_refs, speaker_weights = collect_speaker_references(parsed["speaker_values"])
        emotion_references = collect_emotion_references(parsed["emotion_values"], text)

        output = tts.infer(
            spk_audio_prompt=speaker_refs,
            text=text,
            output_path=output_path,
            emo_alpha=1.0,
            spk_audio_weights=speaker_weights,
            emotion_references=emotion_references,
            use_random=parsed["emo_random"],
            verbose=cmd_args.verbose,
            max_text_tokens_per_segment=parsed["max_text_tokens_per_segment"],
            **parsed["generation_kwargs"],
        )
        if output is None:
            raise RuntimeError("Generation completed without producing an audio file.")
        return (
            gr.update(value=output, visible=True),
            gr.update(value="", visible=False),
        )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}\nSee {WEBUI_LOG_PATH} for full details."
        print("WebUI generation failed:")
        traceback.print_exc()
        with open(WEBUI_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_message}\n")
            handle.write(traceback.format_exc())
            handle.write("\n")
        return (
            gr.update(value=None, visible=False),
            gr.update(value=create_error_message(error_message), visible=True),
        )


def prepare_generation():
    return (
        gr.update(value=None, visible=False),
        gr.update(value=create_info_message(i18n("Generating audio, please wait...")), visible=True),
    )


def on_input_text_change(text, max_text_tokens_per_segment):
    if text:
        text_tokens_list = tts.tokenizer.tokenize(text)
        segments = tts.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        )
        data = []
        for index, segment in enumerate(segments):
            data.append([index, "".join(segment), len(segment)])
        return {segments_preview: gr.update(value=data, visible=True, type="array")}

    df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
    return {segments_preview: gr.update(value=df)}


def on_add_glossary_term(term, reading_zh, reading_en):
    term = term.rstrip()
    reading_zh = reading_zh.rstrip()
    reading_en = reading_en.rstrip()

    if not term:
        gr.Warning(i18n("请输入术语"))
        return gr.update()

    if not reading_zh and not reading_en:
        gr.Warning(i18n("请至少输入一种读法"))
        return gr.update()

    if reading_zh and reading_en:
        reading = {"zh": reading_zh, "en": reading_en}
    elif reading_zh:
        reading = {"zh": reading_zh}
    else:
        reading = {"en": reading_en}

    tts.normalizer.term_glossary[term] = reading

    try:
        tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
        gr.Info(i18n("词汇表已更新"), duration=1)
    except Exception as exc:
        gr.Error(i18n("保存词汇表时出错"))
        print(f"Error details: {exc}")
        return gr.update()

    return gr.update(value=format_glossary_markdown())


def on_glossary_checkbox_change(is_enabled):
    tts.normalizer.enable_glossary = is_enabled
    return gr.update(visible=is_enabled)


def on_demo_load():
    try:
        tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
    except Exception as exc:
        gr.Error(i18n("加载词汇表时出错"))
        print(f"Failed to reload glossary on page load: {exc}")
    return gr.update(value=format_glossary_markdown())


speaker_rows = []
emotion_rows = []

with gr.Blocks(title="IndexTTS Demo") as demo:
    gr.HTML(
        """
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    """
    )

    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            input_text_single = gr.TextArea(
                label=i18n("文本"),
                key="input_text_single",
                placeholder=i18n("请输入目标文本"),
                info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}",
            )
            output_audio = gr.Audio(label=i18n("生成结果"), visible=True, key="output_audio")

        gen_button = gr.Button(i18n("生成语音"), key="gen_button", interactive=True, variant="primary")
        generation_status = gr.HTML(value="", visible=False)

        with gr.Accordion(i18n("音色参考音频"), open=True):
            gr.Markdown(i18n("多参考音色按权重归一化融合。拖动任一滑块时，其余行会自动调整以保持总和为 1。"))
            gr.HTML(
                """
                <div style="display:flex; gap:12px; font-weight:600; margin-bottom:8px;">
                    <div style="flex:4;">音频</div>
                    <div style="flex:2;">权重</div>
                    <div style="width:88px;">操作</div>
                </div>
                """
            )
            for index in range(MAX_SPEAKER_ROWS):
                active = index == 0
                with gr.Group(visible=active) as row_group:
                    with gr.Row():
                        row_audio = gr.Audio(
                            label=f"{i18n('音色参考')} {index + 1}",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                        row_weight = gr.Slider(
                            label=i18n("权重"),
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0 if active else 0.0,
                            step=0.01,
                        )
                        row_remove = gr.Button(i18n("移除"), interactive=index > 0)
                row_active = gr.Checkbox(value=active, visible=False)
                speaker_rows.append(
                    {
                        "active": row_active,
                        "group": row_group,
                        "audio": row_audio,
                        "weight": row_weight,
                        "remove": row_remove,
                    }
                )
            speaker_add_button = gr.Button(i18n("添加音色参考"))

        with gr.Accordion(i18n("情感参考"), open=True):
            create_warning_message(i18n("情感行支持与音色参考、参考音频、情感向量、情感文本混合融合，所有权重统一归一化。"))
            gr.HTML(
                """
                <div style="display:flex; gap:12px; font-weight:600; margin-bottom:8px;">
                    <div style="width:180px;">类型</div>
                    <div style="flex:4;">输入</div>
                    <div style="flex:2;">权重</div>
                    <div style="width:88px;">操作</div>
                </div>
                """
            )
            for index in range(MAX_EMOTION_ROWS):
                active = index == 0
                default_type = EMOTION_TYPE_SPEAKER if index == 0 else EMOTION_TYPE_AUDIO
                with gr.Group(visible=active) as row_group:
                    with gr.Row():
                        row_type = gr.Dropdown(
                            label=i18n("输入类型"),
                            choices=EMOTION_TYPE_CHOICES,
                            value=default_type,
                        )
                        with gr.Column(scale=4):
                            with gr.Group(visible=default_type == EMOTION_TYPE_SPEAKER) as speaker_group:
                                row_speaker_index = gr.Dropdown(
                                    label=SPEAKER_TARGET_LABEL,
                                    info=SPEAKER_TARGET_INFO,
                                    choices=[],
                                    value=None,
                                    allow_custom_value=False,
                                )
                            with gr.Group(visible=default_type == EMOTION_TYPE_AUDIO) as audio_group:
                                row_audio = gr.Audio(
                                    label=f"{i18n('情感参考音频')} {index + 1}",
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                )
                            with gr.Group(visible=False) as vector_group:
                                vector_components = []
                                for vector_offset, label in enumerate(EMOTION_VECTOR_LABELS):
                                    vector_components.append(
                                        gr.Slider(
                                            label=label,
                                            minimum=0.0,
                                            maximum=1.0,
                                            value=0.0,
                                            step=0.05,
                                        )
                                    )
                            with gr.Group(visible=False) as text_group:
                                row_text = gr.Textbox(
                                    label=i18n("情感描述文本"),
                                    placeholder=i18n("留空时自动使用目标文本"),
                                    value="",
                                )
                        row_weight = gr.Slider(
                            label=i18n("权重"),
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0 if active else 0.0,
                            step=0.01,
                        )
                        row_remove = gr.Button(i18n("移除"), interactive=index > 0)
                row_active = gr.Checkbox(value=active, visible=False)
                emotion_rows.append(
                    {
                        "active": row_active,
                        "group": row_group,
                        "type": row_type,
                        "speaker_group": speaker_group,
                        "speaker_index": row_speaker_index,
                        "audio_group": audio_group,
                        "vector_group": vector_group,
                        "text_group": text_group,
                        "audio": row_audio,
                        "text": row_text,
                        "vector": vector_components,
                        "weight": row_weight,
                        "remove": row_remove,
                    }
                )
            emotion_add_button = gr.Button(i18n("添加情感参考"))
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

        with gr.Row():
            glossary_checkbox = gr.Checkbox(
                label=i18n("开启术语词汇读音"),
                value=tts.normalizer.enable_glossary,
            )

        with gr.Accordion(
            i18n("自定义术语词汇读音"),
            open=False,
            visible=tts.normalizer.enable_glossary,
        ) as glossary_accordion:
            gr.Markdown(i18n("自定义个别专业术语的读音"))
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(label=i18n("术语"), placeholder="IndexTTS2")
                    glossary_reading_zh = gr.Textbox(label=i18n("中文读法"), placeholder="Index T-T-S 二")
                    glossary_reading_en = gr.Textbox(label=i18n("英文读法"), placeholder="Index T-T-S two")
                    btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(value=format_glossary_markdown())

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} "
                        "[Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._"
                    )
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(
                            label="repetition_penalty",
                            precision=None,
                            value=10.0,
                            minimum=0.1,
                            maximum=20.0,
                            step=0.1,
                        )
                        length_penalty = gr.Number(
                            label="length_penalty",
                            precision=None,
                            value=0.0,
                            minimum=-2.0,
                            maximum=2.0,
                            step=0.1,
                        )
                    max_mel_tokens = gr.Slider(
                        label="max_mel_tokens",
                        value=1500,
                        minimum=50,
                        maximum=tts.cfg.gpt.max_mel_tokens,
                        step=10,
                        info=i18n("生成Token最大数量，过小导致音频被截断"),
                    )
                with gr.Column(scale=2):
                    gr.Markdown(f"**{i18n('分句设置')}** _{i18n('参数会影响音频质量和生成速度')}_")
                    initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                    max_text_tokens_per_segment = gr.Slider(
                        label=i18n("分句最大Token数"),
                        value=initial_value,
                        minimum=20,
                        maximum=tts.cfg.gpt.max_text_tokens,
                        step=2,
                        info=i18n("建议80~200之间，值越大分句越长；值越小分句越碎。"),
                    )
                    with gr.Accordion(i18n("预览分句结果"), open=True):
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )

    speaker_active_inputs = [row["active"] for row in speaker_rows]
    speaker_weight_inputs = [row["weight"] for row in speaker_rows]
    speaker_visibility_outputs = [row["group"] for row in speaker_rows]
    speaker_weight_outputs = [row["weight"] for row in speaker_rows]
    speaker_management_outputs = speaker_active_inputs + speaker_visibility_outputs + speaker_weight_outputs

    emotion_active_inputs = [row["active"] for row in emotion_rows]
    emotion_weight_inputs = [row["weight"] for row in emotion_rows]
    emotion_visibility_outputs = [row["group"] for row in emotion_rows]
    emotion_weight_outputs = [row["weight"] for row in emotion_rows]
    emotion_management_outputs = emotion_active_inputs + emotion_visibility_outputs + emotion_weight_outputs
    speaker_selector_refresh_inputs = []
    for row in speaker_rows:
        speaker_selector_refresh_inputs.extend([row["active"], row["audio"]])
    emotion_type_inputs = [row["type"] for row in emotion_rows]
    emotion_speaker_index_inputs = [row["speaker_index"] for row in emotion_rows]
    speaker_selector_refresh_inputs.extend(emotion_type_inputs)
    speaker_selector_refresh_inputs.extend(emotion_speaker_index_inputs)
    speaker_selector_outputs = emotion_speaker_index_inputs

    for index, row in enumerate(speaker_rows):
        row["weight"].change(
            lambda *values, row_index=index: update_weights_for_slider(row_index, *values),
            inputs=speaker_active_inputs + speaker_weight_inputs,
            outputs=speaker_weight_outputs,
        )
        row["audio"].change(
            refresh_speaker_target_selectors,
            inputs=speaker_selector_refresh_inputs,
            outputs=speaker_selector_outputs,
        )

    speaker_add_button.click(
        activate_next_row,
        inputs=speaker_active_inputs + speaker_weight_inputs,
        outputs=speaker_management_outputs,
    ).then(
        refresh_speaker_target_selectors,
        inputs=speaker_selector_refresh_inputs,
        outputs=speaker_selector_outputs,
    )

    for index, row in enumerate(speaker_rows[1:], start=1):
        row["remove"].click(
            lambda *values, row_index=index: remove_row(row_index, 1, *values)[1] + [gr.update(value=None)],
            inputs=speaker_active_inputs + speaker_weight_inputs,
            outputs=speaker_management_outputs + [row["audio"]],
        ).then(
            refresh_speaker_target_selectors,
            inputs=speaker_selector_refresh_inputs,
            outputs=speaker_selector_outputs,
        )

    for index, row in enumerate(emotion_rows):
        row["weight"].change(
            lambda *values, row_index=index: update_weights_for_slider(row_index, *values),
            inputs=emotion_active_inputs + emotion_weight_inputs,
            outputs=emotion_weight_outputs,
        )
        row["type"].change(
            emotion_type_visibility,
            inputs=[row["type"]],
            outputs=[row["speaker_group"], row["audio_group"], row["vector_group"], row["text_group"]],
        ).then(
            refresh_speaker_target_selectors,
            inputs=speaker_selector_refresh_inputs,
            outputs=speaker_selector_outputs,
        )

    emotion_add_button.click(
        activate_next_row,
        inputs=emotion_active_inputs + emotion_weight_inputs,
        outputs=emotion_management_outputs,
    ).then(
        refresh_speaker_target_selectors,
        inputs=speaker_selector_refresh_inputs,
        outputs=speaker_selector_outputs,
    )

    for index, row in enumerate(emotion_rows[1:], start=1):
        row["remove"].click(
            lambda *values, row_index=index: reset_emotion_row(row_index, *values),
            inputs=emotion_active_inputs + emotion_weight_inputs,
            outputs=emotion_management_outputs
            + [row["type"], row["speaker_index"], row["audio"], row["text"]]
            + row["vector"]
            + [row["speaker_group"], row["audio_group"], row["vector_group"], row["text_group"]],
        ).then(
            refresh_speaker_target_selectors,
            inputs=speaker_selector_refresh_inputs,
            outputs=speaker_selector_outputs,
        )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview],
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview],
    )

    btn_add_term.click(
        on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table],
    )

    glossary_checkbox.change(
        on_glossary_checkbox_change,
        inputs=[glossary_checkbox],
        outputs=[glossary_accordion],
    )

    demo.load(
        on_demo_load,
        inputs=[],
        outputs=[glossary_table],
    )
    demo.load(
        refresh_speaker_target_selectors,
        inputs=speaker_selector_refresh_inputs,
        outputs=speaker_selector_outputs,
    )

    generation_inputs = [
        input_text_single,
        emo_random,
        max_text_tokens_per_segment,
        do_sample,
        top_p,
        top_k,
        temperature,
        length_penalty,
        num_beams,
        repetition_penalty,
        max_mel_tokens,
    ]
    for row in speaker_rows:
        generation_inputs.extend([row["active"], row["audio"], row["weight"]])
    for row in emotion_rows:
        generation_inputs.extend(
            [row["active"], row["type"], row["speaker_index"], row["audio"], row["text"], row["weight"], *row["vector"]]
        )

    gen_button.click(
        prepare_generation,
        inputs=[],
        outputs=[output_audio, generation_status],
    ).then(
        gen_single,
        inputs=generation_inputs,
        outputs=[output_audio, generation_status],
    )


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
