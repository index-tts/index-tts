# bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1
"""
只生成 conversation（并且合并 merged）
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1 --conversation_only --merge
只生成原来的单轴（不带 conversation）
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1
两者都要（单轴 + conversation）
bash /home/shentong/index-tts-main/axis_pack/run.sh --lines 1 --with_conversation --merge

"""
import argparse
import json
import logging
import random
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple, Any
from pydub import AudioSegment
import shutil
import sys
import unicodedata


# ---- PATCH: avoid torchaudio sox backend segfault when saving wav ----
try:
    import os
    import numpy as np
    import soundfile as sf
    import torchaudio

    def _safe_save(
        filepath,
        src,
        sample_rate,
        channels_first=True,
        format=None,
        encoding=None,
        bits_per_sample=None,
        buffer_size=4096,
        backend=None,
        compression=None,
    ):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # torch tensor -> numpy float32
        if hasattr(src, "detach"):
            x = src.detach().cpu().float().numpy()
        else:
            x = np.asarray(src, dtype=np.float32)

        # torchaudio default shape: [C, T] if channels_first=True
        # soundfile expects: [T, C]
        if x.ndim == 2:
            if channels_first:
                x = x.T  # [T, C]
        elif x.ndim == 1:
            pass
        else:
            # fallback flatten
            x = x.reshape(-1)

        # 1) 清理 NaN / Inf
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) 限制峰值，避免写PCM16时炸麦
        mx = float(np.max(np.abs(x))) if x.size else 0.0
        if mx > 1.0:
            x = x / mx * 0.95  # 归一化到 0.95，留余量

        # write PCM16 WAV (stable)
        sf.write(filepath, x, int(sample_rate), subtype="PCM_16")

    torchaudio.save = _safe_save
    print("[PATCH] torchaudio.save -> soundfile.write (PCM_16)")

except Exception as e:
    print("[PATCH] torchaudio.save patch skipped:", repr(e))
# ---- END PATCH ----

from indextts.infer_v2 import IndexTTS2

# ===================== 全局配置 =====================
EMOTION_DIMS = ["happy","angry","sad","afraid","disgusted","melancholic","surprised","calm"]
CRM_PROMPTS = {}
BASE_EMOTION_MAP = {}
MICROEXP_MAP = {}
SEMANTIC_EMO_MAP = {}
TEXT_PROCESS_CFG = {}
EMOTION_RHYTHM_TEMPLATES = {}
SCENE_BEHAVIOR_PROFILE = {}
FRAUD_KEYWORDS = {}
EMO_ALPHA_MAP = {}
SPEECH_RATE_CONFIG = {}

# ===================== 单轴导出 / Axis mask =====================
AXIS_MODE_ALL = "all"
AXIS_MODE_EMOTION_ONLY = "emotion_only"
AXIS_MODE_MICRO_ONLY = "micro_only"
AXIS_MODE_RATE_ONLY = "rate_only"

AXIS_MODES = {AXIS_MODE_ALL, AXIS_MODE_EMOTION_ONLY, AXIS_MODE_MICRO_ONLY, AXIS_MODE_RATE_ONLY}

def normalize_axis_mode(axis_mode: Optional[str]) -> str:
    """Normalize axis_mode and fallback to 'all'."""
    if not axis_mode:
        return AXIS_MODE_ALL
    axis_mode = str(axis_mode).strip().lower()
    if axis_mode not in AXIS_MODES:
        print(f"[WARN] Unknown axis_mode={axis_mode!r}, fallback to '{AXIS_MODE_ALL}'")
        return AXIS_MODE_ALL
    return axis_mode

def axis_mode_flags(axis_mode: Optional[str]) -> Tuple[bool, bool, bool, bool]:
    """Return (use_emo, use_micro, use_rate, use_intention)."""
    axis_mode = normalize_axis_mode(axis_mode)
    use_emo = axis_mode in (AXIS_MODE_ALL, AXIS_MODE_EMOTION_ONLY)
    use_micro = axis_mode in (AXIS_MODE_ALL, AXIS_MODE_MICRO_ONLY)
    use_rate = axis_mode in (AXIS_MODE_ALL, AXIS_MODE_RATE_ONLY)
    # intention (CRM prompt) is NOT part of the three axes; keep it only in all-mode
    use_intention = axis_mode == AXIS_MODE_ALL
    return use_emo, use_micro, use_rate, use_intention

def masked_labels(
    base_emo: str,
    micro: str,
    rate: str,
    intention: str,
    axis_mode: Optional[str]
) -> Tuple[str, str, str, str]:
    """Mask non-selected axes to baseline values."""
    use_emo, use_micro, use_rate, use_intention = axis_mode_flags(axis_mode)
    emo_out = base_emo if use_emo else "正常"
    micro_out = micro if use_micro else "正常"
    rate_out = rate if use_rate else "正常"
    intention_out = intention if use_intention else ""
    return emo_out, micro_out, rate_out, intention_out

def map_scene(final_label: str) -> str:
    if not final_label:
        return "normal"
    if "正常通话" in final_label:
        return "normal"
    if "涉嫌诈骗" in final_label:
        return "suspicious"
    if "确定诈骗" in final_label:
        return "fraud"
    return "normal"

# ===================== 配置加载 =====================
def load_configs(cfg_dir: str) -> None:
    import json, os

    def _load_json(name):
        path = os.path.join(cfg_dir, name)
        if not os.path.exists(path):
            return {}
        return json.load(open(path, "r", encoding="utf-8"))

    def _load_yaml(path):
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("Missing dependency: pyyaml. Install via pip install pyyaml") from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    global CRM_PROMPTS, BASE_EMOTION_MAP, MICROEXP_MAP
    global SEMANTIC_EMO_MAP, TEXT_PROCESS_CFG
    global EMOTION_RHYTHM_TEMPLATES, SCENE_BEHAVIOR_PROFILE, FRAUD_KEYWORDS
    global EMO_ALPHA_MAP, SPEECH_RATE_CONFIG

    yaml_path = os.path.join(cfg_dir, "config.yaml")
    if os.path.exists(yaml_path):
        cfg = _load_yaml(yaml_path)
        CRM_PROMPTS = cfg.get("crm_prompts", {}) or {}
        BASE_EMOTION_MAP = cfg.get("base_emotions", {}) or {}
        MICROEXP_MAP = cfg.get("microexpressions", {}) or {}
        SEMANTIC_EMO_MAP = cfg.get("semantic_emotions", {}) or {}
        TEXT_PROCESS_CFG = cfg.get("text_process_config", {}) or {}
        EMOTION_RHYTHM_TEMPLATES = cfg.get("emotion_rhythm_templates", {}) or {}
        FRAUD_KEYWORDS = cfg.get("fraud_keywords", {}) or {}
        SCENE_BEHAVIOR_PROFILE = cfg.get("scene_behavior_profile", {}) or {}
        EMO_ALPHA_MAP = cfg.get("emotion_alpha_map", {}) or {}
        SPEECH_RATE_CONFIG = cfg.get("speech_rate_config", {}) or {}
        print(f"[CFG] Loaded YAML: {yaml_path}")
        return

    # fallback: old json layout
    CRM_PROMPTS = _load_json("crm_prompts.json")
    BASE_EMOTION_MAP = _load_json("base_emotions.json")
    MICROEXP_MAP = _load_json("microexpressions.json")
    SEMANTIC_EMO_MAP = _load_json("semantic_emotions.json")
    TEXT_PROCESS_CFG = _load_json("text_process_config.json")
    EMOTION_RHYTHM_TEMPLATES = _load_json("emotion_rhythm_templates.json")
    FRAUD_KEYWORDS = _load_json("fraud_keywords.json")
    SCENE_BEHAVIOR_PROFILE = _load_json("scene_behavior_profile.json")
    EMO_ALPHA_MAP = _load_json("emotion_alpha_map.json")
    SPEECH_RATE_CONFIG = _load_json("speech_rate_config.json")
    print(f"[CFG] Loaded JSON configs from: {cfg_dir}")

def safe_filename_readable(s: str, max_len: int = 24) -> str:
    """
    生成“可读”的文件名片段：保留中文/英文/数字/常见符号，去掉路径非法字符，压缩空白，控制长度。
    适用于 Linux/Windows 通用文件名（避免 : * ? " < > | / \\ 等）。
    """
    if s is None:
        return "utt"

    s = str(s).strip()
    if not s:
        return "utt"

    # 统一字符形态（全角/半角等）
    s = unicodedata.normalize("NFKC", s)

    # 去掉换行/制表等空白，压成单空格
    s = re.sub(r"\s+", " ", s)

    # 替换 Windows/Linux 都不允许的字符
    # 以及控制字符
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", s)

    # 进一步把太“怪”的标点收敛一下（可选，但推荐）
    s = s.replace("…", "_").replace("—", "_")

    # 只保留：中文、字母数字、空格、下划线、连字符、点
    # 其他字符替换为下划线（避免表情符/罕见符号导致问题）
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9 _\.-]", "_", s)

    # 多个下划线合并
    s = re.sub(r"_+", "_", s).strip(" _.-")

    # 截断长度
    if len(s) > max_len:
        s = s[:max_len].rstrip(" _.-")

    return s if s else "utt"

# ===================== 日志 =====================
class TeeLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
    def write(self, msg):
        msg = msg.rstrip()
        if msg:
            self.logger.log(self.level, msg)
    def flush(self): pass

def add_local_log_handler(logger, log_path):
    h = logging.FileHandler(log_path, encoding="utf-8")
    h.setLevel(logging.INFO)
    logger.addHandler(h)
    return h

def get_rate_config(rate: str) -> dict:
    """
    获取语速配置
    """
    if not SPEECH_RATE_CONFIG:
        return {}
    return SPEECH_RATE_CONFIG.get(rate, SPEECH_RATE_CONFIG.get("正常", {}))

def limit_ellipsis(text: str) -> str:
    """
    限制省略号数量，最多两个连续省略号
    """
    # 将连续的省略号替换为最多两个
    text = re.sub(r"…{3,}", r"…", text)
    # 将非末尾位置的单个省略号保留，末尾的也保留
    return text

def enhance_text_with_rhythm_marks(text: str, rate: str) -> str:
    """
    根据语速添加韵律标记
    """
    print(f"[RATE] 语速韵律标记处理: rate={rate}")

    if rate == "语速加快":
        text = re.sub(r"([，。？！…])", r"\1", text)
    elif rate == "语速放慢":
        # 去掉句子开头的省略号
        text = re.sub(r"^…+", "", text)
        # 犹豫情绪不添加额外的省略号
        # 只在非省略号的标点符号后面添加省略号（概率较低）
        text = re.sub(r"([，。？！])", r"\1", text)
    elif rate == "不自然停顿":
        # 不自然停顿模式：不添加额外的省略号，保持原始文本
        pass

    # 限制省略号数量
    text = limit_ellipsis(text)
    print(f"[RATE] 韵律标记处理结果: {text}")
    return text

def get_rate_emo_vector_adjustment(rate: str) -> List[float]:
    """
    获取语速对情绪向量的调整值
    """
    config = get_rate_config(rate)
    return config.get("emo_vector_adjustment", [0.0] * 8)

def get_rate_pause_params(rate: str, position: str = "between_chunks") -> dict:
    """
    获取语速停顿参数
    position: "before_chunk" 或 "between_chunks"
    """
    config = get_rate_config(rate)

    if position == "before_chunk":
        return {
            "short_pause": config.get("short_pause", 100),
            "long_pause": config.get("long_pause", 200),
            "long_pause_prob": config.get("long_pause_prob", 0.1)
        }
    else:
        return {
            "short_pause": config.get("short_pause", 100),
            "long_pause": config.get("long_pause", 200),
            "long_pause_prob": config.get("long_pause_prob", 0.1)
        }

def get_rate_fade_duration(rate: str) -> int:
    """
    获取语速对应的淡入淡出时长
    """
    config = get_rate_config(rate)
    return config.get("fade_duration", 40)

def get_rate_speed_factor(rate: str) -> float:
    """
    获取语速速度因子
    """
    config = get_rate_config(rate)
    return config.get("speed_factor", 1.0)

def adjust_audio_speed(input_wav: str, output_wav: str, rate: str) -> bool:
    """
    使用 pydub 调整音频速度
    注意：pydub 的速度调整会改变音调，如果需要保持音调，可以使用其他库如 librosa
    """
    try:
        from pydub import AudioSegment

        speed_factor = get_rate_speed_factor(rate)

        if speed_factor == 1.0:
            # 语速正常，直接复制文件
            import shutil
            shutil.copy2(input_wav, output_wav)
            print(f"[RATE] 语速正常，直接复制音频")
            return True

        print(f"[RATE] 调整音频速度: {input_wav} -> {output_wav}, 速度因子: {speed_factor}")

        audio = AudioSegment.from_wav(input_wav)

        # 使用 frame_rate 调整速度（会改变音调）
        new_frame_rate = int(audio.frame_rate * speed_factor)
        audio_speed_adjusted = audio._spawn(audio.raw_data, overrides={
            "frame_rate": new_frame_rate
        }).set_frame_rate(audio.frame_rate)

        audio_speed_adjusted.export(output_wav, format="wav")
        print(f"[RATE] 音频速度调整完成")
        return True

    except Exception as e:
        print(f"[ERROR] 音频速度调整失败：{e}")
        import traceback
        traceback.print_exc()
        # 失败时直接复制原始文件
        try:
            import shutil
            shutil.copy2(input_wav, output_wav)
            return True
        except Exception as copy_error:
            print(f"[ERROR] 复制文件失败：{copy_error}")
            return False

# ===================== 文本处理 =====================
def number_to_chinese(num_str: str) -> str:
    """
    将数字转换为中文
    例如：60 -> 六十，120 -> 一百二十
    """
    try:
        num = int(num_str)
        if num == 0:
            return "零"

        digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        units = ["", "十", "百", "千", "万", "十", "百", "千", "亿"]

        result = []
        num_str = str(num)
        length = len(num_str)

        for i, digit in enumerate(num_str):
            digit_int = int(digit)
            if digit_int == 0:
                if i < length - 1 and num_str[i+1] != '0':
                    result.append(digits[0])
            else:
                result.append(digits[digit_int])
                if i < length - 1:
                    unit_pos = length - i - 1
                    result.append(units[unit_pos])

        chinese_num = ''.join(result)

        # 处理特殊情况：10-19
        if 10 <= num < 20:
            chinese_num = chinese_num.replace("一十", "十")

        return chinese_num
    except:
        return num_str

DIGIT_CN = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"
}

def read_digits_separately(num: str, sep: str = " ") -> str:
    """
    逐位读数字：6217 -> 六 二 一 七
    """
    return sep.join(DIGIT_CN.get(c, c) for c in num)


def normalize_text_for_tts(text: str) -> str:
    """
    最终稳定版文本标准化：
    - 金额绝对保护（不拆、不转中文）
    - 账号 / 卡号 / 验证码：逐位读
    - 抖音ID：- → 杠，_ → 下划线，V-L-O-G → V……L……O……G
    - 银行卡省略号：…… = 停顿
    - 不产生 tuple / replace 冲突
    """

    print(f"[DEBUG] 原始文本: {text}")
    text = re.sub(r"\s+", " ", text.strip())

    # ======================================================
    # 1️⃣ 金额锁定（最高优先级）
    # ======================================================
    amount_pattern = re.compile(r"\d+(?:\.\d+)?\s*(?:元|块)")
    amount_slots = []

    def _lock_amount(m):
        key = f"__AMOUNT_{len(amount_slots)}__"
        amount_slots.append(m.group(0))
        return key

    text = amount_pattern.sub(_lock_amount, text)

    # ======================================================
    # 2️⃣ 抖音ID / 账号类文本规则
    # ======================================================
    if any(k in text for k in ["抖音ID", "抖音号", "抖音账号"]):
        # 下划线
        text = text.replace("_", "下划线")

        # V-L-O-G → V……L……O……G
        text = re.sub(
            r"\b([A-Z])(?:-([A-Z])){1,}\b",
            lambda m: "……".join(m.group(0).split("-")),
            text
        )

        # 普通字母-字母 → 杠
        text = re.sub(r"([a-zA-Z])-([a-zA-Z])", r"\1杠\2", text)

    # ======================================================
    # 3️⃣ 银行卡 / 账号 / 验证码：逐位读
    # ======================================================
    def read_digits(match):
        return " ".join(DIGIT_CN.get(c, c) for c in match.group(0))

    if any(k in text for k in ["卡号", "银行卡", "账号", "账户", "收款人", "验证码", "认证代码"]):
        # 数字逐位
        text = re.sub(r"\d{4,}", read_digits, text)

        # ... 或 .... → ……
        text = re.sub(r"\.{3,}", "……", text)

    # ======================================================
    # 4️⃣ 时间单位数字转中文（不含金额）
    # ======================================================
    def _time_cn(m):
        return number_to_chinese(m.group(1)) + m.group(2)

    text = re.sub(r"(\d+)(秒|分|小时|天|周|月|年)", _time_cn, text)

    # ======================================================
    # 5️⃣ 金额恢复（最后一步，绝不再碰）
    # ======================================================
    for i, amt in enumerate(amount_slots):
        text = text.replace(f"__AMOUNT_{i}__", amt)

    print(f"[DEBUG] 标准化后文本: {text}")
    return text

def apply_text_process(text: str) -> str:
    print(f"[DEBUG] 应用文本处理前: {text}")
    cfg = TEXT_PROCESS_CFG
    if not cfg:
        print(f"[DEBUG] 无文本处理配置，返回原文本")
        return text
    original = text
    # 感叹
    ex = cfg.get("punctuation",{}).get("exclamation",{})
    if ex:
        text = re.sub(ex["pattern"], ex["replace"], text)
    # 疑问
    q = cfg.get("punctuation",{}).get("question",{})
    if q:
        text = re.sub(q["pattern"], q["replace"], text)
    # 工号
    job = cfg.get("job_id")  #可拓展
    if isinstance(job,dict) and "pattern" in job:
        def _job(m):
            return m.group(1) + " " + " ".join(m.group(2))
        text = re.sub(job["pattern"], _job, text)

    if text != original:
        print(f"[DEBUG] 文本处理结果: {text}")
    else:
        print(f"[DEBUG] 文本无变化")
    return text

def apply_emotion_rhythm_template(text: str, emo: str) -> str:
    cfg = EMOTION_RHYTHM_TEMPLATES.get(emo)
    if not cfg:
        print(f"[DEBUG] 无对应情绪模板，返回原文本")
        return text
    print(f"[DEBUG] 找到情绪模板，规则数量: {len(cfg.get('rules', []))}")
    for rule in cfg.get("rules", []):
        if "prob" in rule and random.random() > rule["prob"]:
            continue

        pattern = rule["pattern"]
        replace = rule["replace"]
        count = rule.get("count", 0)
        desc = rule.get("desc","")
        if desc == "处理连续标点":
            # 特殊处理连续标点
            text = re.sub(pattern, replace, text)
        elif count > 0:
            text = re.sub(pattern, replace, text, count=count)
        else:
            text = re.sub(pattern, replace, text)
    print(f"[DEBUG] 情绪节奏处理后: {text}")
    return text

def split_text_for_semantic_safety(
    text: str,
    max_len: int = 28,
    rate: str = "正常"
) -> List[str]:
    """
    防止语义重复的安全切块：
    - 控制每块长度
    - 只在自然语言边界切
    - 根据语速动态调整切分策略
    """
    print(f"[DEBUG] 开始语义安全切分文本，最大长度={max_len},语速={rate}")

    # 使用统一的语速配置
    rate_config = get_rate_config(rate)
    adjusted_max_len = rate_config.get("max_len", max_len)
    min_len = rate_config.get("min_len", 10)
    pause_chars = rate_config.get("pause_chars", ["，", "。"])
    split_prob = rate_config.get("split_prob", 0.1)

    print(f"[DEBUG] 语速调整: max_len={adjusted_max_len}, min_len={min_len}, pause_chars={pause_chars}, split_prob={split_prob}")

    chunks = []
    buf = ""

    for ch in text:
        buf += ch
        # 正常切分逻辑：在标点符号处切分
        # 检查缓冲区是否包含实际文字内容，避免只有符号单独成块
        if ch in pause_chars and len(buf) >= adjusted_max_len:
            # 检查缓冲区中是否有非标点符号的文字内容
            has_text = any(c not in pause_chars and not c.isspace() for c in buf)
            if has_text:
                chunks.append(buf)
                buf = ""

    if buf.strip():
        chunks.append(buf)

    # 忽快忽慢：随机合并小分块
    if rate == "忽快忽慢" and len(chunks) > 2:
        merged_chunks = []
        i = 0
        while i < len(chunks):
            if i < len(chunks) - 1 and random.random() < 0.3:
                merged_chunks.append(chunks[i] + chunks[i+1])
                i += 2
            else:
                merged_chunks.append(chunks[i])
                i += 1
        chunks = merged_chunks

    print(f"[DEBUG] 切分结果: {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"  块{i + 1} ({len(chunk)}字): {chunk}")
    return chunks

import os
import re
import uuid

def sanitize_ascii(name: str, max_len: int = 80) -> str:
    """把文件名压成 ASCII 安全短名（用于中间文件），避免中文标点/超长路径引发写文件异常。"""
    name = name.lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:max_len] if len(name) > max_len else name

def is_valid_wav_header(path: str) -> bool:
    """快速检查 WAV 头是否 RIFF....WAVE（不依赖 soundfile/wave）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
        return len(head) == 12 and head[0:4] == b"RIFF" and head[8:12] == b"WAVE"
    except Exception:
        return False

# ===================== emo ===================
def calculate_energy_gain(micro: str, emo: str, rate: str) -> float:
    """
    计算能量增益值（dB）
    统一管理微表情和情绪对能量的影响
    """
    print(f"[DEBUG] 计算能量增益: micro={micro}, emo={emo}, rate={rate}")
    # 微表情基础增益
    base_gain = {
        "冷笑": 0.4,
        "叹息": -1.2,
        "倒吸气": -0.6,
        "哭腔": -0.8
    }.get(micro, 0.0)
    print(f"[DEBUG] 微表情基础增益: {base_gain:.2f}dB")
    # 情绪增益加成
    if emo == "压迫性":
        base_gain += 0.1
    elif emo == "不耐烦":
        base_gain += 0.05
    elif emo == "兴奋":
        base_gain += 0.1

    # 语速增益加成（诈骗场景）
    if rate == "语速加快":
        base_gain += 0.1  # 语速加快时音量稍大
    elif rate == "语速放慢":
        base_gain -= 0.05  # 语速放慢时音量稍小
    elif rate == "不自然停顿":
        base_gain -= 0.1  # 不自然停顿时音量稍小
    elif rate == "忽快忽慢":
        base_gain -= 0.05  # 忽快忽慢时音量稍小

    print(f"[DEBUG] 总能量增益: {base_gain:.2f}dB")
    return base_gain


def build_emo_text(utter: str, intention: str, emo: str, micro: str, rate: str, scene: str = "normal", axis_mode: str = AXIS_MODE_ALL) -> str:
    """
    情绪提示词构建。

    axis_mode:
      - all: 使用 emotion + microexpression + speech_rate_anomaly + intention(CRM)
      - emotion_only: 只使用 emotion
      - micro_only: 只使用 microexpression
      - rate_only: 只使用 speech_rate_anomaly

    设计原则：
      - all 模式保持与旧逻辑一致（避免影响你原有生成结果）
      - 单轴模式下，严格屏蔽其它轴（尤其是 CRM 意图提示）
    """

    axis_mode = normalize_axis_mode(axis_mode)

    # ========== 保持旧逻辑（all 模式完全兼容） ==========
    if axis_mode == AXIS_MODE_ALL:
        semantic_emo_info = SEMANTIC_EMO_MAP.get(emo, {})
        prompt = semantic_emo_info.get("prompt", "")

        # 语速描述映射
        rate_prompt_map = {
            "语速加快": "语速明显加快，带有催促感",
            "语速放慢": "语速刻意放慢，显得犹豫或思考",
            "不自然停顿": "说话时有突兀的停顿，不流畅",
            "忽快忽慢": "语速变化无常，节奏紊乱",
            "正常": "语速适中，没有明显加快或放慢"
        }

        rate_prompt = rate_prompt_map.get(rate, "")

        # 从CRM_PROMPTS获取意图的情绪提示
        crm_emo_prompt = ""
        if intention:
            crm_info = CRM_PROMPTS.get(intention, {})
            if isinstance(crm_info, dict):
                crm_emo_prompt = crm_info.get("情绪提示", "")

        # 构建复合提示
        emo_text = f"表现出{micro}的说话状态，情绪为{emo}"
        if rate_prompt:
            emo_text += f"，{rate_prompt}"
        if crm_emo_prompt:
            emo_text += f"，{crm_emo_prompt}"

        # 加入诈骗场景关键词提示
        urgency_pattern = r"(马上|立刻|赶紧|立即|快)[，。？！…]"
        if re.search(urgency_pattern, utter):
            emo_text += "，带有明显的紧迫感"
        threat_pattern = r"(冻结|查封|逮捕|违法|犯罪)[，。？！…]"
        if re.search(threat_pattern, utter):
            emo_text += "，语气严肃带有威胁性"

        emo_text += f"。{prompt}"

        print(f"[DEBUG] 情绪文本提示: {emo_text}")
        return emo_text

    # ========== 单轴模式（严格屏蔽其它轴） ==========
    use_emo, use_micro, use_rate, use_intention = axis_mode_flags(axis_mode)

    # 语速描述映射（仅 rate_only / all 需要）
    rate_prompt_map = {
        "语速加快": "语速明显加快，带有催促感",
        "语速放慢": "语速刻意放慢，显得犹豫或思考",
        "不自然停顿": "说话时有突兀的停顿，不流畅",
        "忽快忽慢": "语速变化无常，节奏紊乱",
        "正常": ""
    }

    parts = []
    if use_micro and micro and micro != "正常":
        parts.append(f"表现出{micro}的说话状态")
    if use_emo and emo and emo != "正常":
        parts.append(f"情绪为{emo}")
    if use_rate and rate and rate != "正常":
        rp = rate_prompt_map.get(rate, "")
        if rp:
            parts.append(rp)

    # 单轴模式下：明确不使用 CRM 意图提示
    # （即使外部传进来 intention，也强制忽略，避免“意图→情绪/微表情”污染）
    if use_intention:
        crm_info = CRM_PROMPTS.get(intention, {})
        if isinstance(crm_info, dict):
            crm_emo_prompt = crm_info.get("情绪提示", "")
            if crm_emo_prompt:
                parts.append(crm_emo_prompt)

    if not parts:
        parts.append("语气自然")

    emo_text = "，".join(parts)

    # “紧迫感/威胁性”属于情绪类暗示：只在 emotion 轴开启时附加
    if use_emo:
        urgency_pattern = r"(马上|立刻|赶紧|立即|快)[，。？！…]"
        if re.search(urgency_pattern, utter):
            emo_text += "，带有明显的紧迫感"
        threat_pattern = r"(冻结|查封|逮捕|违法|犯罪)[，。？！…]"
        if re.search(threat_pattern, utter):
            emo_text += "，语气严肃带有威胁性"

    # 语义情绪 prompt 也只在 emotion 轴开启时追加
    prompt = ""
    if use_emo:
        semantic_emo_info = SEMANTIC_EMO_MAP.get(emo, {})
        prompt = semantic_emo_info.get("prompt", "")

    if prompt:
        emo_text += f"。{prompt}"
    else:
        emo_text += "。"

    print(f"[DEBUG] 情绪文本提示(axis_mode={axis_mode}): {emo_text}")
    return emo_text

def apply_fraud_keyword_emphasis(text: str) -> str:
    """
    对诈骗关键词进行强化处理
    """
    if re.search(r"\d+\s*(元|块)", text):
        return text

    fraud_keywords = FRAUD_KEYWORDS or {
        "urgency": ["马上", "立刻", "赶紧", "立即", "快"],
        "threat": ["冻结", "查封", "逮捕", "违法", "犯罪"],
        "authority": ["公安局", "检察院", "法院", "银行", "客服"],
        "benefit": ["奖金", "返利", "优惠", "补贴", "补偿"]
    }

    original_text = text

    # 处理紧迫性词汇 - 只在关键词后有标点符号时才处理
    for keyword in fraud_keywords.get("urgency", []):
        # 匹配关键词后跟标点符号的情况
        pattern = fr"({keyword})([，。？！…])"
        if re.search(pattern, text):
            text = re.sub(pattern, r"\1…\2", text)  # 在关键词和标点之间添加省略号

    # 处理权威机构词汇 - 只在关键词后有标点符号时才处理
    for keyword in fraud_keywords.get("authority", []):
        pattern = fr"({keyword})([，。？！…])"
        if re.search(pattern, text):
            text = re.sub(pattern, r"\1…\2", text)  # 在关键词和标点之间添加省略号

    # 处理威胁性词汇 - 只在关键词后有标点符号时才处理
    for keyword in fraud_keywords.get("threat", []):
        pattern = fr"({keyword})([，。？！…])"
        if re.search(pattern, text):
            text = re.sub(pattern, r"\1！\2", text)  # 在关键词和标点之间添加感叹号

    # 限制省略号数量
    text = limit_ellipsis(text)

    if text != original_text:
        print(f"[FRAUD] 诈骗关键词强化: {original_text} -> {text}")

    return text

def apply_conversation_progression(scene: str, turn_idx: int, total_turns: int, base_emo_alpha: float, role: str) -> float:
    """
    基于对话进度的情绪强度调节
    """
    if scene in ["suspicious", "fraud"]:
        progression_factor = min(turn_idx / max(total_turns, 1), 1.0)

        if role == "Agent":
            # 诈骗者：根据场景调整压迫性强度
            if scene == "fraud":
                intensity_boost = 0.20 * progression_factor  # 确定诈骗场景增强更明显
            else:
                intensity_boost = 0.15 * progression_factor  # 涉嫌诈骗场景增强适中
        else:
            # 受害者：根据场景调整恐慌/犹豫强度
            if scene == "fraud":
                intensity_boost = 0.15 * progression_factor  # 确定诈骗场景增强更明显
            else:
                intensity_boost = 0.10 * progression_factor  # 涉嫌诈骗场景增强适中

        adjusted_alpha = min(base_emo_alpha + intensity_boost, 0.75)
        print(f"[PROGRESSION] 对话进度调节: 第{turn_idx}/{total_turns}轮, alpha调整: {base_emo_alpha:.2f} -> {adjusted_alpha:.2f}")
        return adjusted_alpha

    return base_emo_alpha


def build_emo_vector(emo: str, micro: str, rate: str = "正常", scene: str = "normal", axis_mode: str = AXIS_MODE_ALL) -> List[float]:
    """desc --> vector bias 的映射。

    - all 模式保持旧逻辑一致
    - 单轴模式下，按 axis_mode 屏蔽其它轴的向量注入
    """
    axis_mode = normalize_axis_mode(axis_mode)

    # ========== 保持旧逻辑（all 模式完全兼容） ==========
    if axis_mode == AXIS_MODE_ALL:
        print(f"[DEBUG] 构建情绪向量: emo={emo}, micro={micro}, rate={rate}, scene={scene}")
        vec = BASE_EMOTION_MAP.get(emo,[0]*8).copy()
        semantic_emo_info = SEMANTIC_EMO_MAP.get(emo,{})
        desc = semantic_emo_info.get("desc","")
        # 添加场景对情绪向量的调整
        if scene in ["suspicious", "fraud"]:
            if scene == "fraud":
                # 确定诈骗场景：增强负面情绪
                vec[1] += 0.1  # 增强愤怒
                vec[3] += 0.1  # 增强恐惧
            else:
                # 涉嫌诈骗场景：适度增强负面情绪
                vec[1] += 0.05  # 适度增强愤怒
                vec[3] += 0.05  # 适度增强恐惧

            if "愤怒" in desc:
                vec[0] += 0.2
            if "兴奋" in desc:
                vec[1] += 0.1
            if "犹豫" in desc:
                vec[2] += 0.2

        for k,v in MICROEXP_MAP.get(micro,{}).items():
            if k in EMOTION_DIMS:
                vec[EMOTION_DIMS.index(k)] += v
        print(f"[DEBUG] 微表情调整 ({micro}): {MICROEXP_MAP.get(micro, {})}")

        # 应用语速对情绪向量的调整
        rate_adjustment = get_rate_emo_vector_adjustment(rate)
        for i, adj in enumerate(rate_adjustment):
            vec[i] += adj
        print(f"[DEBUG] 语速调整 ({rate}): {rate_adjustment}")

        print(f"[DEBUG] 最终情绪向量: {vec}")
        return vec

    # ========== 单轴模式 ==========
    use_emo, use_micro, use_rate, _ = axis_mode_flags(axis_mode)

    emo_for_vec = emo if use_emo else "正常"
    micro_for_vec = micro if use_micro else "正常"
    rate_for_vec = rate if use_rate else "正常"

    print(f"[DEBUG] 构建情绪向量(axis_mode={axis_mode}): emo={emo_for_vec}, micro={micro_for_vec}, rate={rate_for_vec}, scene={scene}")

    vec = BASE_EMOTION_MAP.get(emo_for_vec, [0]*8).copy()
    semantic_emo_info = SEMANTIC_EMO_MAP.get(emo_for_vec, {})
    desc = semantic_emo_info.get("desc", "")

    # 场景对情绪向量的调整属于“情绪轴”的一部分：只有 emotion 轴开启时才启用
    if use_emo and scene in ["suspicious", "fraud"]:
        if scene == "fraud":
            vec[1] += 0.1
            vec[3] += 0.1
        else:
            vec[1] += 0.05
            vec[3] += 0.05

        if "愤怒" in desc:
            vec[0] += 0.2
        if "兴奋" in desc:
            vec[1] += 0.1
        if "犹豫" in desc:
            vec[2] += 0.2

    # 微表情注入
    if use_micro:
        for k, v in MICROEXP_MAP.get(micro_for_vec, {}).items():
            if k in EMOTION_DIMS:
                vec[EMOTION_DIMS.index(k)] += v
        print(f"[DEBUG] 微表情调整(axis_mode={axis_mode}, micro={micro_for_vec}): {MICROEXP_MAP.get(micro_for_vec, {})}")

    # 语速对情绪向量的调整
    if use_rate:
        rate_adjustment = get_rate_emo_vector_adjustment(rate_for_vec)
        for i, adj in enumerate(rate_adjustment):
            vec[i] += adj
        print(f"[DEBUG] 语速调整(axis_mode={axis_mode}, rate={rate_for_vec}): {rate_adjustment}")

    print(f"[DEBUG] 最终情绪向量(axis_mode={axis_mode}): {vec}")
    return vec

def apply_scene_behavior(
    scene: str,
    role: str,
    base_emo: str,
    micro: str,
    rate: str
) -> Tuple[str, float]:
    """
    scene: normal / suspicious / fraud
    role: Agent / User
    返回：emo, emo_alpha（只增强，不替换）
    """
    # 修复场景名称映射
    scene_mapping = {
        "normal": "正常通话",
        "suspicious": "涉嫌诈骗",
        "fraud": "确定诈骗"
    }
    Chinese_scene = scene_mapping.get(scene, "正常通话")
    profile = SCENE_BEHAVIOR_PROFILE.get(Chinese_scene, {}).get(role, {})

    # 1️⃣ 基础 alpha
    emo_alpha = profile.get(
        "emotion_alpha_base",
        EMO_ALPHA_MAP.get(base_emo, 0.35)
    )

    # 2️⃣ 情绪增强（尊重原始标签）
    emo_boost = profile.get("emotion_boost", {}).get(base_emo, 0.0)
    emo_alpha += emo_boost

    # 3️⃣ 语速对情绪强度的影响（不是改语速）
    rate_bias = profile.get("speech_rate_bias", {}).get(rate, 0.0)
    emo_alpha += rate_bias

    # 4️⃣ 微表情轻微放大（非常克制）
    micro_bias = profile.get("micro_boost", {}).get(micro, 0.0)
    emo_alpha += micro_bias

    # 5️⃣ 安全裁剪
    emo_alpha = max(0.25, min(emo_alpha, 0.75))

    print(
        f"[SCENE] {scene} | {role} | emo={base_emo} "
        f"| alpha={emo_alpha:.2f}"
    )

    return base_emo, emo_alpha

def apply_scene_specific_text_process(text: str, scene: str) -> str:
    """
    应用场景特定的文本处理规则
    """
    if scene in ["suspicious", "fraud"]:
        # 诈骗场景：增强警示性词汇
        warning_patterns = [
            (r"系统", r"【系统】"),
            (r"安全", r"【安全】"),
            (r"验证", r"【验证】")
        ]
        for pattern, replace in warning_patterns:
            text = re.sub(pattern, replace, text)

    return text
#====================== JSON =======================
def extract_conversation(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "content" in obj and isinstance(obj["content"],dict):
        return obj["content"].get("conversation",[])
    if "conversation" in obj:
        if isinstance(obj["conversation"],dict):
            return obj["conversation"].get("conversation",[])
        return obj["conversation"]
    return []

def extract_user_profile(obj: dict) -> str:
    if "user_profile" in obj:
        return obj["user_profile"]
    if "content" in obj:
        return obj["content"].get("user_profile","")
    return ""
#====================== 选择 =======================
def parse_line_selector(expr: str) -> Set[int]:
    if not expr:
        return set()
    result = set()
    for part in expr.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            result.update(range(int(a),int(b)+1))
        else:
            result.add(int(part))
    return result
# ===================== 后处理 =====================
def simple_postprocess(inp: str, outp: str, micro: str, emo: str, rate: str) -> bool:
    try:
        from pydub import AudioSegment
        #确保音频文件存在且可读
        if not os.path.exists(inp):
            print(f"[ERROR]输入音频不存在：{inp}")
            return False
        print(f"[DEBUG] 读取音频文件")
        audio = AudioSegment.from_wav(inp)
        print(f"[DEBUG] 音频时长: {len(audio)}ms")

        # 不要顶满，留足余量（建议 3~6dB）
        audio = audio.normalize(headroom=6.0)

        gain = calculate_energy_gain(micro, emo, rate)
        if gain:
            target_peak = -1.0  # 峰值目标：-1dBFS（很稳）
            peak_after = audio.max_dBFS + gain
            if peak_after > target_peak:
                gain = target_peak - audio.max_dBFS
            audio = audio.apply_gain(gain)

        # 根据语速调整淡入淡出（使用统一配置）
        fade_duration = get_rate_fade_duration(rate)
        audio = audio.fade_in(fade_duration).fade_out(fade_duration)

        os.makedirs(os.path.dirname(outp), exist_ok=True)
        audio.export(outp, format="wav")
        return True

    except Exception as e:
        print(f"[ERROR]后处理失败：{e}")
        import traceback
        traceback.print_exc()
        try:
            shutil.copy2(inp, outp)
            return True
        except Exception as copy_error:
            print(f"[ERROR] 复制文件失败：{copy_error}")
            return False
# ===================== 单轮 =====================
def synthesize_turn(
    tts: Any,
    turn: Dict[str, Any],
    scene: str,
    speaker_map: Dict[str, Any],
    gender: str,
    out_dir: str,
    call_dir: str,
    idx: int,
    total_turns: int,
    axis_mode: str = AXIS_MODE_ALL
) -> Optional[str]:
    keep_intermediates = False
    try:
        utter_raw = turn.get("utterance","").strip()
        base_emo_raw = turn.get("emotion","正常")
        micro_raw = turn.get("microexpression","正常")
        rate_raw = turn.get("speech_rate_anomaly","正常")
        intention_raw = turn.get("intention","")

        axis_mode = normalize_axis_mode(axis_mode)
        base_emo, micro, rate, intention = masked_labels(
            base_emo=base_emo_raw,
            micro=micro_raw,
            rate=rate_raw,
            intention=intention_raw,
            axis_mode=axis_mode
        )
        role = "Agent" if turn.get("speaker","").lower() in ["agent","客服"] else "User"
        speaker = "Agent" if role=="Agent" else f"User_{gender}"

        print(f"\n[处理第{idx}/{total_turns}轮] {role}：{utter_raw}")

        # 场景行为调节
        emo, emo_alpha = apply_scene_behavior(
            scene=scene,
            role=role,
            base_emo=base_emo,
            micro=micro,
            rate=rate
        )

        # 对话进度调节（仅诈骗场景）
        if axis_mode == AXIS_MODE_ALL:
            emo_alpha = apply_conversation_progression(
                scene=scene,
                turn_idx=idx,
                total_turns=total_turns,
                base_emo_alpha=emo_alpha,
                role=role
            )

        # ===== 情绪强度安全阀（听感校准） =====
        if emo == "犹豫":
            # 犹豫不允许过强，否则像演戏
            emo_alpha = min(emo_alpha, 0.42)

        elif emo == "不耐烦":
            emo_alpha = min(emo_alpha, 0.62)

        elif emo == "愤怒":
            # 压抑愤怒，防止破音
            emo_alpha = min(emo_alpha, 0.50)

        # ===== 文本处理 =====
        print(f"\n[DEBUG] 步骤1: 文本处理")
        utter = normalize_text_for_tts(utter_raw)
        utter = apply_text_process(utter)
        utter = apply_emotion_rhythm_template(utter, emo)
        if axis_mode == AXIS_MODE_ALL:
            utter = apply_scene_specific_text_process(utter, scene)
            if scene in ["suspicious", "fraud"]:
                utter = apply_fraud_keyword_emphasis(utter)

        # 语速韵律标记处理
        utter = enhance_text_with_rhythm_marks(utter, rate)

        print(f"[DEBUG] 最终处理文本: {utter}")

        # 用纯ASCII短名，避免中文标点/超长文件名导致写出的wav损坏
        uid = uuid.uuid4().hex[:8]
        role_tag = "agent" if role == "Agent" else "user"
        text_tag = safe_filename_readable(utter_raw, max_len=24)
        base = f"turn{idx:02d}_{role_tag}_{text_tag}_{uid}"
        raw = os.path.join(out_dir, f"{base}_raw.wav")
        out = os.path.join(out_dir, f"{base}.wav")

        # ===== TTS 合成 =====
        print(f"\n[DEBUG] 步骤2: TTS合成")

        # 获取语速配置
        rate_config = get_rate_config(rate)
        # ===== 犹豫语速策略修正 =====
        # 该规则属于"多轴联合策略"，单轴导出时必须禁用，否则会让 emotion_only/micro_only/rate_only 互相污染
        if axis_mode == AXIS_MODE_ALL and emo == "犹豫" and intention in ["理清思路", "信息核实", "寻找借口"]:
            rate = "不自然停顿"

        max_len = rate_config.get("max_len", 28)

        chunks = split_text_for_semantic_safety(
            utter,
            max_len=max_len,
            rate=rate
        )
        audio_all = AudioSegment.empty()
        print(f"[DEBUG] 开始处理 {len(chunks)} 个文本块")

        for j, sub in enumerate(chunks):
            tmp = os.path.join(out_dir, f"{base}_chunk{j}.wav")

            # 根据语速调整停顿（使用统一配置）
            pause_params = get_rate_pause_params(rate, "before_chunk")
            pause_before = 0
            if j > 0:
                if random.random() < pause_params["long_pause_prob"]:
                    pause_before = pause_params["long_pause"]
                else:
                    pause_before = pause_params["short_pause"]

            if pause_before > 0:
                audio_all += AudioSegment.silent(duration=pause_before)

            tts.infer(
                spk_audio_prompt=speaker_map[speaker],
                text=sub,
                output_path=tmp,
                use_emo_text=True,
                emo_text=build_emo_text(utter_raw, intention, emo, micro, rate, scene, axis_mode=axis_mode),
                emo_vector=build_emo_vector(emo, micro, rate, scene, axis_mode=axis_mode),
                emo_alpha=emo_alpha
            )

            if os.path.exists(tmp):
                # 校验WAV头：坏文件就跳过，避免整个流程崩
                if (os.path.getsize(tmp) < 100) or (not is_valid_wav_header(tmp)):
                    print(f"...")
                    if not keep_intermediates:
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                    continue

                try:
                    chunk_segment = AudioSegment.from_wav(tmp)
                    audio_all += chunk_segment
                finally:
                    if not keep_intermediates:
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass

            else:
                print(f"[ERROR] chunk wav not found: {tmp}")
                continue

            if j < len(chunks) - 1:
                pause_params = get_rate_pause_params(rate, "between_chunks")
                gap = pause_params["long_pause"] if random.random() < pause_params["long_pause_prob"] else pause_params[
                    "short_pause"]
                audio_all += AudioSegment.silent(duration=gap)

        audio_all.export(raw, format="wav")
        print(f"[DEBUG] 原始音频保存成功，大小: {os.path.getsize(raw)} bytes")

        # ===== 语速调整 =====
        print(f"\n[DEBUG] 步骤3: 语速调整")
        speed_adjusted = raw.replace("_raw.wav", "_speed.wav")
        adjust_audio_speed(raw, speed_adjusted, rate)
        print(f"[DEBUG] 语速调整完成，文件大小: {os.path.getsize(speed_adjusted)} bytes")

        # ===== 轻量后处理 =====
        print(f"\n[DEBUG] 步骤4: 轻量后处理")
        pp_ok = simple_postprocess(inp=speed_adjusted, outp=out, micro=micro, emo=emo, rate=rate)

        # 如果后处理失败或 out 没生成，至少保证有一个可用 wav（回退到 speed/raw）
        if (not pp_ok) or (not os.path.exists(out)):
            print(f"[WARN] 后处理未生成最终文件，fallback: copy speed/raw -> out")
            try:
                shutil.copy2(speed_adjusted if os.path.exists(speed_adjusted) else raw, out)
            except Exception as e:
                print(f"[ERROR] fallback copy failed: {e}")

        if os.path.exists(out):
            print(f"[DEBUG] 最终输出文件大小: {os.path.getsize(out)} bytes")
            if not keep_intermediates:
                for p in [raw, speed_adjusted]:
                    try:
                        if p and os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
            return out

        print(f"[WARN] 最终 out 仍不存在，返回 raw: {raw}")
        return raw if os.path.exists(raw) else None

    except Exception as e:
        print(f"[ERROR] 第{idx}轮失败：{e}")
        import traceback
        traceback.print_exc()
        return None

# ===================== main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--speaker_map",required=True)
    ap.add_argument("--cfg",required=True)
    ap.add_argument("--model_dir",required=True)
    ap.add_argument("--configs_dir",required=True)
    ap.add_argument("--out_root",required=True)
    ap.add_argument("--merge",action="store_true")
    ap.add_argument("--lines", type=str, default="", help="Select lines to proceess,e.g. 5| 5,8,10 | 5-12")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root,"run_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir,exist_ok=True)

    logger = logging.getLogger("ALL")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(run_dir, "run_main.log"), encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(fh)
    logger.addHandler(sh)

    sys.stdout = TeeLogger(logger)
    sys.stderr = TeeLogger(logger, logging.ERROR)

    load_configs(args.configs_dir)
    speaker_map = json.load(open(args.speaker_map,"r",encoding="utf-8"))
    tts = IndexTTS2(args.cfg,args.model_dir,use_fp16=False)
    line_selector = parse_line_selector(args.lines)

    for i,line in enumerate(open(args.input,"r",encoding="utf-8"),1):
        if line_selector and i not in line_selector:
            continue
        obj = json.loads(line)
        conv = extract_conversation(obj)
        profile = extract_user_profile(obj)
        final_label = obj.get("content", {}).get("final_label", "")
        scene = map_scene(final_label)
        gender = "female" if "女" in profile else "male"

        call_id = str(obj.get("call_id",f"line{i}"))
        name_match = re.search(r"姓名[:：]\s*([^；;,]+)", profile)
        name = name_match.group(1).strip() if name_match else "unknown"
        name = re.sub(r"[\\/*?:\"<>|]", "_", name)
        call_dir = os.path.join(run_dir,f"{call_id}_{name}")
        os.makedirs(call_dir,exist_ok=True)

        raw_input_path = os.path.join(call_dir,"raw_input.jsonl")
        with open(raw_input_path,"a",encoding="utf-8") as f:
            f.write(line.rstrip()+"\n")

        parsed_path = os.path.join(call_dir, f"{call_id}_{name}.json")
        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

        handler = add_local_log_handler(logger, os.path.join(call_dir, "synthesis.log"))
        try:
            turns_dir = os.path.join(call_dir, "turns")
            os.makedirs(turns_dir, exist_ok=True)

            wavs = []
            for idx, turn in enumerate(conv, start=1):
                w = synthesize_turn(tts, turn,
                                    scene=scene,
                                    speaker_map=speaker_map,
                                    gender=gender,
                                    out_dir=turns_dir,
                                    call_dir=call_dir,
                                    idx=idx,
                                    total_turns=len(conv))
                if w:
                    wavs.append(w)

            valid_wavs = []
            for w in wavs:
                try:
                    a = AudioSegment.from_wav(w)
                    if len(a) >= 300:
                        valid_wavs.append(a)
                    else:
                        print(f"[SKIP] 音频过短：{w}")
                except Exception:
                    print(f"[SKIP] 无法读取音频：{w}")

            if args.merge and valid_wavs:
                comb = AudioSegment.silent(300)
                for a in valid_wavs:
                    comb += a + AudioSegment.silent(400)

                merged_path = os.path.join(call_dir, f"merged-{call_id}_{name}.wav")
                comb.export(merged_path, format="wav")
                print(f"[MERGED] 合并完成: {merged_path}")
                print(f"[MERGED] 总时长: {len(comb)}ms")
                print(f"[MERGED] 文件大小: {os.path.getsize(merged_path)} bytes")
                print(f"[MERGED] {merged_path}")
            else:
                print("[INFO] 无可合并音频")

        finally:
            logger.removeHandler(handler)
            handler.close()
            print(f"[DEBUG] 清理日志处理器")

    print(f"ALL DONE!所有文件都保存在了{run_dir}里面。")

if __name__ == "__main__":
    main()
