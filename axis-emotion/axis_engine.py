#axis_engine.py
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import traceback
import unicodedata
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pydub import AudioSegment

# ---- PATCH: avoid torchaudio sox backend segfault when saving wav ----
def _patch_torchaudio_save() -> None:
    """用 soundfile 写 PCM16，规避 torchaudio 在部分 sox backend 上的崩溃。"""
    try:
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

            # torchaudio: [C, T] ; soundfile: [T, C]
            if x.ndim == 2 and channels_first:
                x = x.T
            elif x.ndim not in (1, 2):
                x = x.reshape(-1)

            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            mx = float(np.max(np.abs(x))) if x.size else 0.0
            if mx > 1.0:
                x = x / mx * 0.95

            sf.write(filepath, x, int(sample_rate), subtype="PCM_16")

        torchaudio.save = _safe_save
        print("[PATCH] torchaudio.save -> soundfile.write (PCM_16)")

    except Exception as e:
        # patch 失败不影响流程
        print("[PATCH] torchaudio.save patch skipped:", repr(e))

_patch_torchaudio_save()

# IndexTTS
from indextts.infer_v2 import IndexTTS2

# ===================== 全局配置 =====================
EMOTION_DIMS = ["happy","angry","sad","afraid","disgusted","melancholic","surprised","calm"]

CRM_PROMPTS: Dict[str, Any] = {}
BASE_EMOTION_MAP: Dict[str, List[float]] = {}
SEMANTIC_EMO_MAP: Dict[str, Any] = {}
TEXT_PROCESS_CFG: Dict[str, Any] = {}
EMOTION_RHYTHM_TEMPLATES: Dict[str, Any] = {}
SCENE_BEHAVIOR_PROFILE: Dict[str, Any] = {}
FRAUD_KEYWORDS: Dict[str, Any] = {}
EMO_ALPHA_MAP: Dict[str, float] = {}
SPEECH_RATE_CONFIG: Dict[str, Any] = {}
SPEAKER_MAP: Dict[str, Any] = {}

# 调试开关（可在 YAML 的 debug 里打开）
DEBUG_RHYTHM_TEMPLATES: bool = False

# 可选：把后处理参数放 YAML（没配就用默认）
POSTPROCESS_AUDIO_CFG: Dict[str, Any] = {}

# ===================== 单轴导出 / Axis mask =====================
AXIS_MODE_ALL = "all"
AXIS_MODE_EMOTION_ONLY = "emotion_only"
AXIS_MODE_RATE_ONLY = "rate_only"

AXIS_MODES = {AXIS_MODE_ALL, AXIS_MODE_EMOTION_ONLY, AXIS_MODE_RATE_ONLY}

def normalize_axis_mode(axis_mode: Optional[str]) -> str:
    if not axis_mode:
        return AXIS_MODE_ALL
    m = str(axis_mode).strip().lower()
    if m not in AXIS_MODES:
        print(f"[WARN] Unknown axis_mode={axis_mode!r}, fallback to '{AXIS_MODE_ALL}'")
        return AXIS_MODE_ALL
    return m

def axis_mode_flags(axis_mode: Optional[str]) -> Tuple[bool, bool, bool]:
    """Return (use_emo, use_rate, use_intention).

    NOTE: intention 轴已弃用（为了让 emotion / speech_rate 两个轴更“纯”）。
    因此这里始终返回 use_intention=False。
    """
    axis_mode = normalize_axis_mode(axis_mode)
    use_emo = axis_mode in (AXIS_MODE_ALL, AXIS_MODE_EMOTION_ONLY)
    use_rate = axis_mode in (AXIS_MODE_ALL, AXIS_MODE_RATE_ONLY)
    use_intention = False
    return use_emo, use_rate, use_intention

def masked_labels(
    base_emo: str,
    rate: str,
    intention: str,
    axis_mode: Optional[str]
) -> Tuple[str, str, str]:
    """根据 axis_mode 输出屏蔽后的标签。

    - emotion_only: rate/intention 置为“正常”/空
    - rate_only: emotion/intention 置为“正常”/空
    - all: emotion/rate 保留；intention 仍会被置空（已弃用）
    """
    use_emo, use_rate, _ = axis_mode_flags(axis_mode)
    emo_out = base_emo if use_emo else "正常"
    rate_out = rate if use_rate else "正常"
    intention_out = ""  # intention 已弃用
    return emo_out, rate_out, intention_out

# ===================== 配置加载 =====================
# ===================== 配置加载 =====================
def load_configs(cfg_dir: str) -> None:
    """加载配置。

    优先读取 configs_dir/config.yaml。

    同时兼容两种 YAML 结构：

    1) 旧版（扁平顶层键）：
        - base_emotions / emotion_alpha_map / semantic_emotions / emotion_rhythm_templates
        - speech_rate_config / text_process_config / speaker_map / postprocess_audio
        - （以及旧项目里其它键）

    2) 紧凑版（分组键，推荐）：
        emotion:
          base: ...
          alpha: ...
          semantic: ...
          rhythm_templates: ...
        speech_rate:
          speed_factor: { 正常: 1.0, 语速加快: 1.02, ... }   # 仅保留这一项
        text_process: ...
        speaker_map: ...
        postprocess_audio: ...
        debug:
          emotion_rhythm_templates: true|false
    """

    def _load_yaml(path: str) -> Dict[str, Any]:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("Missing dependency: pyyaml. Install via pip install pyyaml") from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    global CRM_PROMPTS, BASE_EMOTION_MAP
    global SEMANTIC_EMO_MAP, TEXT_PROCESS_CFG
    global EMOTION_RHYTHM_TEMPLATES, SCENE_BEHAVIOR_PROFILE, FRAUD_KEYWORDS
    global EMO_ALPHA_MAP, SPEECH_RATE_CONFIG, SPEAKER_MAP
    global POSTPROCESS_AUDIO_CFG, DEBUG_RHYTHM_TEMPLATES

    yaml_path = os.path.join(cfg_dir, "config.yaml")
    if not os.path.exists(yaml_path):
        print(f"[CFG] config.yaml not found in: {cfg_dir}")
        return

    cfg = _load_yaml(yaml_path)

    # -------- debug flags --------
    dbg = cfg.get("debug", {}) or {}
    DEBUG_RHYTHM_TEMPLATES = bool(
        dbg.get("emotion_rhythm_templates", dbg.get("rhythm_templates", False))
    )

    # -------- emotion (compact) --------
    emo_cfg = cfg.get("emotion", {}) or {}
    BASE_EMOTION_MAP = (emo_cfg.get("base") or cfg.get("base_emotions") or {}) or {}
    EMO_ALPHA_MAP = (emo_cfg.get("alpha") or cfg.get("emotion_alpha_map") or {}) or {}
    SEMANTIC_EMO_MAP = (emo_cfg.get("semantic") or cfg.get("semantic_emotions") or {}) or {}
    EMOTION_RHYTHM_TEMPLATES = (
        emo_cfg.get("rhythm_templates")
        or emo_cfg.get("rhythm")
        or cfg.get("emotion_rhythm_templates")
        or {}
    ) or {}

    # -------- speech rate (compact) --------
    rate_cfg = cfg.get("speech_rate", {}) or {}
    SPEECH_RATE_CONFIG = (
        rate_cfg.get("speed_factor")
        or rate_cfg.get("config")
        or cfg.get("speech_rate_config")
        or {}
    ) or {}

    # -------- text process (compact) --------
    TEXT_PROCESS_CFG = (cfg.get("text_process") or cfg.get("text_process_config") or {}) or {}

    # -------- legacy / optional sections (现在不再影响 emotion 轴) --------
    CRM_PROMPTS = cfg.get("crm_prompts", {}) or {}
    FRAUD_KEYWORDS = cfg.get("fraud_keywords", {}) or {}
    SCENE_BEHAVIOR_PROFILE = cfg.get("scene_behavior_profile", {}) or {}

    # -------- misc --------
    SPEAKER_MAP = cfg.get("speaker_map", {}) or {}
    POSTPROCESS_AUDIO_CFG = cfg.get("postprocess_audio", {}) or {}

    print(f"[CFG] Loaded YAML: {yaml_path}")
    print(f"[CFG] speaker_map_keys={len(SPEAKER_MAP)}")
    if DEBUG_RHYTHM_TEMPLATES:
        print("[CFG] DEBUG_RHYTHM_TEMPLATES=ON")
#============== 小工具 ==================
class TeeLogger:
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, msg: str) -> None:
        msg = msg.rstrip()
        if msg:
            self.logger.log(self.level, msg)

    def flush(self) -> None:
        pass

def add_local_log_handler(logger: logging.Logger, log_path: str) -> logging.Handler:
    h = logging.FileHandler(log_path, encoding="utf-8")
    h.setLevel(logging.INFO)
    logger.addHandler(h)
    return h

def parse_line_selector(expr: str) -> Set[int]:
    if not expr:
        return set()
    result: Set[int] = set()
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    return result

def safe_name(s: str, max_len: int = 80) -> str:
    s = (s or "unknown").strip() or "unknown"
    s = re.sub(r"[\\/*?:\"<>|]", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:max_len]

def safe_filename_readable(s: str, max_len: int = 24) -> str:
    """可读文件名片段（允许中文），同时避免常见非法字符。"""
    if s is None:
        return "utt"
    s = str(s).strip()
    if not s:
        return "utt"

    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", s)
    s = s.replace("…", "_").replace("—", "_")
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9 _\.-]", "_", s)
    s = re.sub(r"_+", "_", s).strip(" _.-")
    if len(s) > max_len:
        s = s[:max_len].rstrip(" _.-")
    return s if s else "utt"

def is_valid_wav_header(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(12)
        return len(head) == 12 and head[0:4] == b"RIFF" and head[8:12] == b"WAVE"
    except Exception:
        return False

# =================== 场景 =================
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

# ===================== 语速配置 =====================
# ===================== 语速配置 =====================
def get_rate_config(rate: str) -> Dict[str, Any]:
    """返回某个语速标签的配置。

    兼容两种写法：
      - 旧版：speech_rate_config: { 语速加快: {speed_factor: 1.02, ...}, ... }
      - 紧凑版：speech_rate: { speed_factor: { 语速加快: 1.02, ... } }
        这里会把 float 自动包成 {"speed_factor": float}。
    """
    if not SPEECH_RATE_CONFIG:
        return {}

    v = SPEECH_RATE_CONFIG.get(rate, SPEECH_RATE_CONFIG.get("正常", {}))
    if isinstance(v, (int, float)):
        return {"speed_factor": float(v)}
    if isinstance(v, dict):
        return v
    return {}

def get_rate_speed_factor(rate: str) -> float:
    return float(get_rate_config(rate).get("speed_factor", 1.0))

# NOTE: 为了让 emotion 轴更纯粹（只由 base_emotions + emotion_alpha_map + semantic_emotions + rhythm_templates 控制）
#       这里不再允许语速去“顺带”改情绪向量。
def get_rate_emo_vector_adjustment(rate: str) -> List[float]:
    return [0.0] * 8

# 下面这些参数仍然保留（主要用于文本分块拼接的停顿、fade），
# 但在“紧凑版 speech_rate 配置”下不会再从 YAML 中被显式配置——缺失时统一走默认值，
# 这样 speech_rate_config 就可以只保留 speed_factor 一个参数。
DEFAULT_PAUSE_CFG = {
    "short_pause": 60,
    "long_pause": 120,
    "long_pause_prob": 0.08,
}

def get_rate_pause_params(rate: str, position: str = "between_chunks") -> Dict[str, Any]:
    cfg = get_rate_config(rate)
    # 目前 before/between 参数一致，保留 position 以便以后扩展
    return {
        "short_pause": int(cfg.get("short_pause", DEFAULT_PAUSE_CFG["short_pause"])),
        "long_pause": int(cfg.get("long_pause", DEFAULT_PAUSE_CFG["long_pause"])),
        "long_pause_prob": float(cfg.get("long_pause_prob", DEFAULT_PAUSE_CFG["long_pause_prob"])),
    }

def get_rate_fade_duration(rate: str) -> int:
    cfg = get_rate_config(rate)
    return int(cfg.get("fade_duration", 40))
def adjust_audio_speed(input_wav: str, output_wav: str, rate: str) -> bool:
    """pydub 变速（会改变音高）。失败则回退复制。"""
    try:
        speed_factor = get_rate_speed_factor(rate)
        if speed_factor == 1.0:
            shutil.copy2(input_wav, output_wav)
            return True

        audio = AudioSegment.from_wav(input_wav)
        new_frame_rate = int(audio.frame_rate * speed_factor)
        audio_speed_adjusted = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate}).set_frame_rate(
            audio.frame_rate
        )
        audio_speed_adjusted.export(output_wav, format="wav")
        return True

    except Exception as e:
        print(f"[ERROR] 音频速度调整失败：{e}")
        try:
            shutil.copy2(input_wav, output_wav)
            return True
        except Exception:
            return False

# =============== 文本处理 ======================
DIGIT_CN = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}

def number_to_chinese(num_str: str) -> str:
    try:
        num = int(num_str)
        if num == 0:
            return "零"

        digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        units = ["", "十", "百", "千", "万", "十", "百", "千", "亿"]

        result = []
        s = str(num)
        length = len(s)

        for i, d in enumerate(s):
            di = int(d)
            if di == 0:
                if i < length - 1 and s[i + 1] != "0":
                    result.append(digits[0])
            else:
                result.append(digits[di])
                if i < length - 1:
                    unit_pos = length - i - 1
                    result.append(units[unit_pos])

        cn = "".join(result)
        if 10 <= num < 20:
            cn = cn.replace("一十", "十")
        return cn
    except Exception:
        return num_str

def limit_ellipsis(text: str) -> str:
    """限制省略号数量，最多保留两个连续的“…”（即“……”）。"""
    # 3 个以上统一压成 2 个，避免出现“………………”导致模型出现怪异喘气/叹息
    return re.sub(r"…{3,}", "……", text)

def normalize_text_for_tts(text: str) -> str:
    """稳定文本标准化（保留你原逻辑）。"""
    text = re.sub(r"\s+", " ", (text or "").strip())

    # 1) 金额锁定
    amount_pattern = re.compile(r"\d+(?:\.\d+)?\s*(?:元|块)")
    amount_slots: List[str] = []

    def _lock_amount(m: re.Match) -> str:
        key = f"__AMOUNT_{len(amount_slots)}__"
        amount_slots.append(m.group(0))
        return key

    text = amount_pattern.sub(_lock_amount, text)

    # 2) 抖音ID/账号规则
    if any(k in text for k in ["抖音ID", "抖音号", "抖音账号"]):
        text = text.replace("_", "下划线")
        text = re.sub(r"\b([A-Z])(?:-([A-Z])){1,}\b", lambda m: "……".join(m.group(0).split("-")), text)
        text = re.sub(r"([a-zA-Z])-([a-zA-Z])", r"\1杠\2", text)

    # 3) 卡号/账号/验证码逐位读
    def _read_digits(m: re.Match) -> str:
        return " ".join(DIGIT_CN.get(c, c) for c in m.group(0))

    if any(k in text for k in ["卡号", "银行卡", "账号", "账户", "收款人", "验证码", "认证代码"]):
        text = re.sub(r"\d{4,}", _read_digits, text)
        text = re.sub(r"\.{3,}", "……", text)

    # 4) 时间单位数字转中文
    def _time_cn(m: re.Match) -> str:
        return number_to_chinese(m.group(1)) + m.group(2)

    text = re.sub(r"(\d+)(秒|分|小时|天|周|月|年)", _time_cn, text)

    # 5) 金额恢复
    for i, amt in enumerate(amount_slots):
        text = text.replace(f"__AMOUNT_{i}__", amt)

    return text

def apply_text_process(text: str) -> str:
    cfg = TEXT_PROCESS_CFG or {}
    if not cfg:
        return text

    original = text

    ex = (cfg.get("punctuation", {}) or {}).get("exclamation", {})
    if ex:
        text = re.sub(ex.get("pattern", ""), ex.get("replace", ""), text)

    q = (cfg.get("punctuation", {}) or {}).get("question", {})
    if q:
        text = re.sub(q.get("pattern", ""), q.get("replace", ""), text)

    job = cfg.get("job_id")
    if isinstance(job, dict) and "pattern" in job:

        def _job(m: re.Match) -> str:
            return m.group(1) + " " + " ".join(m.group(2))

        text = re.sub(job["pattern"], _job, text)

    return text if text != original else original

def apply_emotion_rhythm_template(text: str, emo: str) -> str:
    """按 emotion_rhythm_templates 对文本做轻量“韵律化”改写。

    你遇到的“写了模板但音频没体现”，常见原因有两类：
    1) 规则根本没命中（文本里没有对应标点/关键词）
    2) 命中了，但模型对某些符号（尤其省略号长度）不敏感，听感差异很小

    所以这里增加了一个可选的 debug 开关：
      在 config.yaml 的 debug.emotion_rhythm_templates=true 时，
      会打印每条 rule 的命中情况（subn 次数）以及改写前后文本。
    """
    cfg = (EMOTION_RHYTHM_TEMPLATES or {}).get(emo)
    if not cfg:
        if DEBUG_RHYTHM_TEMPLATES:
            print(f"[RHYTHM] emo={emo} -> no template")
        return text

    rules = cfg.get("rules", []) or []
    if not rules:
        return text

    before = text
    applied: List[Dict[str, Any]] = []

    for i, rule in enumerate(rules):
        try:
            prob = float(rule.get("prob", 1.0))
        except Exception:
            prob = 1.0
        if prob < 1.0 and random.random() > prob:
            continue

        pattern = str(rule.get("pattern", "") or "")
        replace = str(rule.get("replace", "") or "")
        desc = str(rule.get("desc", "") or "")

        if not pattern:
            continue

        # count=0 表示替换全部
        try:
            count = int(rule.get("count", 0) or 0)
        except Exception:
            count = 0

        try:
            if count > 0:
                new_text, n = re.subn(pattern, replace, text, count=count)
            else:
                new_text, n = re.subn(pattern, replace, text)
        except re.error as e:
            # 配置写错正则时，不要整个崩
            if DEBUG_RHYTHM_TEMPLATES:
                print(f"[RHYTHM] emo={emo} rule#{i} regex error: {e} | pattern={pattern}")
            continue

        if n > 0:
            applied.append({"i": i, "n": n, "pattern": pattern, "replace": replace, "desc": desc})
            text = new_text

    if DEBUG_RHYTHM_TEMPLATES and (text != before or applied):
        print(f"[RHYTHM] emo={emo} applied={len(applied)}")
        if text != before:
            print(f"  before: {before}")
            print(f"  after : {text}")
        for a in applied:
            dd = f" desc={a['desc']}" if a.get("desc") else ""
            print(f"  - rule#{a['i']} n={a['n']} pattern={a['pattern']} replace={a['replace']}{dd}")

    return text

def apply_scene_specific_text_process(text: str, scene: str) -> str:
    if scene in ("suspicious", "fraud"):
        for pattern, repl in [(r"系统", r"【系统】"), (r"安全", r"【安全】"), (r"验证", r"【验证】")]:
            text = re.sub(pattern, repl, text)
    return text

def apply_fraud_keyword_emphasis(text: str) -> str:
    # 金额句子不动
    if re.search(r"\d+\s*(元|块)", text):
        return text

    fraud_keywords = FRAUD_KEYWORDS or {
        "urgency": ["马上", "立刻", "赶紧", "立即", "快"],
        "threat": ["冻结", "查封", "逮捕", "违法", "犯罪"],
        "authority": ["公安局", "检察院", "法院", "银行", "客服"],
        "benefit": ["奖金", "返利", "优惠", "补贴", "补偿"],
    }

    # urgency/authority：keyword + 标点 -> keyword…标点
    for kw in fraud_keywords.get("urgency", []) or []:
        text = re.sub(fr"({re.escape(kw)})([，。？！…])", r"\1…\2", text)
    for kw in fraud_keywords.get("authority", []) or []:
        text = re.sub(fr"({re.escape(kw)})([，。？！…])", r"\1…\2", text)

    # threat：keyword + 标点 -> keyword！标点
    for kw in fraud_keywords.get("threat", []) or []:
        text = re.sub(fr"({re.escape(kw)})([，。？！…])", r"\1！\2", text)

    return limit_ellipsis(text)

def enhance_text_with_rhythm_marks(text: str, rate: str) -> str:
    # 目前保留你的“轻处理”策略（不在这里硬塞省略号）
    return limit_ellipsis(text)

def split_text_for_semantic_safety(text: str, max_len: int = 28, rate: str = "正常") -> List[str]:
    """按自然边界切块，避免一次合成太长导致语义/韵律问题。"""
    cfg = get_rate_config(rate)
    adjusted_max_len = int(cfg.get("max_len", max_len))
    pause_chars = cfg.get("pause_chars", ["，", "。"])

    chunks: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in pause_chars and len(buf) >= adjusted_max_len:
            has_text = any((c not in pause_chars) and (not c.isspace()) for c in buf)
            if has_text:
                chunks.append(buf)
                buf = ""
    if buf.strip():
        chunks.append(buf)
    return chunks

# ================= 情绪/微表情/语速 -> 提示词 + 向量 ============================

def calculate_energy_gain(emo: str, rate: str) -> float:
    """轻量后处理增益（dB）。

    为了让单轴更“干净”（尤其是 speech_rate 只看 speed_factor 的效果），
    这里默认不再基于 emotion / rate 额外改增益。
    如需恢复“根据情绪/语速微调响度”的老策略，可以把下面直接改回去。
    """
    return 0.0

def build_emo_text(
    utter: str,
    emo: str,
    rate: str,
    axis_mode: str = AXIS_MODE_ALL,
) -> str:
    """构造给 IndexTTS2 的 emo_text（软指令）。

    这里按你的“单轴更纯粹”的目标做了收敛：

    - **intention 不参与**（已弃用）
    - **speech_rate 只通过后处理 speed_factor 生效**，因此这里不再写“语速加快/放慢”等提示词
      （否则语速就变成“提示词 + speed_factor”双通道，不好判断单轴效果）
    - emotion 的文本提示只来自 semantic_emotions（语义描述）
    """
    axis_mode = normalize_axis_mode(axis_mode)
    use_emo, _use_rate, _use_intention = axis_mode_flags(axis_mode)

    parts: List[str] = []

    # --- emotion（仅由标签驱动，不再根据 utter 做额外关键词推断） ---
    if use_emo and emo and emo != "正常":
        parts.append(f"情绪为{emo}")
    else:
        parts.append("语气自然")

    semantic_prompt = ""
    if use_emo:
        semantic_prompt = str((SEMANTIC_EMO_MAP.get(emo, {}) or {}).get("prompt", "") or "").strip()

    emo_text = "，".join([p for p in parts if p]).strip("，。 ") + "。"
    if semantic_prompt:
        emo_text += semantic_prompt

    print(f"[DEBUG] emo_text={emo_text}")
    return emo_text

def build_emo_vector(
    emo: str,
    axis_mode: str = AXIS_MODE_ALL,
) -> List[float]:
    """构造 emo_vector（8维）。

    纯净策略：
    - 只依赖 base_emotions（配置里的数值向量）
    - 不再叠加 scene / rate / intention 的额外修正（避免“单轴被多轴污染”）
    """
    axis_mode = normalize_axis_mode(axis_mode)
    use_emo, _use_rate, _use_intention = axis_mode_flags(axis_mode)

    emo_for_vec = emo if use_emo else "正常"
    raw = BASE_EMOTION_MAP.get(emo_for_vec) or BASE_EMOTION_MAP.get("正常") or [0.0] * 8

    vec = [float(x) for x in (list(raw)[:8] + [0.0] * 8)[:8]]

    # 轻量归一：避免有人把值写得太大导致模型“表演过度”
    # 这里不改变相对形状，只把整体限制在 [-1, 1] 的安全范围内。
    vec = [max(-1.0, min(1.0, v)) for v in vec]

    print(f"[DEBUG] emo_vector({emo_for_vec})={vec}")
    return vec


def apply_scene_behavior(scene: str, role: str, base_emo: str, rate: str) -> Tuple[str, float]:
    scene_mapping = {"normal": "正常通话", "suspicious": "涉嫌诈骗", "fraud": "确定诈骗"}
    cn_scene = scene_mapping.get(scene, "正常通话")
    profile = (SCENE_BEHAVIOR_PROFILE or {}).get(cn_scene, {}).get(role, {})

    emo_alpha = profile.get("emotion_alpha_base", (EMO_ALPHA_MAP or {}).get(base_emo, 0.35))
    emo_alpha += (profile.get("emotion_boost", {}) or {}).get(base_emo, 0.0)
    emo_alpha += (profile.get("speech_rate_bias", {}) or {}).get(rate, 0.0)
    emo_alpha = max(0.25, min(float(emo_alpha), 0.75))

    return base_emo, emo_alpha

def apply_conversation_progression(scene: str, turn_idx: int, total_turns: int, base_emo_alpha: float, role: str) -> float:
    if scene not in ("suspicious", "fraud"):
        return base_emo_alpha

    progression_factor = min(turn_idx / max(total_turns, 1), 1.0)

    if role == "Agent":
        intensity_boost = (0.20 if scene == "fraud" else 0.15) * progression_factor
    else:
        intensity_boost = (0.15 if scene == "fraud" else 0.10) * progression_factor

    return min(base_emo_alpha + intensity_boost, 0.75)

#====================== JSON解析 =======================
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

# ===================== 后处理 =====================
def _postprocess_cfg() -> Dict[str, float]:
    # 允许在 YAML 增加 postprocess_audio:
    #   normalize_headroom_db, target_peak_dbfs
    cfg = {
        "normalize_headroom_db": 6.0,
        "target_peak_dbfs": -1.0,
    }
    if POSTPROCESS_AUDIO_CFG:
        for k in list(cfg.keys()):
            if k in POSTPROCESS_AUDIO_CFG:
                try:
                    cfg[k] = float(POSTPROCESS_AUDIO_CFG[k])
                except Exception:
                    pass
    return cfg

def simple_postprocess(inp: str, outp: str, emo: str, rate: str) -> bool:
    try:
        if not os.path.exists(inp):
            print(f"[ERROR] 输入音频不存在：{inp}")
            return False

        audio = AudioSegment.from_wav(inp)
        pp = _postprocess_cfg()

        # normalize：留余量
        audio = audio.normalize(headroom=float(pp["normalize_headroom_db"]))

        gain = calculate_energy_gain(emo, rate)
        if gain:
            target_peak = float(pp["target_peak_dbfs"])
            peak_after = audio.max_dBFS + gain
            if peak_after > target_peak:
                gain = target_peak - audio.max_dBFS
            audio = audio.apply_gain(gain)

        fade_duration = get_rate_fade_duration(rate)
        audio = audio.fade_in(fade_duration).fade_out(fade_duration)

        os.makedirs(os.path.dirname(outp), exist_ok=True)
        audio.export(outp, format="wav")
        return True

    except Exception as e:
        print(f"[ERROR] 后处理失败：{e}")
        try:
            shutil.copy2(inp, outp)
            return True
        except Exception:
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

    try:
        utter_raw = turn.get("utterance","").strip()
        base_emo_raw = (turn.get("emotion") or "正常").strip() or "正常"
        rate_raw = (turn.get("speech_rate_anomaly") or "正常").strip() or "正常"
        intention_raw = (turn.get("intention") or "").strip()

        axis_mode = normalize_axis_mode(axis_mode)
        base_emo, rate, intention = masked_labels(
            base_emo=base_emo_raw,
            rate=rate_raw,
            intention=intention_raw,
            axis_mode=axis_mode
        )
        role = "Agent" if turn.get("speaker","").lower() in ["agent","客服"] else "User"
        speaker = "Agent" if role=="Agent" else f"User_{gender}"

        print(f"\n[处理第{idx}/{total_turns}轮] {role}：{utter_raw}")

        # emotion：纯净版本（只由 base_emotions + emotion_alpha_map 决定强度；其余不参与）
        emo = base_emo
        emo_alpha = float(EMO_ALPHA_MAP.get(emo, 0.35))
        emo_alpha = max(0.0, min(1.0, emo_alpha))
        # ===== 文本处理 =====
        print(f"\n[DEBUG] 步骤1: 文本处理")
        utter = normalize_text_for_tts(utter_raw)
        utter = apply_text_process(utter)
        utter = apply_emotion_rhythm_template(utter, emo)
        # scene/fraud/intention 等“多轴注入”已禁用：保持单轴纯净
        # 语速韵律标记处理
        utter = enhance_text_with_rhythm_marks(utter, rate)

        print(f"[DEBUG] 最终处理文本: {utter}")

        # 用纯ASCII短名，避免中文标点/超长文件名导致写出的wav损坏
        uid = uuid.uuid4().hex[:8]
        role_tag = "agent" if role == "Agent" else "user"
        text_tag = safe_filename_readable(utter_raw, max_len=24)
        base = f"turn{idx:02d}_{role_tag}_{text_tag}_{uid}"

        raw = os.path.join(out_dir, f"{base}_raw.wav")
        speed_wav = os.path.join(out_dir, f"{base}_speed.wav")  # 显式定义
        out = os.path.join(out_dir, f"{base}.wav")

        # ===== TTS 合成 =====
        print(f"\n[DEBUG] 步骤2: TTS合成")
        # intention 已弃用：不再基于意图标签改写 emotion / speech_rate

        # 获取语速配置
        rate_config = get_rate_config(rate)
        max_len = rate_config.get("max_len", 28)
        chunks = split_text_for_semantic_safety(utter, max_len=max_len, rate=rate)

        audio_all = AudioSegment.empty()

        print(f"[DEBUG] 开始处理 {len(chunks)} 个文本块")
        for j, sub in enumerate(chunks):
            tmp = os.path.join(out_dir, f"{base}_chunk{j}.wav")

            # 根据语速调整停顿（使用统一配置）
            pause_before = 0
            if j > 0:
                pause_params = get_rate_pause_params(rate, "before_chunk")
                pause_before = (
                    pause_params["long_pause"]
                    if random.random() < float(pause_params["long_pause_prob"])
                    else pause_params["short_pause"]
                )
            if pause_before > 0:
                audio_all += AudioSegment.silent(duration=pause_before)

            tts.infer(
                spk_audio_prompt=speaker_map[speaker],
                text=sub,
                output_path=tmp,
                use_emo_text=True,
                emo_text=build_emo_text(sub, emo, rate, axis_mode=axis_mode),
                emo_vector=build_emo_vector(emo, axis_mode=axis_mode),
                emo_alpha=float(emo_alpha),
            )

            if not os.path.exists(tmp):
                continue
            if os.path.getsize(tmp) < 100 or (not is_valid_wav_header(tmp)):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                continue

            try:
                audio_all += AudioSegment.from_wav(tmp)
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass

            if j < len(chunks) - 1:
                pause_params = get_rate_pause_params(rate, "between_chunks")
                gap = (
                    pause_params["long_pause"]
                    if random.random() < float(pause_params["long_pause_prob"])
                    else pause_params["short_pause"]
                )
                audio_all += AudioSegment.silent(duration=int(gap))

        os.makedirs(out_dir, exist_ok=True)
        audio_all.export(raw, format="wav")
        print(f"[DEBUG] 原始音频保存成功，大小: {os.path.getsize(raw)} bytes")

        # ===== 语速调整 =====
        print(f"\n[DEBUG] 步骤3: 语速调整")
        speed_wav = raw.replace("_raw.wav", "_speed.wav")
        adjust_audio_speed(raw, speed_wav, rate)
        print(f"[DEBUG] 语速调整完成，文件大小: {os.path.getsize(speed_wav)} bytes")

        # ===== 轻量后处理 =====
        print(f"\n[DEBUG] 步骤4: 轻量后处理")
        ok = simple_postprocess(inp=speed_wav, outp=out, emo=emo, rate=rate)

        # 如果后处理失败或 out 没生成，至少保证有一个可用 wav（回退到 speed/raw）
        if (not ok) or (not os.path.exists(out)):
            print(f"[WARN] 后处理未生成最终文件，fallback: copy speed/raw -> out")
            try:
                fallback_src = speed_wav if os.path.exists(speed_wav) else raw
                shutil.copy2(fallback_src, out)
            except Exception as e:
                print(f"[ERROR] fallback copy failed: {e}")

        print(f"[DEBUG] 最终输出文件大小: {os.path.getsize(out)} bytes")
        # 清理中间文件
        for p in (raw, speed_wav):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        return out if os.path.exists(out) else None


    except Exception as e:
        print(f"[ERROR] synthesize_turn failed: turn={idx} err={e}")
        traceback.print_exc()  # 建议加上堆栈打印，方便排查具体哪行错
        return None

# ====================== 导出格式 ===========================
def axis_catalog() -> Dict[str, Tuple[str, str]]:
    """axis_dir_name -> (axis_mode, label_key)"""
    return {
        "emotion": (AXIS_MODE_EMOTION_ONLY, "emotion"),
                "speech_rate_anomaly": (AXIS_MODE_RATE_ONLY, "speech_rate_anomaly"),
    }

def list_final_wavs(turns_dir: str) -> List[str]:
    if not os.path.isdir(turns_dir):
        return []
    files = []
    for f in os.listdir(turns_dir):
        if not f.lower().endswith(".wav"):
            continue
        if "_chunk" in f or f.endswith("_raw.wav") or f.endswith("_speed.wav"):
            continue
        files.append(os.path.join(turns_dir, f))
    return sorted(files)


def merge_turns_wav(turns_dir: str, merged_path: str) -> bool:
    wav_files = list_final_wavs(turns_dir)
    if not wav_files:
        return False

    segs: List[AudioSegment] = []
    for w in wav_files:
        try:
            a = AudioSegment.from_wav(w)
            if len(a) >= 300:
                segs.append(a)
        except Exception:
            pass

    if not segs:
        return False

    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    comb = AudioSegment.silent(300)
    for a in segs:
        comb += a + AudioSegment.silent(400)
    comb.export(merged_path, format="wav")
    return True


def relpath(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start)
    except Exception:
        return path


def export_dataset(
    *,
    input_jsonl: str,
    speaker_map_path: Optional[str] = None,
    cfg: str,
    model_dir: str,
    configs_dir: str,
    out_root: str,
    lines: str = "",
    merge: bool = False,
    with_conversation: bool = False,
    conversation_only: bool = False,
    conversation_dir_name: str = "conversation",
    axes: Optional[List[str]] = None,
    fp16: bool = False,
) -> str:
    """导出数据集。返回 run_dir。"""

    if conversation_only:
        with_conversation = True

    run_dir = os.path.join(out_root, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("AXIS_EXPORT")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(run_dir, "run_main.log"), encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(fh)
    logger.addHandler(sh)

    sys.stdout = TeeLogger(logger)
    sys.stderr = TeeLogger(logger, logging.ERROR)

    # configs + model
    load_configs(configs_dir)
    speaker_map: Dict[str, Any] = {}

    # 1) 优先用 YAML 的 speaker_map
    if isinstance(SPEAKER_MAP, dict) and SPEAKER_MAP:
        speaker_map = SPEAKER_MAP
    # 2) YAML 没有才用 json 文件（兼容旧流程）
    elif speaker_map_path:
        with open(speaker_map_path, "r", encoding="utf-8") as f:
            speaker_map = json.load(f)

    if not isinstance(speaker_map, dict) or not speaker_map:
        raise RuntimeError(
            "speaker_map 为空：请在 configs_dir/config.yaml 顶层加入 speaker_map，"
            "或传入 --speaker_map 指向旧的 speaker_map.json"
        )

    print(f"[CFG] speaker_map_keys_used={len(speaker_map)}")

    tts = IndexTTS2(cfg, model_dir, use_fp16=fp16)

    selector = parse_line_selector(lines)

    # axes selection
    catalog = axis_catalog()
    selected_axes = axes or list(catalog.keys())
    selected_axes = [a.strip() for a in selected_axes if a and a.strip()]
    for a in list(selected_axes):
        if a not in catalog:
            raise ValueError(f"Unknown axis '{a}'. Allowed: {list(catalog.keys())}")

    # manifest writers
    manifest_fps: Dict[str, Any] = {}
    if not conversation_only:
        for axis_dir in selected_axes:
            axis_path = os.path.join(run_dir, axis_dir)
            os.makedirs(axis_path, exist_ok=True)
            manifest_fps[axis_dir] = open(
                os.path.join(axis_path, "manifest.jsonl"), "w", encoding="utf-8", buffering=1
            )

    conv_manifest_fp = None
    conv_root = None
    if with_conversation:
        conv_root = os.path.join(run_dir, conversation_dir_name)
        os.makedirs(conv_root, exist_ok=True)
        conv_manifest_fp = open(os.path.join(conv_root, "manifest.jsonl"), "w", encoding="utf-8", buffering=1)

    try:
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if selector and line_idx not in selector:
                    continue

                obj = json.loads(line)
                conv_list = extract_conversation(obj)
                profile = extract_user_profile(obj)
                final_label = (obj.get("content", {}) or {}).get("final_label", "")
                scene = map_scene(final_label)
                gender = "female" if "女" in (profile or "") else "male"

                call_id = str(obj.get("call_id", f"line{line_idx}"))
                name_match = re.search(r"姓名[:：]\s*([^；;,]+)", profile or "")
                name = name_match.group(1).strip() if name_match else "unknown"
                name = safe_name(name)
                call_key = f"{call_id}_{name}"

                print(f"\n[LINE {line_idx}] call={call_key} turns={len(conv_list)} scene={scene} gender={gender}")

                # conversation output
                conv_call_dir = None
                conv_turns_dir = None
                if with_conversation and conv_root:
                    conv_call_dir = os.path.join(conv_root, call_key)
                    conv_turns_dir = os.path.join(conv_call_dir, "turns")
                    os.makedirs(conv_turns_dir, exist_ok=True)

                    with open(os.path.join(conv_call_dir, "raw_input.jsonl"), "a", encoding="utf-8") as wf:
                        wf.write(line.rstrip() + "\n")

                    parsed_path = os.path.join(conv_call_dir, f"{call_key}.json")
                    with open(parsed_path, "w", encoding="utf-8") as wf:
                        json.dump(obj, wf, ensure_ascii=False, indent=2)

                # per-turn
                for turn_idx, turn in enumerate(conv_list, start=1):
                    utter_raw = (turn.get("utterance") or "").strip()

                    # 1) conversation (all)
                    if with_conversation and conv_turns_dir:
                        wav_all = synthesize_turn(
                            tts=tts,
                            turn=turn,
                            scene=scene,
                            speaker_map=speaker_map,
                            gender=gender,
                            out_dir=conv_turns_dir,
                            call_dir=conv_call_dir or conv_turns_dir,
                            idx=turn_idx,
                            total_turns=len(conv_list),
                            axis_mode=AXIS_MODE_ALL,
                        )
                        if wav_all and conv_manifest_fp:
                            role = "Agent" if (turn.get("speaker") or "").lower() in ["agent", "客服"] else "User"
                            rec_all = {
                                "axis": conversation_dir_name,
                                "axis_mode": AXIS_MODE_ALL,
                                "scene": scene,
                                "call_id": call_id,
                                "name": name,
                                "turn_idx": turn_idx,
                                "role": role,
                                "text": utter_raw,
                                "emotion": (turn.get("emotion") or "正常").strip() or "正常",
                                "speech_rate_anomaly": (turn.get("speech_rate_anomaly") or "正常").strip() or "正常",
                                "intention": (turn.get("intention") or "").strip(),
                                "wav": relpath(wav_all, run_dir),
                            }
                            conv_manifest_fp.write(json.dumps(rec_all, ensure_ascii=False) + "\n")

                    # 2) axis mask outputs
                    if conversation_only:
                        continue

                    for axis_dir in selected_axes:
                        axis_mode, label_key = catalog[axis_dir]
                        label_val = (turn.get(label_key) or "正常").strip() or "正常"
                        label_dir = safe_name(label_val)

                        call_dir = os.path.join(run_dir, axis_dir, label_dir, call_key)
                        turns_dir = os.path.join(call_dir, "turns")
                        os.makedirs(turns_dir, exist_ok=True)

                        wav_path = synthesize_turn(
                            tts=tts,
                            turn=turn,
                            scene=scene,
                            speaker_map=speaker_map,
                            gender=gender,
                            out_dir=turns_dir,
                            call_dir=call_dir,
                            idx=turn_idx,
                            total_turns=len(conv_list),
                            axis_mode=axis_mode,
                        )

                        if not wav_path:
                            logger.error(f"[FAIL] axis={axis_dir} turn={turn_idx} label={label_val} call={call_key}")
                            continue

                        role = "Agent" if (turn.get("speaker") or "").lower() in ["agent", "客服"] else "User"
                        rec = {
                            "axis": axis_dir,
                            "axis_mode": axis_mode,
                            "label": label_val,
                            "scene": scene,
                            "call_id": call_id,
                            "name": name,
                            "turn_idx": turn_idx,
                            "role": role,
                            "text": utter_raw,
                            "wav": relpath(wav_path, run_dir),
                        }
                        manifest_fps[axis_dir].write(json.dumps(rec, ensure_ascii=False) + "\n")

                # merge
                if with_conversation and merge and conv_turns_dir and conv_call_dir:
                    merged_path = os.path.join(conv_call_dir, f"merged-{call_key}.wav")
                    if merge_turns_wav(conv_turns_dir, merged_path):
                        print(f"[MERGED] {relpath(merged_path, run_dir)}")

                if merge and (not conversation_only):
                    for axis_dir in selected_axes:
                        axis_root = os.path.join(run_dir, axis_dir)
                        if not os.path.isdir(axis_root):
                            continue
                        for label_dir in os.listdir(axis_root):
                            call_dir = os.path.join(axis_root, label_dir, call_key)
                            turns_dir = os.path.join(call_dir, "turns")
                            if not os.path.isdir(turns_dir):
                                continue
                            merged_path = os.path.join(call_dir, f"merged-{call_key}.wav")
                            if merge_turns_wav(turns_dir, merged_path):
                                print(f"[MERGED] {relpath(merged_path, run_dir)}")

    finally:
        for fp in manifest_fps.values():
            try:
                fp.close()
            except Exception:
                pass
        if conv_manifest_fp:
            try:
                conv_manifest_fp.close()
            except Exception:
                pass

    print(f"\nALL DONE! 输出到：{run_dir}")
    return run_dir

def _parse_axes_arg(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

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
    ap.add_argument(
        "--with_conversation",
        action="store_true",
        help="同时导出 conversation(all) 输出：run_dir/<conversation_dir_name>/<call>/turns",
    )
    ap.add_argument(
        "--conversation_only",
        action="store_true",
        help="只导出 conversation(all)，不生成单轴目录",
    )
    ap.add_argument(
        "--conversation_dir_name",
        type=str,
        default="conversation",
        help="conversation 输出目录名（默认: conversation）",
    )

    ap.add_argument(
        "--axes",
        type=str,
        default="emotion,speech_rate_anomaly",
        help="要生成的单轴目录，逗号分隔（可选：emotion,speech_rate_anomaly）。\n"
             "例如只生成 emotion：--axes emotion",
    )

    ap.add_argument("--fp16", action="store_true", help="Use FP16 to reduce VRAM")

    args = ap.parse_args()

    export_dataset(
        input_jsonl=args.input,
        speaker_map_path=args.speaker_map,
        cfg=args.cfg,
        model_dir=args.model_dir,
        configs_dir=args.configs_dir,
        out_root=args.out_root,
        lines=args.lines,
        merge=args.merge,
        with_conversation=args.with_conversation,
        conversation_only=args.conversation_only,
        conversation_dir_name=args.conversation_dir_name,
        axes=_parse_axes_arg(args.axes),
        fp16=args.fp16,
    )

if __name__ == "__main__":
    main()
