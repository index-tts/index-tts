"""Voice-oriented post-processing for IndexTTS output.

The synthesizer keeps speaker identity; this module optionally cleans mud,
boxiness, and sibilance without overwriting the original WAV.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import soundfile as sf


PRESET_FIELDS = (
    "low_cut",
    "bass",
    "mud",
    "box",
    "presence",
    "de_ess",
    "high_cut",
    "gain",
    "normalize",
)

# Stable IDs used by the WebUI, CLI, and tests. Display names are i18n keys.
TUNING_PRESETS = {
    "bypass": {
        "low_cut": 20,
        "bass": 0.0,
        "mud": 0.0,
        "box": 0.0,
        "presence": 0.0,
        "de_ess": 0.0,
        "high_cut": 11000,
        "gain": 0.0,
        "normalize": False,
    },
    "voice_clarity": {
        "low_cut": 62,
        "bass": -1.4,
        "mud": -3.0,
        "box": -1.5,
        "presence": 1.0,
        "de_ess": 0.0,
        "high_cut": 10500,
        "gain": 0.0,
        "normalize": False,
    },
    "clear_narration": {
        "low_cut": 58,
        "bass": -1.0,
        "mud": -3.2,
        "box": -1.8,
        "presence": 1.2,
        "de_ess": 0.8,
        "high_cut": 9800,
        "gain": 0.0,
        "normalize": True,
    },
    "deharsh": {
        "low_cut": 55,
        "bass": 0.0,
        "mud": -2.0,
        "box": -1.0,
        "presence": 0.0,
        "de_ess": 2.5,
        "high_cut": 9200,
        "gain": 0.0,
        "normalize": False,
    },
    "warm": {
        "low_cut": 45,
        "bass": 1.0,
        "mud": -2.2,
        "box": -1.0,
        "presence": 0.5,
        "de_ess": 1.0,
        "high_cut": 10500,
        "gain": 0.0,
        "normalize": False,
    },
}

PRESET_LABELS = {
    "bypass": "原声（不处理）",
    "voice_clarity": "旁白清晰",
    "clear_narration": "清澈旁白",
    "deharsh": "轻柔去毛刺",
    "warm": "温暖自然",
}

DEFAULT_PRESET = "voice_clarity"

# CLI aliases, including older local names.
PRESET_ALIASES = {
    "none": "bypass",
    "bypass": "bypass",
    "score92": "voice_clarity",
    "voice-clarity": "voice_clarity",
    "voice_clarity": "voice_clarity",
    "clear-narration": "clear_narration",
    "clear_narration": "clear_narration",
    "smooth": "deharsh",
    "deharsh": "deharsh",
    "warm": "warm",
}


def resolve_preset(name: str | None) -> str:
    if not name:
        return DEFAULT_PRESET
    preset_id = PRESET_ALIASES.get(str(name).strip())
    if preset_id is None:
        raise ValueError(f"Unknown tuning preset: {name}")
    return preset_id


def preset_values(name: str) -> tuple:
    preset = TUNING_PRESETS[resolve_preset(name)]
    return tuple(preset[field] for field in PRESET_FIELDS)


def preset_choices(translate=None) -> list[tuple[str, str]]:
    label = translate or (lambda key: key)
    return [(label(PRESET_LABELS[preset_id]), preset_id) for preset_id in TUNING_PRESETS]


def _eq_filter(frequency: int, gain: float, width: float) -> str | None:
    if abs(float(gain)) < 0.05:
        return None
    return f"equalizer=f={frequency}:t=q:w={width}:g={float(gain):.2f}"


def tune_audio(
    input_path: str | None,
    low_cut: float,
    bass: float,
    mud: float,
    box: float,
    presence: float,
    de_ess: float,
    high_cut: float,
    gain: float,
    normalize: bool,
    output_path: str | None = None,
) -> tuple[str, str]:
    """Apply voice-oriented FFmpeg filters and return output path + summary."""
    if not input_path:
        raise ValueError("请先上传音频，或点击“载入刚生成的音频”。")

    source = Path(input_path)
    if not source.is_file():
        raise ValueError(f"找不到输入音频：{source}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，请先安装 ffmpeg。")

    info = sf.info(str(source))
    sample_rate = int(info.samplerate)
    nyquist = sample_rate / 2
    safe_high_cut = min(float(high_cut), nyquist * 0.98)
    safe_low_cut = min(max(float(low_cut), 20.0), safe_high_cut - 100)

    filters: list[str] = []
    if safe_low_cut > 20.5:
        filters.append(f"highpass=f={safe_low_cut:.1f}:p=2")

    # Voice-oriented bands: body, mud, boxiness, presence, sibilance.
    for item in (
        _eq_filter(145, bass, 0.8),
        _eq_filter(330, mud, 0.9),
        _eq_filter(750, box, 0.9),
        _eq_filter(2800, presence, 1.0),
        _eq_filter(6500, -abs(float(de_ess)), 1.2),
    ):
        if item:
            filters.append(item)

    if safe_high_cut < nyquist * 0.98:
        filters.append(f"lowpass=f={safe_high_cut:.1f}:p=2")
    if abs(float(gain)) >= 0.05:
        filters.append(f"volume={float(gain):.2f}dB")
    if normalize:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    filters.append("alimiter=limit=0.95")

    if output_path:
        destination = Path(output_path)
        if destination.resolve() == source.resolve():
            raise ValueError("输出路径不能与输入音频相同。")
        destination.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("outputs") / "tuned"
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / (
            f"{source.stem}_tuned_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
        )

    is_bypass = (
        safe_low_cut <= 20.5
        and float(high_cut) >= min(11000, nyquist * 0.98)
        and all(abs(float(value)) < 0.05 for value in (bass, mud, box, presence, de_ess, gain))
        and not normalize
    )
    if is_bypass:
        shutil.copy2(source, destination)
        return str(destination), f"未应用调音，已复制原声：{sample_rate} Hz / {info.duration:.1f} 秒"

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-af",
        ",".join(filters),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "未知 FFmpeg 错误"
        raise RuntimeError(f"音频处理失败：{message}")

    duration = sf.info(str(destination)).duration
    summary = (
        f"处理完成：{sample_rate} Hz / {duration:.1f} 秒  \n"
        f"低切 {safe_low_cut:.0f} Hz · 低频 {float(bass):+.1f} dB · "
        f"浑浊 {float(mud):+.1f} dB · 箱体 {float(box):+.1f} dB · "
        f"清晰度 {float(presence):+.1f} dB · 去毛刺 {float(de_ess):.1f} dB"
    )
    return str(destination), summary


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-process a voice WAV; prints one JSON result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", help="Input audio path")
    parser.add_argument("--output", help="Exact output WAV path")
    parser.add_argument("--preset", choices=sorted(PRESET_ALIASES), default=DEFAULT_PRESET)
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--low-cut", type=float, default=None, help="High-pass cutoff Hz")
    parser.add_argument("--bass", type=float, default=None, help="145 Hz gain dB")
    parser.add_argument("--mud", type=float, default=None, help="330 Hz gain dB")
    parser.add_argument("--box", type=float, default=None, help="750 Hz gain dB")
    parser.add_argument("--presence", type=float, default=None, help="2.8 kHz gain dB")
    parser.add_argument("--de-ess", type=float, default=None, help="6.5 kHz reduction dB")
    parser.add_argument("--high-cut", type=float, default=None, help="Low-pass cutoff Hz")
    parser.add_argument("--gain", type=float, default=None, help="Output gain dB")
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize to -16 LUFS",
    )
    args = parser.parse_args(argv)

    def emit(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False))

    if args.list_presets:
        emit(
            {
                "ok": True,
                "default": DEFAULT_PRESET,
                "presets": {
                    preset_id: {"label": PRESET_LABELS[preset_id], **params}
                    for preset_id, params in TUNING_PRESETS.items()
                },
                "aliases": PRESET_ALIASES,
            }
        )
        return 0
    if not args.input:
        emit({"ok": False, "error": "--input is required"})
        return 2

    try:
        preset_id = resolve_preset(args.preset)
        params = dict(TUNING_PRESETS[preset_id])
        overrides = {
            "low_cut": args.low_cut,
            "bass": args.bass,
            "mud": args.mud,
            "box": args.box,
            "presence": args.presence,
            "de_ess": args.de_ess,
            "high_cut": args.high_cut,
            "gain": args.gain,
            "normalize": args.normalize,
        }
        params.update({key: value for key, value in overrides.items() if value is not None})
        output, summary = tune_audio(
            args.input,
            *(params[field] for field in PRESET_FIELDS),
            output_path=args.output,
        )
    except Exception as exc:
        emit({"ok": False, "error": str(exc)})
        return 1

    emit(
        {
            "ok": True,
            "input": str(Path(args.input)),
            "output": output,
            "preset": preset_id,
            "parameters": params,
            "summary": summary.replace("  \n", "；"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
