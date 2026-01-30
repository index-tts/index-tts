"""
export_single_axis.py

薄封装：统一调用 axis_engine.export_dataset()，避免重复实现导致的逻辑漂移/bug。
输出结构与 axis_engine.export_dataset 保持一致：
  run_%Y%m%d_%H%M%S/
    emotion/<label>/<call>/turns/*.wav
    microexpression/<label>/<call>/turns/*.wav
    speech_rate_anomaly/<label>/<call>/turns/*.wav
    conversation/<call>/turns/*.wav   (可选)
"""

import argparse
import axis_engine as conv


def _parse_axes_arg(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--configs_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--speaker_map", default="", help="(可选) 旧版 speaker_map.json，YAML 已配置可不传")
    ap.add_argument("--lines", type=str, default="", help="Select lines: 5 | 5,8,10 | 5-12")
    ap.add_argument("--merge", action="store_true", help="(可选) 合并每个 call 的 turns -> merged-*.wav")
    ap.add_argument("--with_conversation", action="store_true", help="(可选) 同时导出 conversation(all) 输出")
    ap.add_argument("--conversation_only", action="store_true", help="只导出 conversation(all)，不生成单轴目录")
    ap.add_argument("--conversation_dir_name", type=str, default="conversation", help="conversation 输出目录名")
    ap.add_argument(
        "--axes",
        type=str,
        default="emotion,microexpression,speech_rate_anomaly",
        help="选择要导出的单轴目录（逗号分隔）。可选: emotion,microexpression,speech_rate_anomaly。默认全部。",
    )
    ap.add_argument("--fp16", action="store_true", help="Use FP16 to reduce VRAM")

    args = ap.parse_args()

    conv.export_dataset(
        input_jsonl=args.input,
        cfg=args.cfg,
        model_dir=args.model_dir,
        configs_dir=args.configs_dir,
        speaker_map_path=(args.speaker_map or None),
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
