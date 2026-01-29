"""export_axis_mask_dataset.py

单轴数据导出脚本（Axis Mask）。

目标：在不影响你原有“对话流合成”脚本的前提下，额外生成 3 份数据：
  1) emotion_only
  2) micro_only
  3) rate_only

并按如下结构落盘：
  out_root/
    run_%Y%m%d_%H%M%S/
      emotion/<label>/<call_id>_<name>/turns/*.wav
      microexpression/<label>/<call_id>_<name>/turns/*.wav
      speech_rate_anomaly/<label>/<call_id>_<name>/turns/*.wav

每个 axis 也会生成 manifest.jsonl，方便训练/评测直接读取。

用法示例：
/home/shentong/miniconda3/envs/indextts2/bin/python axis_pack/export_single_axis.py \
  --input axis_pack/all_conversations.jsonl \
  --speaker_map axis_pack/speaker_map.json \
  --cfg checkpoints/config.yaml \
  --model_dir checkpoints/IndexTTS-2 \
  --configs_dir axis_pack/configs \
  --out_root axis_pack/output_export_by_tags \
  --lines 1

注意：
  - 本脚本依赖你已经在 conversation_test.py 中加入 axis mask（synthesize_turn 支持 axis_mode）。
  - 如果你把 patch 版本命名为 conversation_test_axis_mask.py，请修改下面 import。
"""

import argparse
import json
import re
import logging
from datetime import datetime
from typing import Dict, Any, Set
import sys, os
from pydub import AudioSegment
import axis_engine as conv
from indextts.infer_v2 import IndexTTS2

def parse_line_selector(expr: str) -> Set[int]:
    return conv.parse_line_selector(expr)

def safe_name(s: str, max_len: int = 80) -> str:
    s = s.strip() if s else "unknown"
    s = re.sub(r"[\\/*?:\"<>|]", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:max_len]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def relpath(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start)
    except Exception:
        return path

def axis_spec():
    """(axis_dir_name, axis_mode, label_key)"""
    return [
        ("emotion", conv.AXIS_MODE_EMOTION_ONLY, "emotion"),
        ("microexpression", conv.AXIS_MODE_MICRO_ONLY, "microexpression"),
        ("speech_rate_anomaly", conv.AXIS_MODE_RATE_ONLY, "speech_rate_anomaly"),
    ]


def list_final_wavs(turns_dir: str):
    """只挑最终成品 wav，过滤掉 _raw/_speed/_chunk 等中间文件。"""
    if not os.path.isdir(turns_dir):
        return []
    wav_files = sorted([
        os.path.join(turns_dir, f)
        for f in os.listdir(turns_dir)
        if f.lower().endswith(".wav")
           and ("_chunk" not in f)
           and (not f.endswith("_raw.wav"))
           and (not f.endswith("_speed.wav"))
    ])
    return wav_files


def merge_turns_wav(turns_dir: str, merged_path: str) -> bool:
    """把 turns/*.wav 合成一个 merged wav（和 conversation_test 同策略：300ms 开头 + 每段 400ms 间隔）。"""
    wav_files = list_final_wavs(turns_dir)
    if not wav_files:
        return False

    segs = []
    for w in wav_files:
        try:
            a = AudioSegment.from_wav(w)
            if len(a) >= 300:
                segs.append(a)
        except Exception:
            pass
    if not segs:
        return False

    ensure_dir(os.path.dirname(merged_path))
    comb = AudioSegment.silent(300)
    for a in segs:
        comb += a + AudioSegment.silent(400)
    comb.export(merged_path, format="wav")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--speaker_map", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--configs_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--lines", type=str, default="", help="Select lines to process, e.g. 5 | 5,8,10 | 5-12")
    ap.add_argument("--merge", action="store_true", help="(可选) 在每个 label/call 目录下把 turns 合并成 merged.wav")
    ap.add_argument(
        "--with_conversation",
        action="store_true",
        help="(可选) 同时导出 conversation_test 风格的整合输出（all 轴，不按 label 分目录）",
    )
    ap.add_argument(
        "--conversation_dir_name",
        type=str,
        default="conversation",
        help="整合输出目录名（默认: conversation；会与 emotion/microexpression/... 同级）",
    )
    ap.add_argument(
        "--conversation_only",
        action="store_true",
        help="只导出 conversation 整合输出（all 轴），不生成 emotion/micro/rate 的单轴目录",
    )
    ap.add_argument("--fp16", action="store_true", help="Use FP16 to reduce VRAM")
    args = ap.parse_args()

    if args.conversation_only:
        args.with_conversation = True

    run_dir = os.path.join(args.out_root, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_dir)

    # 全局日志
    logger = logging.getLogger("AXIS_EXPORT")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(run_dir, "run_main.log"), encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(fh)
    logger.addHandler(sh)
    sys.stdout = conv.TeeLogger(logger)
    sys.stderr = conv.TeeLogger(logger, logging.ERROR)

    # 加载配置
    conv.load_configs(args.configs_dir)
    speaker_map: Dict[str, Any] = json.load(open(args.speaker_map, "r", encoding="utf-8"))
    tts = IndexTTS2(args.cfg, args.model_dir, use_fp16=args.fp16)

    line_selector = parse_line_selector(args.lines)

    # 每个 axis 一个 manifest
    manifest_fps = {}
    if not args.conversation_only:
        for axis_dir, axis_mode, label_key in axis_spec():
            axis_path = os.path.join(run_dir, axis_dir)
            ensure_dir(axis_path)
            manifest_fps[axis_dir] = open(
                os.path.join(axis_path, "manifest.jsonl"),
                "w",
                encoding="utf-8",
                buffering=1,
            )

    # 额外：整合输出（conversation_test 风格）
    conv_manifest_fp = None
    conv_root = None
    if args.with_conversation:
        conv_root = os.path.join(run_dir, args.conversation_dir_name)
        ensure_dir(conv_root)
        conv_manifest_fp = open(
            os.path.join(conv_root, "manifest.jsonl"),
            "w",
            encoding="utf-8",
            buffering=1,
        )

    try:
        for i, line in enumerate(open(args.input, "r", encoding="utf-8"), start=1):
            if line_selector and i not in line_selector:
                continue

            obj = json.loads(line)
            conv_list = conv.extract_conversation(obj)
            profile = conv.extract_user_profile(obj)
            final_label = obj.get("content", {}).get("final_label", "")
            scene = conv.map_scene(final_label)
            gender = "female" if "女" in (profile or "") else "male"

            call_id = str(obj.get("call_id", f"line{i}"))
            name_match = re.search(r"姓名[:：]\s*([^；;,]+)", profile or "")
            name = name_match.group(1).strip() if name_match else "unknown"
            name = safe_name(name)

            call_key = f"{call_id}_{name}"
            print(f"\n[LINE {i}] call={call_key} turns={len(conv_list)} scene={scene} gender={gender}")

            # =====================
            # (可选) 整合输出：conversation_test 同款目录结构
            # run_dir/<conversation_dir_name>/<call_id>_<name>/turns/*.wav
            # =====================
            conv_call_dir = None
            conv_turns_dir = None
            if args.with_conversation and conv_root:
                conv_call_dir = os.path.join(conv_root, call_key)
                conv_turns_dir = os.path.join(conv_call_dir, "turns")
                ensure_dir(conv_turns_dir)

                # 备份输入（保持与 conversation_test 接近的落盘方式）
                with open(os.path.join(conv_call_dir, "raw_input.jsonl"), "a", encoding="utf-8") as f:
                    f.write(line.rstrip() + "\n")

                parsed_path = os.path.join(conv_call_dir, f"{call_key}.json")
                with open(parsed_path, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)

            for idx, turn in enumerate(conv_list, start=1):
                utter_raw = (turn.get("utterance") or "").strip()

                # ====== 1) 整合输出：all 轴（不做 mask）======
                if args.with_conversation and conv_turns_dir:
                    wav_path_all = conv.synthesize_turn(
                        tts=tts,
                        turn=turn,
                        scene=scene,
                        speaker_map=speaker_map,
                        gender=gender,
                        out_dir=conv_turns_dir,
                        call_dir=conv_call_dir or conv_turns_dir,
                        idx=idx,
                        total_turns=len(conv_list),
                        axis_mode=conv.AXIS_MODE_ALL,
                    )
                    if wav_path_all and conv_manifest_fp:
                        role = "Agent" if (turn.get("speaker") or "").lower() in ["agent", "客服"] else "User"
                        rec_all = {
                            "axis": args.conversation_dir_name,
                            "axis_mode": conv.AXIS_MODE_ALL,
                            "scene": scene,
                            "call_id": call_id,
                            "name": name,
                            "turn_idx": idx,
                            "role": role,
                            "text": utter_raw,
                            # 额外保留原始标签（方便你后续对齐）
                            "emotion": (turn.get("emotion") or "正常").strip() or "正常",
                            "microexpression": (turn.get("microexpression") or "正常").strip() or "正常",
                            "speech_rate_anomaly": (turn.get("speech_rate_anomaly") or "正常").strip() or "正常",
                            "intention": (turn.get("intention") or "").strip(),
                            "wav": relpath(wav_path_all, run_dir),
                        }
                        conv_manifest_fp.write(json.dumps(rec_all, ensure_ascii=False) + "\n")
                        conv_manifest_fp.flush()

                ## ====== 2) 单轴输出：emotion/micro/rate ======
                if not args.conversation_only:
                    for axis_dir, axis_mode, label_key in axis_spec():
                        # 取原始标签作为“归档标签”（即使合成时其它轴被 mask，也不会影响归档）
                        label_val = (turn.get(label_key) or "正常").strip() or "正常"
                        label_dir = safe_name(label_val)

                        call_dir = os.path.join(run_dir, axis_dir, label_dir, call_key)
                        turns_dir = os.path.join(call_dir, "turns")
                        ensure_dir(turns_dir)

                        # 生成单轮
                        wav_path = conv.synthesize_turn(
                            tts=tts,
                            turn=turn,
                            scene=scene,
                            speaker_map=speaker_map,
                            gender=gender,
                            out_dir=turns_dir,
                            call_dir=call_dir,
                            idx=idx,
                            total_turns=len(conv_list),
                            axis_mode=axis_mode,
                        )

                    if not wav_path:
                        logger.error(f"[FAIL] axis={axis_dir} turn={idx} label={label_val} call={call_key}")
                        continue

                    # 写 manifest
                    role = "Agent" if (turn.get("speaker") or "").lower() in ["agent", "客服"] else "User"
                    rec = {
                        "axis": axis_dir,
                        "axis_mode": axis_mode,
                        "label": label_val,
                        "scene": scene,
                        "call_id": call_id,
                        "name": name,
                        "turn_idx": idx,
                        "role": role,
                        "text": utter_raw,
                        "wav": relpath(wav_path, run_dir),
                    }
                    manifest_fps[axis_dir].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    manifest_fps[axis_dir].flush()

            # =====================
            # (可选) 整合输出 merge
            # =====================
            if args.with_conversation and args.merge and conv_turns_dir:
                merged_path = os.path.join(conv_call_dir or conv_root or run_dir, f"merged-{call_key}.wav")
                if merge_turns_wav(conv_turns_dir, merged_path):
                    print(f"[MERGED] {relpath(merged_path, run_dir)}")

            # merge（可选）：在每个 axis/label/call 目录下把 turns 合并
            if args.merge and (not args.conversation_only):
                for axis_dir, _, _ in axis_spec():
                    axis_root = os.path.join(run_dir, axis_dir)
                    if not os.path.isdir(axis_root):
                        continue
                    # 遍历该 call_key 所在的 label 目录（可能多个 label 都包含该 call）
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

    print(f"\nALL DONE! 单轴数据已输出到：{run_dir}")


if __name__ == "__main__":
    main()
