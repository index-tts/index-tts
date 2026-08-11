#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TTS 前端文本归一化（Text Normalization）。

基于 ``nemo_text_processing`` 把数字/符号/日期/货币等 non-standard words 展开成
可朗读文本（例如 ``"25%"`` -> ``"twenty five percent"``）。

设计要点：
- **输入是上游服务语言码**（ar/zh/es/en/ja 这类 ISO 639-1 风格短码）。本模块内部
  维护 ``_SERVICE_TO_NEMO`` 把它转成 NeMo 需要的语言码。也兼容上游直接传 ISO 639-3
  （arb/arz/... 等）的情况——会先折回服务码再查。
- **NeMo 不是所有语言都有 TN grammar**（如日语 ja 没有）。不支持的语言直接返回
  原文透传。
- **懒加载 + 缓存**：``Normalizer`` 构建 grammar 较慢（秒级），按语言缓存实例。
- **失败降级**：``nemo_text_processing`` 未安装、grammar 构建失败、或 normalize
  调用抛异常时，记 warning 并返回原文，绝不中断合成。
"""

import os
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 语言码映射：上游服务码 / ISO 639-3 -> NeMo TN 语言码
#
# 仅列出 NeMo 目前有 TN grammar 的语言。未列出的（如 ja 日语）会跳过归一化。
# NeMo 语言码见 nemo_text_processing.text_normalization.normalize.Normalizer(lang=...)。
# 需要扩充时，确认对应语言在你安装的 nemo 版本里确有 TN grammar 后再加。
# ---------------------------------------------------------------------------

# 服务码（ISO 639-1 风格）-> NeMo 语言码
_SERVICE_TO_NEMO: Dict[str, str] = {
    "ar": "ar",
    "zh": "zh",
    "es": "es",
    "en": "en",
    # "ja": NeMo 无日语 TN grammar，故意不列入 -> 跳过归一化
}

# ISO 639-3 -> 服务码
_ISO3_TO_SERVICE: Dict[str, str] = {
    "arb": "ar",  # standard arabic
    "arz": "ar",  # egyptian arabic
    "ary": "ar",  # moroccan arabic
    "ars": "ar",  # najdi arabic
    "zho": "zh",
    "cmn": "zh",
    "spa": "es",
    "eng": "en",
    "jpn": "ja",
}


def _to_nemo_lang(lang: Optional[str]) -> Optional[str]:
    """把上游语言码映射成 NeMo TN 语言码；不支持归一化则返回 None。"""
    if not lang:
        return None
    key = lang.lower()
    if key in _SERVICE_TO_NEMO:
        return _SERVICE_TO_NEMO[key]
    # 上游可能直接传了 ISO 639-3（如 arb / spa），先折回服务码再查
    svc = _ISO3_TO_SERVICE.get(key)
    if svc and svc in _SERVICE_TO_NEMO:
        return _SERVICE_TO_NEMO[svc]
    return None


class TextNormalizer:
    """按语言懒加载并缓存 NeMo ``Normalizer`` 的封装。

    单例式使用（见模块底部 ``get_text_normalizer()``），使 grammar 只构建一次并跨调用复用。

    Args:
        input_case: NeMo 的大小写处理模式。``"cased"`` 保留大小写（默认，适合含专有
            名词/多语种混排的文本）；``"lower_cased"`` 先转小写再归一化。
    """

    def __init__(self, input_case: str = "cased"):
        self.input_case = input_case
        # nemo_lang -> Normalizer 实例；值为 None 表示该语言不可用（已尝试过并失败）
        self._cache: Dict[str, Optional[object]] = {}

    def _get_normalizer(self, nemo_lang: str):
        """返回缓存的 Normalizer；首次构建，失败则缓存 None 以避免反复重试。"""
        if nemo_lang in self._cache:
            return self._cache[nemo_lang]

        normalizer = None
        try:
            from nemo_text_processing.text_normalization.normalize import Normalizer

            normalizer = Normalizer(input_case=self.input_case, lang=nemo_lang)
            logger.info(f"nemo Normalizer(lang={nemo_lang}) initialized")
        except Exception as e:
            logger.warning(
                f"build nemo Normalizer(lang={nemo_lang}) failed -> "
                f"skip text normalization for this language: {e}"
            )
            normalizer = None

        self._cache[nemo_lang] = normalizer
        return normalizer

    def normalize(self, text: Optional[str], lang: Optional[str]) -> Optional[str]:
        """对 ``text`` 做文本归一化。

        语言不支持 / NeMo 不可用 / 归一化抛异常时，原样返回 ``text``（降级透传）。

        Args:
            text: 待归一化文本。
            lang: 上游语言码（服务码或 ISO 639-3）。

        Returns:
            归一化后的文本；无法处理时返回原文。
        """
        if not text:
            return text

        nemo_lang = _to_nemo_lang(lang)
        if nemo_lang is None:
            # 语言无关模式或 NeMo 无该语言 TN（如 ja）：跳过
            return text

        normalizer = self._get_normalizer(nemo_lang)
        if normalizer is None:
            return text

        try:
            return normalizer.normalize(text, verbose=False)
        except Exception as e:
            logger.warning(
                f"text normalization failed (lang={lang}->{nemo_lang}) -> "
                f"use raw text: {e}"
            )
            return text


_DEFAULT_NORMALIZER: Optional[TextNormalizer] = None


def get_text_normalizer(input_case: str = "cased") -> TextNormalizer:
    """返回进程级共享的 ``TextNormalizer`` 单例。"""
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = TextNormalizer(input_case=input_case)
    return _DEFAULT_NORMALIZER


def normalize_text(text: Optional[str], lang: Optional[str]) -> Optional[str]:
    """便捷入口：用共享单例对 ``text`` 按 ``lang`` 做归一化。"""
    return get_text_normalizer().normalize(text, lang)

def print_nemo_results(lang, result_dir='nemo_tn_result'):
    """读取 result_{lang}.tsv 并逐行打印 nemo_result 列。"""
    result_path = os.path.join(result_dir, f'result_{lang}_front.tsv')
    if not os.path.exists(result_path):
        print(f"[SKIP] {result_path} not found")
        return
    with open(result_path, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                print(parts[3])

def get_nemo_result_main():
    target_langs = ['ja']
    normalize_root = 'nemo_tn_testdata'
    output_dir = 'nemo_tn_result'
    os.makedirs(output_dir, exist_ok=True)

    normalizer = get_text_normalizer()

    for lang in target_langs:
        testset = os.path.join(normalize_root, f'testset_{lang}.tsv')
        if not os.path.exists(testset):
            print(f"[SKIP] {testset} not found")
            continue

        output_path = os.path.join(output_dir, f'result_{lang}.tsv')
        total, match, mismatch = 0, 0, 0
        t_start = time.perf_counter()

        with open(testset, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            header = fin.readline().strip()
            fout.write(f"{header}\tnemo_result\tstatus\n")

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                sid, original, gt = parts[0], parts[1], parts[2]

                nemo_result = normalizer.normalize(original, lang)
                # 去掉首尾空格后比较
                nemo_result = nemo_result.strip() if nemo_result else ""
                gt = gt.strip()
                status = "✅" if nemo_result == gt else "❌"
                total += 1
                if status == "✅":
                    match += 1
                else:
                    mismatch += 1

                fout.write(f"{sid}\t{original}\t{gt}\t{nemo_result}\t{status}\n")

        elapsed = time.perf_counter() - t_start
        avg_ms = elapsed / total * 1000 if total > 0 else 0
        print(f"[{lang.upper()}] total={total}, match={match}, mismatch={mismatch}, "
              f"accuracy={match/total*100:.1f}%, "
              f"avg={avg_ms:.1f}ms/sentence, total_time={elapsed:.2f}s  -> {output_path}")
        

if __name__ == '__main__':
    print_nemo_results('zh')