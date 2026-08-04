"""Text normalization and BPE tokenization for Faster IndexTTS-2."""

import os
import re
import traceback
import warnings
from typing import List, Union, overload

from sentencepiece import SentencePieceProcessor


# ---------------------------------------------------------------------------
# CJK helpers (from indextts/utils/common.py)
# ---------------------------------------------------------------------------

def tokenize_by_CJK_char(line: str, do_upper_case=True) -> str:
    CJK_RANGE_PATTERN = (
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF"
        r"\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = re.split(CJK_RANGE_PATTERN, line.strip())
    return " ".join(
        [w.strip().upper() if do_upper_case else w.strip() for w in chars if w.strip()]
    )


def de_tokenized_by_CJK_char(line: str, do_lower_case=False) -> str:
    english_word_pattern = re.compile(r"([A-Z]+(?:[\s-][A-Z-]+)*)", re.IGNORECASE)
    english_sents = english_word_pattern.findall(line)
    for i, sent in enumerate(english_sents):
        line = line.replace(sent, f"<sent_{i}>")

    words = line.split()
    sent_placeholder_pattern = re.compile(r"^.*?(<sent_(\d+)>)")
    for i in range(len(words)):
        m = sent_placeholder_pattern.match(words[i])
        if m:
            placeholder_index = int(m.group(2))
            words[i] = words[i].replace(m.group(1), english_sents[placeholder_index])
            if do_lower_case:
                words[i] = words[i].lower()
    return "".join(words)


# ---------------------------------------------------------------------------
# TextNormalizer
# ---------------------------------------------------------------------------

class TextNormalizer:
    def __init__(self):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",", "；": ",", ";": ",", "，": ",", "。": ".", "！": "!",
            "？": "?", "\n": " ", "·": "-", "、": ",", "...": "…", ",,,": "…",
            "，，，": "…", "……": "…", "\u201c": "'", "\u201d": "'", '"': "'",
            "\u2018": "'", "\u2019": "'", "（": "'", "）": "'", "(": "'",
            ")": "'", "《": "'", "》": "'", "【": "'", "】": "'", "[": "'",
            "]": "'", "—": "-", "～": "-", "~": "-", "「": "'", "」": "'",
            ":": ",",
        }
        self.zh_char_rep_map = {"$": ".", **self.char_rep_map}

    def match_email(self, email):
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = (
        r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?"
        r"(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?"
        r"|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
    )
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
    ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"

    def use_chinese(self, s):
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True
        has_pinyin = bool(re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self):
        import platform
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        if platform.system() != "Linux":
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tagger_cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, ".gitignore"), "w") as f:
                    f.write("*\n")
            self.zh_normalizer = NormalizerZh(
                cache_dir=cache_dir, remove_interjections=False,
                remove_erhua=False, overwrite_cache=False,
            )
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    def normalize(self, text: str) -> str:
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        if self.use_chinese(text):
            text = re.sub(self.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            replaced_text, pinyin_list = self.save_pinyin_tones(text.rstrip())
            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())
            result = self.restore_names(result, original_name_list)
            result = self.restore_pinyin_tones(result, pinyin_list)
            pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                text = re.sub(self.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                result = self.en_normalizer.normalize(text)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    def correct_pinyin(self, pinyin: str):
        if pinyin[0] not in "jqxJQX":
            return pinyin
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text):
        name_pattern = re.compile(self.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")
        return transformed_text, original_name_list

    def restore_names(self, normalized_text, original_name_list):
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text
        transformed_text = normalized_text
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_pinyin_tones(self, original_text):
        origin_pinyin_pattern = re.compile(self.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text
        transformed_text = normalized_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        return transformed_text


# ---------------------------------------------------------------------------
# TextTokenizer
# ---------------------------------------------------------------------------

class TextTokenizer:
    def __init__(self, vocab_file: str, normalizer: TextNormalizer = None):
        self.vocab_file = vocab_file
        self.normalizer = normalizer
        if self.vocab_file is None:
            raise ValueError("vocab_file is None")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"vocab_file {self.vocab_file} does not exist")
        if self.normalizer:
            self.normalizer.load()
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)
        self.pre_tokenizers = [tokenize_by_CJK_char]

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]: ...

    def convert_ids_to_tokens(self, ids: Union[List[int], int]):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        return self.encode(text, out_type=str)

    def encode(self, text: str, **kwargs):
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)
        if self.normalizer:
            text = self.normalizer.normalize(text)
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)

    def decode(self, ids: Union[List[int], int], do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop("out_type", str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    punctuation_marks_tokens = [".", "!", "?", "▁.", "▁?", "▁..."]

    @staticmethod
    def split_segments_by_token(
        tokenized_str: List[str],
        split_tokens: List[str],
        max_text_tokens_per_segment: int,
        quick_streaming_tokens: int = 0,
    ) -> List[List[str]]:
        if len(tokenized_str) == 0:
            return []
        segments: List[List[str]] = []
        current_segment = []
        current_segment_tokens_len = 0
        i = 0
        while i < len(tokenized_str):
            token = tokenized_str[i]
            current_segment.append(token)
            current_segment_tokens_len += 1
            if (
                not ("," in split_tokens or "▁," in split_tokens)
                and ("," in current_segment or "▁," in current_segment)
            ):
                sub_segments = TextTokenizer.split_segments_by_token(
                    current_segment, [",", "▁,"],
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    quick_streaming_tokens=quick_streaming_tokens,
                )
            elif "-" not in split_tokens and "-" in current_segment:
                sub_segments = TextTokenizer.split_segments_by_token(
                    current_segment, ["-"],
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    quick_streaming_tokens=quick_streaming_tokens,
                )
            elif current_segment_tokens_len <= max_text_tokens_per_segment:
                if token in split_tokens and current_segment_tokens_len > 2:
                    if i < len(tokenized_str) - 1:
                        if tokenized_str[i + 1] in ["'", "▁'"]:
                            current_segment.append(tokenized_str[i + 1])
                            current_segment_tokens_len += 1
                            i += 1
                    segments.append(current_segment)
                    current_segment = []
                    current_segment_tokens_len = 0
                i += 1
                continue
            else:
                sub_segments = []
                for j in range(0, len(current_segment), max_text_tokens_per_segment):
                    if j + max_text_tokens_per_segment < len(current_segment):
                        sub_segments.append(current_segment[j:j + max_text_tokens_per_segment])
                    else:
                        sub_segments.append(current_segment[j:])
                warnings.warn(
                    f"Segment exceeds limit {max_text_tokens_per_segment}, "
                    f"tokens: {current_segment}.",
                    RuntimeWarning,
                )
            segments.extend(sub_segments)
            current_segment = []
            current_segment_tokens_len = 0
            i += 1
        if current_segment_tokens_len > 0:
            assert current_segment_tokens_len <= max_text_tokens_per_segment
            segments.append(current_segment)

        merged_segments = []
        total_token = 0
        for segment in segments:
            total_token += len(segment)
            if len(segment) == 0:
                continue
            if len(merged_segments) == 0:
                merged_segments.append(segment)
            elif (
                len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment
                and total_token > quick_streaming_tokens
            ):
                merged_segments[-1] = merged_segments[-1] + segment
            elif len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment / 2:
                merged_segments[-1] = merged_segments[-1] + segment
            else:
                merged_segments.append(segment)
        return merged_segments

    def split_segments(
        self, tokenized: List[str], max_text_tokens_per_segment=120,
        quick_streaming_tokens=0,
    ) -> List[List[str]]:
        return TextTokenizer.split_segments_by_token(
            tokenized, self.punctuation_marks_tokens,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )
