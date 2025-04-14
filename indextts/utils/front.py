# -*- coding: utf-8 -*-
import traceback
import re

class TextNormalizer:
    def __init__(self):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            "……": "…",
            "$": ".",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }

    def match_email(self, email):
        # 正则表达式匹配邮箱格式：数字英文@数字英文.英文
        pattern = r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$'
        return re.match(pattern, email) is not None
    """
    匹配拼音声调格式：pinyin+数字，声调1-5，5表示轻声
    例如：xuan4, jve2, ying1, zhong4, shang5
    """
    PINYIN_TONE_PATTERN = r"([bmnpqdfghjklzcsxwy]?h?[aeiouüv]{1,2}[ng]*|ng)([1-5])"
    def use_chinese(self, s):
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', s))
        has_alpha = bool(re.search(r'[a-zA-Z]', s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        has_pinyin = bool(re.search(self.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self):
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        # sys.path.append(model_dir)
        import platform
        if platform.system() == "Darwin":
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(remove_erhua=False,lang="zh",operator="tn")
            self.en_normalizer = Normalizer(lang="en",operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            self.zh_normalizer = NormalizerZh(remove_interjections=False, remove_erhua=False,overwrite_cache=False)
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    def infer(self, text: str):
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        replaced_text, pinyin_list = self.save_pinyin_tones(text.rstrip())

        try:
            normalizer = self.zh_normalizer if self.use_chinese(replaced_text) else self.en_normalizer
            result = normalizer.normalize(replaced_text)
        except Exception:
            result = ""
            print(traceback.format_exc())
        result = self.restore_pinyin_tones(result, pinyin_list)
        pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
        result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    def correct_pinyin(self, pinyin):
        """
        将 jqx 的韵母为 u/ü 的拼音转换为 v
        如：ju -> jv , que -> qve, xün -> xvn
        """
        if pinyin[0] not in "jqx":
            return pinyin
        # 匹配 jqx 的韵母为 u/ü 的拼音
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin)
        return pinyin

    def save_pinyin_tones(self, original_text):
        """
        替换拼音声调为占位符 <pinyin_a>, <pinyin_b>, ...
        例如：xuan4 -> <pinyin_a>
        """
        # 声母韵母+声调数字
        origin_pinyin_pattern = re.compile(self.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set(''.join(p) for p in original_pinyin_list))
        transformed_text = original_text
        # 替换为占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")
            
        # print("original_text: ", original_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """
        恢复拼音中的音调数字（1-5）为原来的拼音
        例如：<pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替换为占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        # print("normalized_text: ", normalized_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text

if __name__ == '__main__':
    # 测试程序
    text_normalizer = TextNormalizer()
    text_normalizer.load()
    cases = [
        "我爱你！",
        "I love you!",
        "我爱你的英语是”I love you“",
        "2.5平方电线",
        "共465篇，约315万字",
        "2002年的第一场雪，下在了2003年",
        "速度是10km/h",
        "现在是北京时间2025年01月11日 20:00",
        "他这条裤子是2012年买的，花了200块钱",
        "电话：135-4567-8900",
        "1键3连",
        "他这条视频点赞3000+，评论1000+，收藏500+",
        "这是1024元的手机，你要吗？",
        "受不liao3你了",
        "”衣裳“不读衣chang2，而是读衣shang5",
        "最zhong4要的是：不要chong2蹈覆辙",
        "IndexTTS 正式发布1.0版本了，效果666",
        "See you at 8:00 AM",
        "8:00 AM 开会",
        "苹果于2030/1/2发布新 iPhone 2X 系列手机，最低售价仅 ¥12999",
    ]
    for case in cases:
        print(f"原始文本: {case}")
        print(f"处理后文本: {text_normalizer.infer(case)}")
        print("-" * 50)
