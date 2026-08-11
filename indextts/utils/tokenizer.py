import base64
import os
from functools import lru_cache
from typing import Optional
import torch
from transformers import AutoTokenizer
from whisper.tokenizer import Tokenizer

import tiktoken

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian", 
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
    "minnan": "minnan",
    "wuyu": "wuyu",
    "dialect": "dialect",
    "zh/en": "zh/en",
    "en/zh": "en/zh",
    "common": "common",
}

# 增加 LANGUAGE_DICT 用于映射
LANGUAGE_DICT = {lang: index for index, lang in enumerate(LANGUAGES.keys())}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

AUDIO_EVENT = {
    "ASR": "ASR",
    "AED": "AED",
    "SER": "SER",
    "Speech": "Speech",
    "/Speech": "/Speech",
    "BGM": "BGM",
    "/BGM": "/BGM",
    "Laughter": "Laughter",
    "/Laughter": "/Laughter",
    "Applause": "Applause",
    "/Applause": "/Applause",
}

EMOTION = {
    "HAPPY": "HAPPY",
    "SAD": "SAD",
    "ANGRY": "ANGRY",
    "NEUTRAL": "NEUTRAL",
}

TTS_Vocal_Token = {
    "TTS/B": "TTS/B",
    "TTS/O": "TTS/O",
    "TTS/Q": "TTS/Q",
    "TTS/A": "TTS/A",
    "TTS/CO": "TTS/CO",
    "TTS/CL": "TTS/CL",
    "TTS/H": "TTS/H",
    **{f"TTS/SP{i:02d}": f"TTS/SP{i:02d}" for i in range(1, 14)}
}


def lang_to_token(lang):
    lang = lang.lower()
    if lang not in LANGUAGE_DICT:
        lang = "common"
    return LANGUAGE_DICT[lang]


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99, model_dir: str = "checkpoints"):
    vocab_path = os.path.join(model_dir, f'{name}.tiktoken')
    
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        *[f"<|{audio_event}|>" for audio_event in list(AUDIO_EVENT.keys())],
        *[f"<|{emotion}|>" for emotion in list(EMOTION.keys())],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],        # register special tokens for ASR
        *[f"<|{tts}|>" for tts in list(TTS_Vocal_Token.keys())],  # register special tokens for TTS
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1
    
    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )



class WhisperTokenizer(Tokenizer):
    """
    Whisper tokenizer 没有提供 tokenize, convert_tokens_to_ids, convert_ids_to_tokens 函数
    如果使用 encode 将 str 转为 list[int] 再单独 decode 每个 token 会丢失上下文信息，对于像日语单个字符可能需要多个token来表示
    所以无法单纯使用 token_list = [tokenizer.decode([token_id]) for token_id in token_ids] 去做 token->index 的转换
    因此这里添加了 3 个函数来做这件事
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, text, max_token_comb=4):
        """
        将输入文本根据token切分开转为list[str]
        通过智能组合token避免出现不完整的乱码字符
        """
        token_ids = self.encode(text, allowed_special="all")

        # 分组合并token以获得有意义的字符
        tokens = []
        i = 0

        while i < len(token_ids):
            # 从当前位置开始，尝试不同长度的组合
            best_token = None
            best_length = 0

            # 尝试1到4个token的组合（根据需要可以调整这个范围）
            for length in range(1, min(max_token_comb+1, len(token_ids) - i + 1)):
                try:
                    candidate_ids = token_ids[i:i+length]
                    candidate_token = self.decode(candidate_ids)
                    # 检查是否是有效token（没有乱码）
                    if "\ufffd" not in candidate_token and candidate_token.strip():
                        best_token = candidate_token
                        best_length = length
                        break  # 找到第一个有效的就停止
                except:
                    continue

            # 如果找到了有效token
            if best_token is not None:
                tokens.append(best_token)
                i += best_length
            else:
                # 如果没有找到，就使用单个token（即使可能有乱码）
                try:
                    single_token = self.decode([token_ids[i]])
                    tokens.append(single_token)
                except:
                    tokens.append("<UNK>")
                i += 1

        return tokens



@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
    model_dir: str = "checkpoints",
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual_zh_ja_yue_char_del"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = get_encoding(name=encoding_name, num_languages=num_languages, model_dir=model_dir)

    return WhisperTokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )


class QwenTokenizer():
    def __init__(self, token_path, skip_special_tokens=True):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ],
            'nonverbalspeech38k_speech_tokens': [
                '[snore]', '[throatclearing]', '[crying]',
                '[sniff]', '[laughing]', '[coughing]',
                '[gasp]', '[yawn]', '<B>', '</B>'
            ]
        }
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text


@lru_cache(maxsize=None)
def get_qwen_tokenizer(
    token_path: str,
    skip_special_tokens: bool
) -> QwenTokenizer:
    return QwenTokenizer(token_path=token_path, skip_special_tokens=skip_special_tokens)





if __name__ == "__main__":

    text_list = [
        "IndexTTS 正式发布1.0版本了，效果666",
        "晕XUAN4是一种GAN3觉",
        "我爱你！",
        "I love you!",
        "“我爱你”的英语是“I love you”",
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
        "“衣裳”不读衣chang2，而是读衣shang5",
        "最zhong4要的是：不要chong2蹈覆辙",
        "不zuo1死就不会死",
        "See you at 8:00 AM",
        "8:00 AM 开会",
        "Couting down 3, 2, 1, go!",
        "数到3就开始：1、2、3",
        "This sales for 2.5% off, only $12.5.",
        "5G网络是4G网络的升级版，2G网络是3G网络的前身",
        "苹果于2030/1/2发布新 iPhone 2X 系列手机，最低售价仅 ¥12999",
        "这酒...里...有毒...",
        # 异常case
        "只有,,,才是最好的",
        "babala2是什么？",  # babala二是什么?
        "用beta1测试",  # 用beta一测试
        "have you ever been to beta2?",  # have you ever been to beta two?
        "such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS",  # such as xtts,cosyvoice two,fish-speech,and f five-tts
        "where's the money?",  # where is the money?
        "who's there?",  # who is there?
        "which's the best?",  # which is the best?
        "how's it going?",  # how is it going?
        "今天是个好日子 it's a good day",  # 今天是个好日子 it is a good day
        # 人名
        "约瑟夫·高登-莱维特（Joseph Gordon-Levitt is an American actor）",
        "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），美国商业经理、工业工程师和工业开发商，现任苹果公司首席执行官。",
        # 长句子
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。影片剧情游走于梦境与现实之间，被定义为“发生在意识结构内的当代动作科幻片”，讲述了由莱昂纳多·迪卡普里奥扮演的造梦师，带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。",
        "清晨拉开窗帘，阳光洒在窗台的Bloomixy花艺礼盒上——薰衣草香薰蜡烛唤醒嗅觉，永生花束折射出晨露般光泽。设计师将“自然绽放美学”融入每个细节：手工陶瓷花瓶可作首饰收纳，香薰精油含依兰依兰舒缓配方。限量款附赠《365天插花灵感手册》，让每个平凡日子都有花开仪式感。\n宴会厅灯光暗下的刹那，Glimmeria星月系列耳坠开始发光——瑞士冷珐琅工艺让蓝宝石如银河流动，钛合金骨架仅3.2g无负重感。设计师秘密：内置微型重力感应器，随步伐产生0.01mm振幅，打造“行走的星光”。七夕限定礼盒含星座定制铭牌，让爱意如星辰永恒闪耀。",
        "电影1：“黑暗骑士”（演员：克里斯蒂安·贝尔、希斯·莱杰；导演：克里斯托弗·诺兰）；电影2：“盗梦空间”（演员：莱昂纳多·迪卡普里奥；导演：克里斯托弗·诺兰）；电影3：“钢琴家”（演员：艾德里安·布洛迪；导演：罗曼·波兰斯基）；电影4：“泰坦尼克号”（演员：莱昂纳多·迪卡普里奥；导演：詹姆斯·卡梅隆）；电影5：“阿凡达”（演员：萨姆·沃辛顿；导演：詹姆斯·卡梅隆）；电影6：“南方公园：大电影”（演员：马特·斯通、托马斯·艾恩格瑞；导演：特雷·帕克）",
        "そうですね、ほんと1年前、まあコロナだったので家のリビングからあの話して、すごい緊張してしまって、もう手が冷たくなったのを今でも覚えてるんですけど、新潟にいるメンバーが",
        "また、青少年健全育成などに功績がある、市内の団体を表彰する団体省令の推薦も合わせて受け付けています",
        "たねん、おんてきであるは、しかがどのにこうさんして、 しゅくんにたいしてゆみをひくとゆうことは。",
        "実は昨年、11kgの減量にも成功していたという。",
    ]
    
    from indextts.utils.common import tokenize_by_CJK_char
    tokenizer = get_tokenizer(multilingual=True)
    success_count = 0
    error_count = 0
    for raw_text in text_list:
        # print(f"raw text: {text}")
        text = tokenize_by_CJK_char(raw_text)
        # print(f"cleaned text: {text}")
        text_ja = f'<|ja|> {text}'

        # 验证 tokenize 函数
        tokens = tokenizer.tokenize(text_ja)
        ret1 = text_ja == "".join(tokens)
        print(f"tokens: {tokens}")
        # print(text_ja == "".join(tokens))
        # print(f"text_ja: {text_ja}")
        # print("tokens: ", "".join(tokens))

        # # 验证 convert_tokens_to_ids 和 convert_ids_to_tokens 函数
        # ids = tokenizer.encode(text_ja, allowed_special="all")
        # ids_to_tokens = tokenizer.convert_ids_to_tokens(ids)
        # tokens_to_ids = tokenizer.convert_tokens_to_ids(ids_to_tokens)
        # ret2 = ids == tokens_to_ids
        # print(f"raw_ids      : {ids}")
        # print(f"tokens_to_ids: {tokens_to_ids}")
        # print(ids == tokens_to_ids)
        
        # if ret1 and ret2:
        if ret1:
            print("Success:", raw_text)
            success_count += 1
        else:
            print("Error:", raw_text)
            error_count += 1
    print(f"Total Success: {success_count}, Total Error: {error_count}")