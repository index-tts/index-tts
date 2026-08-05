import re
import random


class JapaneseG2PProcessor:
    """
    日语文本分词 + 平假名化处理器。
    依赖 fugashi + unidic-lite（或系统 MeCab）。
    安装: pip install fugashi unidic-lite
    """

    def __init__(self, g2p_ratio=0.2):
        self.g2p_ratio = g2p_ratio
        self._init_tagger()

    def _init_tagger(self):
        try:
            import fugashi
            self.tagger = fugashi.Tagger()
            self.backend = 'fugashi'
        except ImportError:
            try:
                import MeCab
                self.tagger = MeCab.Tagger('-Ochasen')
                self.backend = 'mecab'
            except ImportError:
                raise ImportError(
                    "请安装 fugashi: pip install fugashi unidic-lite，"
                    "或安装系统 MeCab 后 pip install mecab-python3"
                )

    def tokenize(self, text: str) -> list:
        """
        日语分词，返回 [(surface, reading_katakana), ...] 列表。
        reading 为片假名读音；若无法获取则等于 surface。
        """
        tokens = []
        if self.backend == 'fugashi':
            for token in self.tagger(text):
                surface = token.surface
                try:
                    reading = token.feature.kana
                    if not reading or reading == '*':
                        reading = surface
                except AttributeError:
                    feat = token.feature.split(',')
                    reading = feat[7] if len(feat) > 7 and feat[7] != '*' else surface
                tokens.append((surface, reading))
        else:
            for line in self.tagger.parse(text).splitlines():
                if line in ('EOS', ''):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    reading = parts[1] if parts[1] != '*' else parts[0]
                    tokens.append((surface, reading))
        return tokens

    @staticmethod
    def kata2hira(text: str) -> str:
        """片假名 → 平假名（ァ-ン → ぁ-ん）"""
        return ''.join(
            chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch
            for ch in text
        )

    @staticmethod
    def _has_kanji(text: str) -> bool:
        """判断字符串是否含有汉字"""
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)

    def _process_segment(self, text: str) -> str:
        """对单个无空格片段做汉字 token 的局部平假名替换。"""
        tokens = self.tokenize(text)
        kanji_indices = [i for i, (surface, _) in enumerate(tokens) if self._has_kanji(surface)]
        num_to_replace = int(len(kanji_indices) * self.g2p_ratio)
        if num_to_replace == 0 and kanji_indices and random.random() < self.g2p_ratio:
            num_to_replace = 1
        replace_set = set(random.sample(kanji_indices, min(num_to_replace, len(kanji_indices))))
        result = []
        for i, (surface, reading) in enumerate(tokens):
            if i in replace_set:
                hira = self.kata2hira(reading)
                result.append(hira)
            else:
                result.append(surface)
        return ' '.join(result)


    def process_ja_text(self, text: str) -> str:
        """
        日语文本分词后，对含汉字的 token 按 g2p_ratio 概率替换为
        平假名读音，其余保留原字。
        输入中原有的空格位置在输出中保留。
        """
        # 按空格拆分，保留空格位置，逐段处理后拼回
        parts = re.split(r'( +)', text)   # 奇数位为空格，偶数位为文本段
        return ''.join(
            self._process_segment(p) if p.strip() else p
            for p in parts
        )


# ----------------------------------------------------------------------
if __name__ == '__main__':
    processor = JapaneseG2PProcessor(g2p_ratio=0.5)
    test_text = 'ちょうど 探しに行こうかなって 思っていたんだ。'
    test_text = '足が長く見えるように、真ん中のエンブレムのところまで、伸ばした感じで、メインの骨格を作りました。'
    print('原文:', test_text)
    tokens = processor.tokenize(test_text)
    print('分词结果:')
    for surface, reading in tokens:
        print(f'  {surface!r:10s} → {reading!r}')
    for _ in range(3):
        print('增强:', processor.process_ja_text(test_text))
#     processor = JapaneseG2PProcessor(g2p_ratio=0)
    # f = open("./japan_label.list", 'w')
    # for x in open("./japan.list", 'r').readlines():
        # org = x.strip().split('|')[4]
        # tar = processor.process_ja_text(org)
        # f.write(f'{org},{tar}\n')
#     f.close()

