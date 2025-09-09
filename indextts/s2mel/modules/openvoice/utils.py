import re
import json
import numpy as np


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def string_to_bits(string, pad_len=8):
    # Convert each character to its ASCII value
    ascii_values = [ord(char) for char in string]
    
    # Convert ASCII values to binary representation
    binary_values = [bin(value)[2:].zfill(8) for value in ascii_values]
    
    # Convert binary strings to integer arrays
    bit_arrays = [[int(bit) for bit in binary] for binary in binary_values]
    
    # Convert list of arrays to NumPy array
    numpy_array = np.array(bit_arrays)
    numpy_array_full = np.zeros((pad_len, 8), dtype=numpy_array.dtype)
    numpy_array_full[:, 2] = 1
    max_len = min(pad_len, len(numpy_array))
    numpy_array_full[:max_len] = numpy_array[:max_len]
    return numpy_array_full


def bits_to_string(bits_array):
    # Convert each row of the array to a binary string
    binary_values = [''.join(str(bit) for bit in row) for row in bits_array]
    
    # Convert binary strings to ASCII values
    ascii_values = [int(binary, 2) for binary in binary_values]
    
    # Convert ASCII values to characters
    output_string = ''.join(chr(value) for value in ascii_values)
    
    return output_string


def split_segment(text, min_len=10, language_str='[EN]'):
    if language_str in ['EN']:
        segments = split_segments_latin(text, min_len=min_len)
    else:
        segments = split_segments_zh(text, min_len=min_len)
    return segments

def split_segments_latin(text, min_len=10):
    """Split Long sentences into list of short segments.

    Args:
        str: Input sentences.

    Returns:
        List[str]: list of output segments.
    """
    # deal with dirty text characters
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    text = re.sub('[“”]', '"', text)
    text = re.sub('[‘’]', "'", text)
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    text = re.sub('[\n\t ]+', ' ', text)
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    # split
    segments = [s.strip() for s in text.split('$#!')]
    if len(segments[-1]) == 0: del segments[-1]

    new_segments = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(segments):
        # print(sent)
        new_sent.append(sent)
        count_len += len(sent.split(" "))
        if count_len > min_len or ind == len(segments) - 1:
            count_len = 0
            new_segments.append(' '.join(new_sent))
            new_sent = []
    return merge_short_segments_latin(new_segments)


def merge_short_segments_latin(sens):
    """Avoid short segments by merging them with the following segment.

    Args:
        List[str]: list of input segments.

    Returns:
        List[str]: list of output segments.
    """
    sens_out = []
    for s in sens:
        # If the previous segment is too short, merge them with
        # the current segment.
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out

def split_segments_zh(text, min_len=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    # 将文本中的换行符、空格和制表符替换为空格
    text = re.sub('[\n\t ]+', ' ', text)
    # 在标点符号后添加一个空格
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    # 分隔句子并去除前后空格
    # segments = [s.strip() for s in re.split('(。|！|？|；)', text)]
    segments = [s.strip() for s in text.split('$#!')]
    if len(segments[-1]) == 0: del segments[-1]

    new_segments = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(segments):
        new_sent.append(sent)
        count_len += len(sent)
        if count_len > min_len or ind == len(segments) - 1:
            count_len = 0
            new_segments.append(' '.join(new_sent))
            new_sent = []
    return merge_short_segments_zh(new_segments)


def merge_short_segments_zh(sens):
    # return sens
    """Avoid short segments by merging them with the following segment.

    Args:
        List[str]: list of input segments.

    Returns:
        List[str]: list of output segments.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current segment.
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out