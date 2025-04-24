from queue import Queue
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import SampleDecoderOnlyOutput


class TokenStreamer(BaseStreamer):
    """
    流式输出token的生成器，用于收集生成过程中的token。
    参数:
        callback (`Callable`, 可选):
            每当有新的token生成时被调用的回调函数
        skip_prompt (`bool`, 默认为 `True`):
            是否跳过提示部分，只输出新生成的部分
        start_mel_token (int):
            开始的Mel标记，用于标识有效序列的开始
        stop_mel_token (int):
            结束的Mel标记，用于标识生成结束
    """

    def __init__(
        self,
        callback=None,
        skip_prompt=True,
        start_mel_token=None,
        stop_mel_token=None,
    ):
        super().__init__()
        self.callback = callback
        self.skip_prompt = skip_prompt
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.started = False
        self.ended = False
        self.tokens_queue = Queue()
        self.tokens_buffer = []  # 存储不包含start_mel_token的有效token
        self.all_tokens = []  # 存储所有token，包括start_mel_token
        self.stop_reason = None
        self.mel_token_detected = False  # 标记是否检测到mel_token

    def put(self, value):
        """向队列添加新生成的token，但跳过包含start_mel_token的token"""
        if isinstance(value, SampleDecoderOnlyOutput):
            # Get the token ids from the model output
            new_tokens = value.sequences[:, -1].detach().cpu().tolist()
        else:
            new_tokens = value

        if isinstance(new_tokens, int):
            new_tokens = [new_tokens]

        # 检查new_tokens中是否包含start_mel_token
        if self.start_mel_token is not None and self.start_mel_token in new_tokens:
            # 标记已检测到start_mel_token
            self.mel_token_detected = True
            # 只记录到all_tokens中，但不放入buffer和queue
            self.all_tokens.append(new_tokens)
            if self.callback:
                # 可以选择通知callback这是start_mel_token，但标记为不处理
                self.callback(new_tokens, is_end=False, is_start_mel=True)
            return

        # 正常处理不包含start_mel_token的token
        self.tokens_buffer.append(new_tokens)
        self.all_tokens.append(new_tokens)
        self.tokens_queue.put(new_tokens)
        if self.callback:
            self.callback(new_tokens, is_end=False)

    def end(self):
        """结束流式输出，清理资源"""
        if not self.ended and self.callback:
            self.callback(self.tokens_buffer, is_end=True)
        self.ended = True
        # 向队列添加结束标志
        self.tokens_queue.put(None)
        # 重置mel_token检测状态
        self.mel_token_detected = False

    def __iter__(self):
        """使TokenStreamer成为一个迭代器对象"""
        return self

    def __next__(self):
        """实现迭代器的下一个方法，从队列获取token"""
        token = self.tokens_queue.get()
        if token is None:
            raise StopIteration
        return token

    def get_tokens(self):
        """获取所有收集的token（不包含start_mel_token）"""
        return self.tokens_buffer

    def get_all_tokens(self):
        """获取所有token，包括start_mel_token"""
        return self.all_tokens

    def has_detected_mel_token(self):
        """检查是否已经检测到start_mel_token"""
        return self.mel_token_detected
