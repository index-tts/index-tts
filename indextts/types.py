import time

class PromptCache:
    def __init__(self, audio_prompt, cond_mel):
        self.audio_prompt = audio_prompt
        self.cond_mel = cond_mel
        self.used_at = time.time()
