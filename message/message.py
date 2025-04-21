class WebsocketMessage:
    String = "str"
    Blob = "blob"
    Exit = "quit"
    def __init__(self, type: str):
        self.type = type
        self.text = ""
        self.blob = b''

class TTSMessage:
    String = "str"
    Exit = "quit"
    def __init__(self, type: str, content: str):
        self.type = type
        self.content = content
