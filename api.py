import os
import re
import json
import asyncio
from bs4 import BeautifulSoup
from markdown import markdown
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ollama import chat
from indextts.infer import IndexTTS
from queue import SimpleQueue
from threading import Thread

from message.message import TTSMessage, WebsocketMessage

# 匹配标点符号（包括中英文标点）
pattern = r'(?<=[.!?;。！？；])\s*'
audioPrompt = "/home/zhuangwj/index-tts/prompts/"
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <label for="voice-select">选择音色:</label>
            <select name="voice" id="voice-select">
                <option value="">--请选择音色--</option>
                <option value="Default_Speaker.wav">默认音色(男)</option>
                <option value="Default_Yueyunpeng.wav">岳云鹏</option>
                <option value="Default_Trump.wav">特朗普</option>
                <option value="Default_Musk.wav">马斯克</option>
                <option value="Default_Leijun.mp3">雷军</option>
                <option value="Default_Liuyifei.wav">刘亦菲</option>
                <option value="Default_Wangou.wav">王鸥</option>
            </select>
            <label for="messageText">输入对话内容:</label>
            <input type="text" id="messageText" autocomplete="off" value="你是谁?你知道什么是黑洞吗?" />
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var schema = document.location.protocol === "http:" ? "ws://" : "wss://"
            var ws = new WebSocket(schema+document.location.host+"/ws");
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            let nextTime = 0;
            let queue = [];
            let isProcessing = false;
            let started = false;
            var messages = document.getElementById('messages')
            var content = null
            // 启动音频（在用户交互后调用，以激活 AudioContext）
            function startAudio() {
                if (audioContext.state === 'suspended') {
                    audioContext.resume();
                }
                nextTime = audioContext.currentTime + 0.05;
            }
            // 3. 队列处理
            async function processQueue() {
                isProcessing = true;
                while (queue.length > 0) {
                    const buffer = queue.shift();
                    try {
                    // 解码音频数据
                    const audioBuffer = await audioContext.decodeAudioData(buffer);
                    // 如果 playback 落后太多，重置 nextTime
                    if (audioContext.currentTime > nextTime) {
                        nextTime = audioContext.currentTime + 0.05;
                    }
                    // 创建并调度播放
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);
                    source.start(nextTime);
                    // 更新下一个播放时间点
                    nextTime += audioBuffer.duration;
                    } catch (err) {
                    console.error('解码失败', err);
                    }
                }
                isProcessing = false;
            }
            ws.onmessage = async function(event) {
                if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
                    let arrayBuffer = await (event.data instanceof Blob 
                        ? event.data.arrayBuffer() 
                        : event.data);
                    queue.push(arrayBuffer);
                    if (!isProcessing) {
                        processQueue();
                    }
                } else if (content) {
                    content.textContent += event.data
                }
            };
            function sendMessage(event) {
                if (!started) {
                    startAudio();
                }
                var input = document.getElementById("messageText")
                var voice = document.getElementById('voice-select').value
                ws.send(JSON.stringify({
                    text: input.value,
                    voice: voice,
                    lang: 'zh-CN'
                }))
                input.value = ''
                event.preventDefault()
                var message = document.createElement('li')
                content = document.createTextNode('')
                message.appendChild(content)
                messages.appendChild(message)
            }
        </script>
    </body>
</html>
"""

def has_punctuation(text):
    # 定义需要检测的标点集合（中英文组合）
    punctuations = {'.', '!', '?', ';', '。', '！', '？', '；'}
    return any(char in punctuations for char in text)

def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)
    print(f"html {html}")
    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))
    text = text.replace('* **', '')
    text = text.replace('**', '')
    text = text.replace('*   ', '')
    text = text.replace('#', '')
    text = text.strip()

    return text

async def sendMessageToWebsocket(queue: SimpleQueue, websocket: WebSocket):
    while True:
        msg:WebsocketMessage = queue.get(True, 1000)
        if (msg.type == WebsocketMessage.String):
            print(msg.text, end='', flush=True)
            await websocket.send_text(msg.text)
            await asyncio.sleep(0)
        elif (msg.type == WebsocketMessage.Blob):
            await websocket.send_bytes(msg.blob)
            await asyncio.sleep(0)
        else:
            return

def ttsInfer(prompt: str, queue: SimpleQueue, wsQueue: SimpleQueue):
    while True:
        msg: TTSMessage = queue.get(True, 1000)
        if (msg.type == TTSMessage.String):
            tts.inferToWebsocket(prompt, msg.content, wsQueue)
        else:
            print("ttsInfer thread finished and exit")
            return

def between_callback(queue: SimpleQueue, websocket: WebSocket):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(sendMessageToWebsocket(queue, websocket))
    loop.close()
    print("websocket thread finished and exit")

@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    wsQueue = SimpleQueue()
    wsThread = Thread(target=between_callback, args=[wsQueue, websocket])
    wsThread.start()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                chatMsg = json.loads(data)
                print(f"Chat Message: {chatMsg}")
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
                websocket.close()
                return
            stream = chat(
                model='gemma3:27b',
                messages=[
                    {
                        'role': 'user',
                        'content': chatMsg['text'],
                    },
                ],
                stream=True,
            )
            # or access fields directly from the response object
            ttsQueue = SimpleQueue()
            promptFile = os.path.join(audioPrompt, chatMsg['voice'])
            ttsThread = Thread(target=ttsInfer, args=[promptFile, ttsQueue, wsQueue])
            ttsThread.start()
            text = ""
            inferStarted = False
            chunkCount = 0
            for chunk in stream:
                wsMsg = WebsocketMessage(WebsocketMessage.String)
                wsMsg.text = chunk.message.content
                wsQueue.put(wsMsg)
                await asyncio.sleep(0)
                # print(chunk.message.content, end='', flush=True)
                chunkCount += 1
                text += chunk.message.content
                if has_punctuation(text):
                    ttsText = ""
                    sentences = re.split(pattern, markdown_to_text(text))
                    text = ""
                    for sent in sentences:
                        if has_punctuation(sent):
                            ttsText += sent
                        else:
                            text += sent
                    if ttsText != "":
                        print(f"TTS Text={ttsText}")
                        ttsMsg = TTSMessage(TTSMessage.String, ttsText)
                        ttsQueue.put(ttsMsg)
                        await asyncio.sleep(0)
                    print(f"Remain Text={text}")
                if not inferStarted and chunkCount >= 3:
                        ttsText = text
                        text = ""
                        inferStarted = True
                        ttsMsg = TTSMessage(TTSMessage.String, markdown_to_text(ttsText))
                        ttsQueue.put(ttsMsg)
                        await asyncio.sleep(0)
            if text != "":
                ttsMsg = TTSMessage(TTSMessage.String, markdown_to_text(text))
                ttsQueue.put(ttsMsg)
                await asyncio.sleep(0)
            ttsMsg = TTSMessage(TTSMessage.Exit, "")
            ttsQueue.put(ttsMsg)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        wsMsg = WebsocketMessage(WebsocketMessage.Exit)
        wsQueue.put(wsMsg)
