from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, JSONResponse
import os
import time
import uvicorn
from indextts.infer import IndexTTS
import tempfile
import hashlib
from typing import Dict
import base64
from pydantic import BaseModel

app = FastAPI(title="IndexTTS API")

# 初始化TTS模型
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")

# 确保prompts目录存在
os.makedirs("prompts", exist_ok=True)

@app.post("/upload_audio")
async def upload_audio(audio: UploadFile = File(...)) -> Dict[str, str]:
    """
    上传音频文件并保存
    
    参数:
    - audio: 音频文件
    
    返回:
    - 包含保存的文件名的JSON响应
    """
    # 读取文件内容并计算MD5
    content = await audio.read()
    md5_hash = hashlib.md5(content).hexdigest()
    
    # 获取文件扩展名
    file_extension = os.path.splitext(audio.filename)[1]
    if not file_extension:
        file_extension = ".wav"  # 默认扩展名
    
    # 生成新的文件名
    new_filename = f"{md5_hash}{file_extension}"
    save_path = os.path.join("prompts", new_filename)
    
    # 保存文件
    with open(save_path, "wb") as f:
        f.write(content)
    
    return {"filename": new_filename}

@app.post("/synthesize")
async def synthesize_speech(
    prompt_audio: UploadFile = File(...),
    text: str = Form(...),
    infer_mode: str = Form("普通推理")
):
    """
    合成语音API
    
    参数:
    - prompt_audio: 参考音频文件
    - text: 要合成的文本
    - infer_mode: 推理模式 ("普通推理" 或 "批次推理")
    
    返回:
    - JSON响应，包含音频文件的base64编码
    """
    # 创建临时文件保存上传的音频
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        content = await prompt_audio.read()
        temp_audio.write(content)
        temp_audio_path = temp_audio.name
    
    # 生成输出文件路径
    output_path = os.path.join("outputs", f"api_synth_{int(time.time())}.wav")
    
    try:
        # 调用TTS模型进行合成
        if infer_mode == "普通推理":
            output = tts.infer(temp_audio_path, text, output_path)
        else:
            output = tts.infer_fast(temp_audio_path, text, output_path)
        
        # 读取生成的音频文件并转换为base64
        with open(output, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        
        return JSONResponse(
            content={
                "code": 0,
                "message": "语音合成成功",
                "audio": audio_base64
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 1,
                "message": f"合成失败: {str(e)}"
            }
        )
    finally:
        # 清理临时文件
        os.unlink(temp_audio_path)

class SynthesizeRequest(BaseModel):
    filename: str
    text: str
    infer_mode: str = "普通推理"

@app.post("/synthesize_by_filename")
async def synthesize_speech_by_filename(request: SynthesizeRequest = Body(...)):
    """
    使用已上传的音频文件进行语音合成
    
    参数:
    - request: 包含filename、text和infer_mode的请求体
    
    返回:
    - JSON响应，包含音频文件的base64编码
    """
    # 构建音频文件完整路径
    prompt_audio_path = os.path.join("prompts", request.filename)
    
    # 检查文件是否存在
    if not os.path.exists(prompt_audio_path):
        return JSONResponse(
            status_code=404,
            content={
                "code": 2,
                "message": f"音频文件 {request.filename} 不存在"
            }
        )
    
    # 生成输出文件路径
    output_path = os.path.join("outputs", f"api_synth_{int(time.time())}.wav")
    
    try:
        # 调用TTS模型进行合成
        if request.infer_mode == "普通推理":
            output = tts.infer(prompt_audio_path, request.text, output_path)
        else:
            output = tts.infer_fast(prompt_audio_path, request.text, output_path)
        
        # 读取生成的音频文件并转换为base64
        with open(output, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        
        return JSONResponse(
            content={
                "code": 0,
                "message": "语音合成成功",
                "audio": audio_base64
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 1,
                "message": f"合成失败: {str(e)}"
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 