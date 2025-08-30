from fastapi import FastAPI, File, UploadFile, HTTPException, Form  # 添加Form导入
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import time
import argparse
from indextts.infer import IndexTTS

# 新增独立配置
parser = argparse.ArgumentParser(description="IndexTTS API")
parser.add_argument("--port", type=int, default=6008, help="API服务端口")
parser.add_argument("--host", type=str, default="127.0.0.1", help="API服务地址")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="模型目录")
api_args = parser.parse_args()

# 初始化模型
tts = IndexTTS(
    model_dir=api_args.model_dir,
    cfg_path=os.path.join(api_args.model_dir, "config.yaml")
)

COUNT = 0

# 新增模型文件检查
required_files = [
    "bigvgan_generator.pth",
    "bpe.model",
    "gpt.pth",
    "config.yaml",
]
for file in required_files:
    if not os.path.exists(os.path.join(api_args.model_dir, file)):
        raise FileNotFoundError(f"缺少必要模型文件: {file}")

app = FastAPI(title="IndexTTS API")

class TTSRequest(BaseModel):
    text: str
    infer_mode: str = "普通推理"
    max_text_tokens_per_sentence: int = 120
    sentences_bucket_max_size: int = 4
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 1.0
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 600

@app.post("/tts")
async def synthesize(
    prompt_audio: UploadFile = File(..., description="参考音频文件"),
    text: str = Form(..., example="欢迎使用语音合成接口"),
    infer_mode: str = Form("普通推理"),
    max_text_tokens_per_sentence: int = Form(120),
    sentences_bucket_max_size: int = Form(4),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    temperature: float = Form(1.0),
    length_penalty: float = Form(0.0),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(10.0),
    max_mel_tokens: int = Form(600)
):
    """单次推理接口"""
    try:
        global COUNT

        # 创建输出目录
        os.makedirs("outputs", exist_ok=True)
        
        # 保存上传的参考音频
        prompt_path = f"temp_prompt_{int(time.time())}.wav"
        with open(prompt_path, "wb") as f:
            f.write(await prompt_audio.read())
        
        # 准备输出路径 循环1000次覆盖
        COUNT += 1
        if COUNT > 1000:
            COUNT = 0
        output_path = os.path.join("outputs", f"api_output_{COUNT}.wav")

        
        # 调用生成逻辑
        kwargs = {
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k if top_k > 0 else None,
            "temperature": temperature,
            "length_penalty": length_penalty,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "max_mel_tokens": max_mel_tokens,
        }

        if infer_mode == "普通推理":
            tts.infer(
                prompt_path,
                text,
                output_path,
                max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                **kwargs
            )
        else:
            tts.infer_fast(
                prompt_path,
                text,
                output_path,
                max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                sentences_bucket_max_size=sentences_bucket_max_size,
                **kwargs
            )
        
        os.remove(prompt_path)
        return FileResponse(output_path, media_type="audio/wav")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=api_args.host, port=api_args.port)