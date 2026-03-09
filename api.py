import argparse
import hashlib
import json
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Generator, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 动态添加到系统路径
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.extend([str(CURRENT_DIR), str(CURRENT_DIR / "indextts")])

from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

# ================= 1. 配置与初始化 =================

parser = argparse.ArgumentParser(description="IndexTTS FastAPI Server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
parser.add_argument("--port", type=int, default=8001, help="Port to run FastAPI server on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run FastAPI server on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", help="Use FP16 for inference")
parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed")
parser.add_argument("--cuda_kernel", action="store_true", help="Use CUDA kernel")
parser.add_argument("--fa2", action="store_true", help="Use Flash Attention 2")
parser.add_argument("--compile", action="store_true", help="Use Torch Compile")
args = parser.parse_args()

# 核心常量与目录结构初始化
MODEL_DIR = Path(args.model_dir)
TEMP_DIR = Path("temp_audio")
OUTPUT_DIR = Path("outputs/tasks")
PROMPTS_DIR = Path("prompts")

for d in (TEMP_DIR, OUTPUT_DIR, PROMPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

VOICE_MAP = {
    "female_cute": str(PROMPTS_DIR / "female_cute.wav"),
}

# 校验模型文件
REQUIRED_FILES = ["bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]
if not MODEL_DIR.exists() or any(not (MODEL_DIR / f).exists() for f in REQUIRED_FILES):
    sys.exit(f"Error: Model files missing in {MODEL_DIR}. Please check required files: {REQUIRED_FILES}")

# 全局组件
mutex = threading.Lock()
i18n = I18nAuto(language="Auto")
tts = IndexTTS2(
    model_dir=str(MODEL_DIR),
    cfg_path=str(MODEL_DIR / "config.yaml"),
    use_fp16=args.fp16,
    use_deepspeed=args.deepspeed,
    use_cuda_kernel=args.cuda_kernel,
    use_accel=args.fa2,
    use_torch_compile=args.compile
)

app = FastAPI(title="IndexTTS2 API", description="IndexTTS2 Text-to-Speech Fast API Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ================= 2. 核心辅助函数 =================

def save_temp_file(upload_file: Optional[UploadFile]) -> Optional[str]:
    """保存并去重上传文件（基于 MD5 哈希）"""
    if not upload_file or not upload_file.filename:
        return None
        
    content = upload_file.file.read()
    if not content:
        return None
        
    file_hash = hashlib.md5(content).hexdigest()
    suffix = Path(upload_file.filename).suffix or ".wav"
    file_path = TEMP_DIR / f"{file_hash}{suffix}"
    
    with mutex:
        if file_path.exists():
            file_path.touch()  # 更新访问时间，防止被清理脚本误删
        else:
            file_path.write_bytes(content)
            
    return str(file_path)

def tensor_to_pcm(wav_tensor: torch.Tensor) -> bytes:
    """Torch音频张量转原生PCM字节"""
    pcm_tensor = torch.clamp(wav_tensor, -32767.0, 32767.0).to(torch.int16)
    return pcm_tensor.cpu().squeeze().numpy().tobytes()

def get_latest_voices():
    """实时扫描 prompts 目录并读取 voices.json"""
    voice_map = {}
    voices_info = {}
    
    # 1. 尝试读取 metadata
    json_path = PROMPTS_DIR / "voices.json"
    metadata = {}
    if json_path.exists():
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception: 
            pass # 优雅忽略格式错误的 JSON

    # 2. 扫描音频文件并组装数据
    for wav_path in PROMPTS_DIR.glob("*.wav"):
        v_id = wav_path.stem
        voice_map[v_id] = str(wav_path)
        
        meta = metadata.get(v_id, {})
        voices_info[v_id] = {
            "id": v_id,
            "name": meta.get("name", f"{v_id}"),
            "description": meta.get("description", f"自动加载的音色: {v_id}"),
            "preview_audio_url": f"http://{args.host}:{args.port}/tts/audio/preview/{v_id}"
        }
    return voice_map, voices_info

def get_audio_response(result: Union[str, Generator], is_stream: bool, prefix: str = "audio") -> StreamingResponse:
    """统一音频返回响应"""
    if is_stream:
        return StreamingResponse(
            result, 
            media_type="audio/pcm", 
            headers={"Transfer-Encoding": "chunked"}
        )
        
    def iter_file():
        with open(result, "rb") as f:
            yield from f
            
    filename = f"{prefix}_{int(time.time())}.wav"
    return StreamingResponse(
        iter_file(), 
        media_type="audio/wav", 
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

def tts_core_process(
    prompt_audio_path: str, text: str, output_path: Optional[str], stream_return: bool = False,
    emo_control_method: int = 0, emo_ref_path: Optional[str] = None, emo_weight: float = 0.65,
    emo_text: str = "", emo_vecs: list = None, emo_random: bool = False,
    interval_silence: int = 200, max_text_tokens_per_segment: int = 120, **kwargs
) -> Union[str, Generator]:
    """TTS 核心调度引擎"""
    with mutex:
        vec = tts.normalize_emo_vec(emo_vecs, apply_bias=True) if emo_control_method == 2 and emo_vecs else None
        emo_ref = emo_ref_path if emo_control_method in (1, 2) else None
        
        gen_kwargs = {
            "spk_audio_prompt": prompt_audio_path, "text": text, "output_path": output_path,
            "emo_audio_prompt": emo_ref, "emo_alpha": emo_weight, "emo_vector": vec,
            "use_emo_text": (emo_control_method == 3), "emo_text": emo_text.strip() or None,
            "use_random": emo_random, "interval_silence": interval_silence,
            "verbose": args.verbose, "max_text_tokens_per_segment": int(max_text_tokens_per_segment),
            "stream_return": stream_return, **kwargs
        }

        try:
            if not stream_return:
                return tts.infer(**gen_kwargs)

            def audio_generator():
                for segment in tts.infer_generator(**gen_kwargs):
                    if isinstance(segment, torch.Tensor):
                        yield tensor_to_pcm(segment)
            return audio_generator()
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"语音生成失败: {str(e)}")


# ================= 3. 路由与 API =================

class SpeechRequest(BaseModel):
    model: str = "IndexTTS-2"
    input: str
    voice: str
    emo_control_method: int = 0
    emo_weight: float = 0.65
    stream_return: bool = False


@app.post("/tts/generate")
async def tts_generate(
    text: str = Form(...), prompt_audio: UploadFile = File(...), stream_return: bool = Form(False),
    interval_silence: int = Form(200), emo_control_method: int = Form(0), emo_ref_audio: UploadFile = File(None),
    emo_weight: float = Form(0.65), emo_text: str = Form(""),
    emo_vec1: float = Form(0.0), emo_vec2: float = Form(0.0), emo_vec3: float = Form(0.0), emo_vec4: float = Form(0.0),
    emo_vec5: float = Form(0.0), emo_vec6: float = Form(0.0), emo_vec7: float = Form(0.0), emo_vec8: float = Form(0.0),
    emo_random: bool = Form(False), max_text_tokens_per_segment: int = Form(120),
    do_sample: bool = Form(True), top_p: float = Form(0.8), top_k: int = Form(30),
    temperature: float = Form(0.8), length_penalty: float = Form(0.0), num_beams: int = Form(3),
    repetition_penalty: float = Form(10.0), max_mel_tokens: int = Form(1500)
):
    """标准表单生成接口（支持全量参数）"""
    prompt_path = save_temp_file(prompt_audio)
    emo_path = save_temp_file(emo_ref_audio)
    out_path = str(OUTPUT_DIR / f"spk_{int(time.time())}.wav") if not stream_return else None

    result = tts_core_process(
        prompt_audio_path=prompt_path, text=text, output_path=out_path, stream_return=stream_return,
        emo_control_method=emo_control_method, emo_ref_path=emo_path, emo_weight=emo_weight, emo_text=emo_text,
        emo_vecs=[emo_vec1, emo_vec2, emo_vec3, emo_vec4, emo_vec5, emo_vec6, emo_vec7, emo_vec8],
        emo_random=emo_random, interval_silence=interval_silence, max_text_tokens_per_segment=max_text_tokens_per_segment,
        do_sample=do_sample, top_p=top_p, top_k=top_k if top_k > 0 else None, temperature=temperature,
        length_penalty=length_penalty, num_beams=num_beams, repetition_penalty=repetition_penalty, max_mel_tokens=max_mel_tokens
    )
    return get_audio_response(result, stream_return, prefix="tts")


@app.post("/tts/audio/speech")
async def tts_audio_speech(request: SpeechRequest):
    """适配前端的 JSON 标准接口 (实时匹配音色)"""
    # 实时获取最新的映射关系
    voice_map, _ = get_latest_voices()
    
    if request.voice not in voice_map:
        raise HTTPException(status_code=400, detail=f"未找到音色: {request.voice}")
        
    prompt_path = voice_map[request.voice]
    out_path = str(OUTPUT_DIR / f"speech_{int(time.time())}.wav") if not request.stream_return else None

    result = tts_core_process(
        prompt_audio_path=prompt_path, text=request.input, output_path=out_path, 
        stream_return=request.stream_return, emo_control_method=request.emo_control_method, emo_weight=request.emo_weight
    )
    return get_audio_response(result, request.stream_return, prefix="speech")


@app.get("/tts/info")
async def get_tts_info():
    """获取模型基础信息"""
    return {
        "model_version": getattr(tts, "model_version", "1.0"),
        "max_mel_tokens": getattr(tts, "cfg", {}).gpt.max_mel_tokens if hasattr(tts, "cfg") else 1500,
        "support_stream_return": True,
        "supported_emotion_modes": [
            {"id": 0, "name": "与音色参考音频相同"}, {"id": 1, "name": "使用情感参考音频"},
            {"id": 2, "name": "使用情感向量控制"}, {"id": 3, "name": "使用情感描述文本控制"}
        ]
    }


@app.get("/tts/audio/voices")
async def list_voices():
    """请求时实时返回最新的音色列表"""
    _, voices_info = get_latest_voices()
    return voices_info


@app.get("/tts/audio/preview/{voice_id}")
async def preview_voice(voice_id: str):
    """实时检查并返回音色试听"""
    voice_map, _ = get_latest_voices()
    if voice_id not in voice_map:
        raise HTTPException(status_code=404, detail="音频文件已不存在")
    
    return get_audio_response(voice_map[voice_id], is_stream=False, prefix=voice_id)


@app.get("/tts/models")
async def list_models():
    """返回可用模型信息"""
    return [{"id": "IndexTTS-2", "name": "IndexTTS-2", "provider": "index-tts-2", "contextLength": 0}]


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="info" if args.verbose else "warning")