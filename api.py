"""
IndexTTS2 API
版本: v1.0.0
"""

import os
import sys
import uuid
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import argparse

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

# 解析命令行参数
parser = argparse.ArgumentParser(
    description="IndexTTS API",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
cmd_args = parser.parse_args()

# 检查模型文件
if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

# 导入IndexTTS
from indextts.infer_v2 import IndexTTS2

# 创建FastAPI应用
app = FastAPI(
    title="IndexTTS API",
    description="IndexTTS Text-to-Speech API",
    version="1.0.0"
)

# 初始化模型
tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
)

# 创建输出目录
os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

# 任务状态存储
tasks = {}

# 请求模型定义
class TTSRequest(BaseModel):
    text: str
    prompt_audio: str = "tests/sample_prompt.wav"
    return_audio: bool = False
    emo_control_method: int = 0  # 0: same as speaker, 1: from reference audio, 2: from vectors, 3: from text
    emo_ref_path: Optional[str] = None
    emo_weight: float = 0.65
    emo_text: Optional[str] = None
    emo_vec: Optional[List[float]] = None
    emo_random: bool = False
    max_text_tokens_per_segment: int = 120
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 0.8
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500

# 任务状态模型
class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    message: str
    result_path: Optional[str]

def process_tts_task(task_id: str, request: TTSRequest):
    """
    处理TTS任务的后台函数
    """
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Task is being processed"
        
        # 设置输出路径
        output_path = os.path.join("outputs/tasks", f"{task_id}.wav")
        
        # 准备参数
        kwargs = {
            "do_sample": request.do_sample,
            "top_p": request.top_p,
            "top_k": request.top_k if request.top_k > 0 else None,
            "temperature": request.temperature,
            "length_penalty": request.length_penalty,
            "num_beams": request.num_beams,
            "repetition_penalty": request.repetition_penalty,
            "max_mel_tokens": request.max_mel_tokens,
        }
        
        # 处理情感控制参数
        emo_vector = None
        if request.emo_control_method == 2 and request.emo_vec:
            emo_vector = tts.normalize_emo_vec(request.emo_vec, apply_bias=True)
        
        if request.emo_text == "":
            request.emo_text = None
        
        # 调用TTS推理
        tts.infer(
            spk_audio_prompt=request.prompt_audio,
            text=request.text,
            output_path=output_path,
            emo_audio_prompt=request.emo_ref_path,
            emo_alpha=request.emo_weight,
            emo_vector=emo_vector,
            use_emo_text=(request.emo_control_method == 3),
            emo_text=request.emo_text,
            use_random=request.emo_random,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
            **kwargs
        )
        
        # 更新任务状态
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = "Task completed successfully"
        tasks[task_id]["result_path"] = output_path
        
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Task failed: {str(e)}"
        tasks[task_id]["result_path"] = None

@app.post("/api/v1/tts/tasks", response_model=TaskStatus)
async def create_tts_task(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    创建一个新的TTS任务，支持异步处理和同步返回音频两种模式
    """
    # 检查参考音频文件是否存在
    if not os.path.exists(request.prompt_audio):
        raise HTTPException(status_code=400, detail="Prompt audio file does not exist")
    
    if request.emo_ref_path and not os.path.exists(request.emo_ref_path):
        raise HTTPException(status_code=400, detail="Emotion reference audio file does not exist")
    
    # 创建任务ID
    task_id = str(uuid.uuid4())
    
    # 如果需要直接返回音频
    if request.return_audio:
        try:
            output_path = os.path.join("outputs/tasks", f"{task_id}.wav")
            
            # 准备参数
            kwargs = {
                "do_sample": request.do_sample,
                "top_p": request.top_p,
                "top_k": request.top_k if request.top_k > 0 else None,
                "temperature": request.temperature,
                "length_penalty": request.length_penalty,
                "num_beams": request.num_beams,
                "repetition_penalty": request.repetition_penalty,
                "max_mel_tokens": request.max_mel_tokens,
            }
            
            # 处理情感控制参数
            emo_vector = None
            if request.emo_control_method == 2 and request.emo_vec:
                emo_vector = tts.normalize_emo_vec(request.emo_vec, apply_bias=True)
            
            if request.emo_text == "":
                request.emo_text = None
            
            # 调用TTS推理
            tts.infer(
                spk_audio_prompt=request.prompt_audio,
                text=request.text,
                output_path=output_path,
                emo_audio_prompt=request.emo_ref_path,
                emo_alpha=request.emo_weight,
                emo_vector=emo_vector,
                use_emo_text=(request.emo_control_method == 3),
                emo_text=request.emo_text,
                use_random=request.emo_random,
                max_text_tokens_per_segment=request.max_text_tokens_per_segment,
                **kwargs
            )
            
            # 返回音频文件
            return FileResponse(output_path, media_type='audio/wav', filename=f"{task_id}.wav")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")
    
    # 异步处理模式
    tasks[task_id] = {
        "status": "pending",
        "message": "Task created, waiting to be processed",
        "result_path": None
    }
    
    # 在后台处理任务
    background_tasks.add_task(process_tts_task, task_id, request)
    
    return TaskStatus(
        task_id=task_id,
        status=tasks[task_id]["status"],
        message=tasks[task_id]["message"],
        result_path=tasks[task_id]["result_path"]
    )

@app.get("/api/v1/tts/tasks", response_class=FileResponse)
async def create_tts_task_sync(text: str, prompt_audio: str = "tests/sample_prompt.wav"):
    """
    通过GET请求创建TTS任务并直接返回音频文件
    """
    # 检查参考音频文件是否存在
    if not os.path.exists(prompt_audio):
        raise HTTPException(status_code=400, detail="Prompt audio file does not exist")
    
    # 创建任务ID
    task_id = str(uuid.uuid4())
    output_path = os.path.join("outputs/tasks", f"{task_id}.wav")
    
    try:
        # 使用线程池执行器设置超时时间
        import concurrent.futures
        import functools
        
        # 创建带超时的推理函数
        infer_with_timeout = functools.partial(
            tts.infer,
            spk_audio_prompt=prompt_audio,
            text=text,
            output_path=output_path
        )
        
        # 使用线程池执行，设置120秒超时
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(infer_with_timeout)
            future.result(timeout=120)  # 120秒超时限制
        
        # 返回音频文件
        return FileResponse(output_path, media_type='audio/wav', filename=f"{task_id}.wav")
        
    except concurrent.futures.TimeoutError:
        # 清理可能已创建的文件
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=408, detail="Task processing timeout (120 seconds)")
    except Exception as e:
        # 清理可能已创建的文件
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

@app.get("/api/v1/tts/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    获取指定任务的当前状态
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        result_path=task["result_path"]
    )

@app.get("/api/v1/tts/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """
    获取已完成任务的音频结果
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed yet. Current status: {task['status']}")
    
    if not task["result_path"] or not os.path.exists(task["result_path"]):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(task["result_path"], media_type='audio/wav', filename=f"{task_id}.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port)