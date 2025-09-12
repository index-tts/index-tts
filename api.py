"""
IndexTTS API
版本: v1.0.0
作者：鸦无量（https://space.bilibili.com/2838092）
具体用法请参考`tests/api_test.py`。
"""

import os
import sys
import time
import uuid
import json
import threading
import asyncio
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from indextts.infer import IndexTTS

app = FastAPI(title="IndexTTS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建任务存储目录
os.makedirs("outputs/tasks", exist_ok=True)

# 初始化模型
tts = IndexTTS(
    model_dir="checkpoints",
    cfg_path=os.path.join("checkpoints", "config.yaml")
)

# 任务状态存储
tasks: Dict[str, Dict[str, Any]] = {}

class TTSRequest(BaseModel):
    prompt_audio: str  # 参考音频路径
    text: str  # 合成文本
    infer_mode: str = "批次推理"  # 推理模式: "普通推理" 或 "批次推理"
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
    return_audio: bool = False  # 默认为False，为True时直接返回音频文件

class TTSQueryParams(BaseModel):
    prompt_audio: str = Query(...)
    text: str = Query(...)
    infer_mode: str = Query("批次推理")
    max_text_tokens_per_sentence: int = Query(120)
    sentences_bucket_max_size: int = Query(4)
    do_sample: bool = Query(True)
    top_p: float = Query(0.8)
    top_k: int = Query(30)
    temperature: float = Query(1.0)
    length_penalty: float = Query(0.0)
    num_beams: int = Query(3)
    repetition_penalty: float = Query(10.0)
    max_mel_tokens: int = Query(600)

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: Optional[str] = None
    result_path: Optional[str] = None

def process_tts_task(task_id: str, request: TTSRequest):
    """
    处理TTS任务的后台函数
    """
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Processing TTS task"
        
        # 设置输出路径
        output_path = os.path.join("outputs/tasks", f"{task_id}.wav")
        
        # 构建参数
        kwargs = {
            "do_sample": bool(request.do_sample),
            "top_p": float(request.top_p),
            "top_k": int(request.top_k) if int(request.top_k) > 0 else None,
            "temperature": float(request.temperature),
            "length_penalty": float(request.length_penalty),
            "num_beams": request.num_beams,
            "repetition_penalty": float(request.repetition_penalty),
            "max_mel_tokens": int(request.max_mel_tokens),
        }
        
        # 执行推理
        if request.infer_mode == "普通推理":
            tts.infer(
                request.prompt_audio, 
                request.text, 
                output_path, 
                verbose=False,
                max_text_tokens_per_sentence=int(request.max_text_tokens_per_sentence),
                **kwargs
            )
        else:
            # 批次推理
            tts.infer_fast(
                request.prompt_audio, 
                request.text, 
                output_path, 
                verbose=False,
                max_text_tokens_per_sentence=int(request.max_text_tokens_per_sentence),
                sentences_bucket_max_size=request.sentences_bucket_max_size,
                **kwargs
            )
        
        # 更新任务状态
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = "TTS task completed successfully"
        tasks[task_id]["result_path"] = output_path
        
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Task failed: {str(e)}"
        print(f"Error processing TTS task {task_id}: {e}")
    finally:
        # 如果存在事件对象，设置它以通知等待的请求
        if task_id in task_events:
            task_events[task_id].set()

task_events: Dict[str, asyncio.Event] = {}

def run_tts_task_in_thread(task_id: str, request: TTSRequest):
    """
    在线程中运行TTS任务的包装函数
    """
    try:
        print(f"Starting TTS task {task_id} in thread")
        process_tts_task(task_id, request)
        print(f"TTS task {task_id} completed in thread")
    except Exception as e:
        print(f"Error in thread for task {task_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保事件被触发
        if task_id in task_events:
            try:
                print(f"Setting event for task {task_id}")
                task_events[task_id].set()
            except Exception as e:
                print(f"Error setting event for task {task_id}: {e}")

@app.post("/api/v1/tts/tasks", response_model=TaskStatus)
async def create_tts_task(request: TTSRequest):
    """
    创建一个新的TTS任务
    
    Args:
        request: TTS请求参数
        
    Returns:
        TaskStatus: 任务状态信息，或直接返回音频文件
    """
    # 验证参考音频文件是否存在
    if not os.path.exists(request.prompt_audio):
        raise HTTPException(status_code=400, detail="Prompt audio file not found")
    
    # 验证推理模式
    if request.infer_mode not in ["普通推理", "批次推理"]:
        raise HTTPException(status_code=400, detail="Invalid infer_mode. Must be '普通推理' or '批次推理'")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks[task_id] = {
        "status": "pending",
        "message": "Task created, waiting to be processed",
        "result_path": None
    }
    
    # 如果需要直接返回音频，创建事件对象
    if request.return_audio:
        task_events[task_id] = asyncio.Event()
    
    # 启动线程运行TTS任务
    thread = threading.Thread(target=run_tts_task_in_thread, args=(task_id, request))
    thread.start()
    
    # 如果需要直接返回音频，则等待任务完成并返回音频
    if request.return_audio:
        try:
            print(f"Waiting for task {task_id} to complete")
            # 等待任务完成（最多等待60秒）
            await asyncio.wait_for(task_events[task_id].wait(), timeout=60.0)
            print(f"Task {task_id} event triggered")
            
            # 检查任务状态
            task = tasks[task_id]
            print(f"Task {task_id} final status: {task['status']}")
            if task["status"] == "completed":
                result_path = task.get("result_path")
                if result_path and os.path.exists(result_path):
                    return FileResponse(result_path, media_type="audio/wav", filename=f"{task_id}.wav")
                else:
                    raise HTTPException(status_code=404, detail="Result file not found")
            elif task["status"] == "failed":
                raise HTTPException(status_code=500, detail=task.get("message", "Task failed"))
            else:
                raise HTTPException(status_code=500, detail="Unexpected task state")
        except asyncio.TimeoutError:
            print(f"Task {task_id} timeout")
            raise HTTPException(status_code=408, detail="Task timeout")
        except Exception as e:
            print(f"Error while waiting for task {task_id}: {e}")
            raise
        finally:
            # 清理事件对象
            if task_id in task_events:
                print(f"Cleaning up event for task {task_id}")
                del task_events[task_id]
    
    # 默认行为：返回任务状态
    return TaskStatus(
        task_id=task_id,
        status="pending",
        message="Task created, waiting to be processed"
    )

@app.get("/api/v1/tts/tasks", response_class=FileResponse)
async def create_tts_task_get(
    prompt_audio: str = Query(..., description="参考音频路径"),
    text: str = Query(..., description="合成文本"),
    infer_mode: str = Query("批次推理", description="推理模式: '普通推理' 或 '批次推理'"),
    max_text_tokens_per_sentence: int = Query(120, description="每句最大文本token数"),
    sentences_bucket_max_size: int = Query(4, description="批次大小"),
    do_sample: bool = Query(True, description="是否采样"),
    top_p: float = Query(0.8, description="top_p采样参数"),
    top_k: int = Query(30, description="top_k采样参数"),
    temperature: float = Query(1.0, description="温度参数"),
    length_penalty: float = Query(0.0, description="长度惩罚"),
    num_beams: int = Query(3, description="beam search数量"),
    repetition_penalty: float = Query(10.0, description="重复惩罚"),
    max_mel_tokens: int = Query(600, description="最大mel token数")
):
    """
    通过GET请求创建一个新的TTS任务并返回音频文件
    
    Args:
        prompt_audio: 参考音频路径
        text: 合成文本
        infer_mode: 推理模式
        max_text_tokens_per_sentence: 每句最大文本token数
        sentences_bucket_max_size: 批次大小
        do_sample: 是否采样
        top_p: top_p采样参数
        top_k: top_k采样参数
        temperature: 温度参数
        length_penalty: 长度惩罚
        num_beams: beam search数量
        repetition_penalty: 重复惩罚
        max_mel_tokens: 最大mel token数
        
    Returns:
        FileResponse: 音频文件
    """
    # 构造请求对象
    request = TTSRequest(
        prompt_audio=prompt_audio,
        text=text,
        infer_mode=infer_mode,
        max_text_tokens_per_sentence=max_text_tokens_per_sentence,
        sentences_bucket_max_size=sentences_bucket_max_size,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        length_penalty=length_penalty,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        max_mel_tokens=max_mel_tokens,
        return_audio=False  # 强制设为False，因为我们会在函数内直接处理等待逻辑
    )
    
    # 验证参考音频文件是否存在
    if not os.path.exists(request.prompt_audio):
        raise HTTPException(status_code=400, detail="Prompt audio file not found")
    
    # 验证推理模式
    if request.infer_mode not in ["普通推理", "批次推理"]:
        raise HTTPException(status_code=400, detail="Invalid infer_mode. Must be '普通推理' or '批次推理'")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks[task_id] = {
        "status": "pending",
        "message": "Task created, waiting to be processed",
        "result_path": None
    }
    
    # 创建事件对象用于等待任务完成
    task_events[task_id] = asyncio.Event()
    
    # 启动线程运行TTS任务
    thread = threading.Thread(target=run_tts_task_in_thread, args=(task_id, request))
    thread.start()
    
    try:
        print(f"Waiting for task {task_id} to complete")
        # 等待任务完成（最多等待60秒）
        await asyncio.wait_for(task_events[task_id].wait(), timeout=60.0)
        print(f"Task {task_id} event triggered")
        
        # 检查任务状态
        task = tasks[task_id]
        print(f"Task {task_id} final status: {task['status']}")
        if task["status"] == "completed":
            result_path = task.get("result_path")
            if result_path and os.path.exists(result_path):
                return FileResponse(result_path, media_type="audio/wav", filename=f"{task_id}.wav")
            else:
                raise HTTPException(status_code=404, detail="Result file not found")
        elif task["status"] == "failed":
            raise HTTPException(status_code=500, detail=task.get("message", "Task failed"))
        else:
            raise HTTPException(status_code=500, detail="Unexpected task state")
    except asyncio.TimeoutError:
        print(f"Task {task_id} timeout")
        raise HTTPException(status_code=408, detail="Task timeout")
    except Exception as e:
        print(f"Error while waiting for task {task_id}: {e}")
        raise
    finally:
        # 清理事件对象
        if task_id in task_events:
            print(f"Cleaning up event for task {task_id}")
            del task_events[task_id]

@app.get("/api/v1/tts/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        TaskStatus: 任务状态信息
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        result_path=task.get("result_path")
    )

@app.get("/api/v1/tts/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """
    获取任务结果音频文件
    
    Args:
        task_id: 任务ID
        
    Returns:
        FileResponse: 音频文件
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task['status']}")
    
    result_path = task.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(result_path, media_type="audio/wav", filename=f"{task_id}.wav")

if __name__ == "__main__":
    host = "0.0.0.0"
    # host = "127.0.0.1"
    uvicorn.run(app, host=host, port=8000)