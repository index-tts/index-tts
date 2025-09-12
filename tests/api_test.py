# file: api_test.py
import os
import time
import requests
import json
from pathlib import Path

# API 基础URL
BASE_URL = "http://127.0.0.1:8000/api/v1/tts/tasks"

def test_create_tts_task():
    """
    测试创建TTS任务
    """
    # 准备测试数据
    test_prompt = "tests/sample_prompt.wav"  # 请确保这个文件存在
    
    # 如果测试音频文件不存在，则创建一个占位文件
    if not os.path.exists(test_prompt):
        os.makedirs("prompts", exist_ok=True)
        Path(test_prompt).touch()
        print(f"Created placeholder prompt file: {test_prompt}")
    
    payload = {
        "prompt_audio": test_prompt,
        "text": "你好，这是一个测试语音合成任务。欢迎使用Index T T S API！",
        "infer_mode": "批次推理",
        "max_text_tokens_per_sentence": 120,
        "sentences_bucket_max_size": 4,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 1.0,
        "length_penalty": 0.0,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 600
    }
    
    try:
        response = requests.post(BASE_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            print("✓ 创建TTS任务成功")
            print(f"  Task ID: {result['task_id']}")
            print(f"  Status: {result['status']}")
            return result['task_id']
        else:
            print(f"✗ 创建TTS任务失败: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return None
    
def test_create_tts_task_with_audio_return():
    """
    测试创建TTS任务并直接返回音频
    """
    # 准备测试数据
    test_prompt = "tests/sample_prompt.wav"
    
    # 如果测试音频文件不存在，则创建一个占位文件
    if not os.path.exists(test_prompt):
        os.makedirs("prompts", exist_ok=True)
        Path(test_prompt).touch()
        print(f"Created placeholder prompt file: {test_prompt}")
    
    payload = {
        "prompt_audio": test_prompt,
        "text": "这是测试直接返回音频功能的语音合成任务。",
        "return_audio": True  # 设置为True以直接返回音频
    }
    
    try:
        response = requests.post(BASE_URL, json=payload)
        if response.status_code == 200 and response.headers.get('content-type', '').startswith('audio/'):
            # 保存音频文件
            output_file = "test_output_direct_audio.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            print("✓ 创建TTS任务并直接返回音频成功")
            print(f"  音频文件已保存为: {output_file}")
            return True
        elif response.status_code == 408:
            print("✗ 请求超时，任务未在规定时间内完成")
            return False
        elif response.status_code == 500:
            print(f"✗ 任务执行失败: {response.text}")
            return False
        else:
            print(f"✗ 创建TTS任务并直接返回音频失败: {response.status_code}")
            print(f"  Response headers: {response.headers}")
            print(f"  Response content: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False
    
def test_create_tts_task_get():
    """
    测试通过GET请求创建TTS任务并直接返回音频
    """
    # 准备测试数据
    test_prompt = "tests/sample_prompt.wav"
    
    # 如果测试音频文件不存在，则创建一个占位文件
    if not os.path.exists(test_prompt):
        os.makedirs("prompts", exist_ok=True)
        Path(test_prompt).touch()
        print(f"Created placeholder prompt file: {test_prompt}")
    
    # 构造查询参数
    params = {
        "prompt_audio": test_prompt,
        "text": "这是通过GET请求创建的测试语音合成任务。",
        "infer_mode": "批次推理",
        "max_text_tokens_per_sentence": 120,
        "sentences_bucket_max_size": 4,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 1.0,
        "length_penalty": 0.0,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 600
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200 and response.headers.get('content-type', '').startswith('audio/'):
            # 保存音频文件
            output_file = "test_output_get_request.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            print("✓ 通过GET请求创建TTS任务并返回音频成功")
            print(f"  音频文件已保存为: {output_file}")
            return True
        elif response.status_code == 408:
            print("✗ GET请求超时，任务未在规定时间内完成")
            return False
        elif response.status_code == 500:
            print(f"✗ GET请求任务执行失败: {response.text}")
            return False
        else:
            print(f"✗ 通过GET请求创建TTS任务失败: {response.status_code}")
            print(f"  Response headers: {response.headers}")
            print(f"  Response content: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"✗ GET请求失败: {e}")
        return False

def test_get_task_status(task_id):
    """
    测试获取任务状态
    """
    if not task_id:
        print("无效的任务ID")
        return None
        
    try:
        response = requests.get(f"{BASE_URL}/{task_id}")
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 获取任务状态成功")
            print(f"  Task ID: {result['task_id']}")
            print(f"  Status: {result['status']}")
            print(f"  Message: {result['message']}")
            return result
        else:
            print(f"✗ 获取任务状态失败: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return None

def test_get_task_result(task_id):
    """
    测试获取任务结果
    """
    if not task_id:
        print("无效的任务ID")
        return False
        
    try:
        response = requests.get(f"{BASE_URL}/{task_id}/result")
        if response.status_code == 200:
            # 保存音频文件
            output_file = f"test_output_{task_id}.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"✓ 下载音频文件成功: {output_file}")
            return True
        else:
            print(f"✗ 下载音频文件失败: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False

def test_invalid_task():
    """
    测试无效任务ID
    """
    print("测试无效任务ID...")
    try:
        response = requests.get(f"{BASE_URL}/invalid_task_id")
        if response.status_code == 404:
            print("✓ 正确处理无效任务ID")
        else:
            print(f"✗ 未正确处理无效任务ID: {response.status_code}")
    except Exception as e:
        print(f"✗ 请求失败: {e}")

def test_invalid_prompt():
    """
    测试无效的参考音频路径
    """
    print("测试无效参考音频路径...")
    payload = {
        "prompt_audio": "nonexistent.wav",
        "text": "测试无效音频路径",
        "infer_mode": "普通推理"
    }
    
    try:
        response = requests.post(BASE_URL, json=payload)
        if response.status_code == 400:
            print("✓ 正确处理无效音频路径")
        else:
            print(f"✗ 未正确处理无效音频路径: {response.status_code}")
    except Exception as e:
        print(f"✗ 请求失败: {e}")

def test_invalid_infer_mode():
    """
    测试无效的推理模式
    """
    print("测试无效推理模式...")
    payload = {
        "prompt_audio": "prompts/sample_prompt.wav",
        "text": "测试无效推理模式",
        "infer_mode": "无效模式"
    }
    
    try:
        response = requests.post(BASE_URL, json=payload)
        if response.status_code == 400:
            print("✓ 正确处理无效推理模式")
        else:
            print(f"✗ 未正确处理无效推理模式: {response.status_code}")
    except Exception as e:
        print(f"✗ 请求失败: {e}")

def main():
    """
    主测试函数
    """
    print("开始测试 IndexTTS API...")
    print("=" * 50)
    
    # 测试1: 创建TTS任务
    print("1. 测试创建TTS任务:")
    task_id = test_create_tts_task()
    print()
    
    if task_id:
        # 测试2: 获取任务状态
        print("2. 测试获取任务状态:")
        status = test_get_task_status(task_id)
        print()
        
        # 等待一段时间让任务完成（模拟）
        print("等待任务处理完成...")
        time.sleep(10)
        
        # 再次检查状态
        status = test_get_task_status(task_id)
        print()
        
        # 如果任务完成，测试获取结果
        if status and status.get('status') == 'completed':
            print("3. 测试获取任务结果:")
            test_get_task_result(task_id)
            print()
        else:
            print("任务未完成，跳过结果下载测试")
            print()
    
    # 测试直接返回音频功能
    print("4. 测试创建TTS任务并直接返回音频:")
    test_create_tts_task_with_audio_return()
    print()
    
    # 测试GET请求版本
    print("5. 测试通过GET请求创建TTS任务并返回音频:")
    test_create_tts_task_get()
    print()
    
    # 测试错误情况
    print("6. 测试错误处理:")
    test_invalid_task()
    test_invalid_prompt()
    test_invalid_infer_mode()
    print()
    
    print("测试完成!")

if __name__ == "__main__":
    main()