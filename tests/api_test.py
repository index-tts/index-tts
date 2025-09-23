import requests
import time

# API 服务器地址
BASE_URL = "http://localhost:8000"

# 测试用的音频文件
TEST_PROMPT_AUDIO = "tests/sample_prompt.wav"

def test_create_task_async():
    """测试异步创建 TTS 任务"""
    print("\n测试异步创建 TTS 任务...")
    try:
        # 准备测试数据
        payload = {
            "text": "这是异步创建TTS任务的测试。",
            "prompt_audio": TEST_PROMPT_AUDIO,
            "return_audio": False
        }
        
        # 发送 POST 请求创建任务
        response = requests.post(f"{BASE_URL}/api/v1/tts/tasks", json=payload)
        assert response.status_code == 200, f"创建任务失败，状态码: {response.status_code}"
        
        # 解析响应
        task_data = response.json()
        task_id = task_data["task_id"]
        assert task_data["status"] == "pending", f"任务状态应为 pending，实际为: {task_data['status']}"
        
        print(f"✓ 异步任务创建成功，任务ID: {task_id}")
        return task_id
    except Exception as e:
        print(f"✗ 异步任务创建测试失败: {e}")
        return None

def test_create_task_sync():
    """测试同步创建 TTS 任务并返回音频"""
    print("\n测试同步创建 TTS 任务...")
    try:
        # 发送 GET 请求创建任务并直接获取音频
        params = {
            "text": "这是同步创建TTS任务并返回音频的测试。",
            "prompt_audio": TEST_PROMPT_AUDIO
        }
        response = requests.get(f"{BASE_URL}/api/v1/tts/tasks", params=params)
        
        # 检查响应
        assert response.status_code == 200, f"同步任务创建失败，状态码: {response.status_code}"
        assert response.headers['content-type'] in ['audio/wav', 'application/octet-stream'], \
            f"返回内容类型不正确: {response.headers.get('content-type')}"
        
        # 保存音频文件
        with open("test_sync_output.wav", "wb") as f:
            f.write(response.content)
        
        print("✓ 同步任务创建成功，音频已保存为 test_sync_output.wav")
        return True
    except Exception as e:
        print(f"✗ 同步任务创建测试失败: {e}")
        return False

def test_get_task_status(task_id):
    """测试获取任务状态"""
    print("\n测试获取任务状态...")
    try:
        # 发送 GET 请求获取任务状态
        response = requests.get(f"{BASE_URL}/api/v1/tts/tasks/{task_id}")
        assert response.status_code == 200, f"获取任务状态失败，状态码: {response.status_code}"
        
        # 解析响应
        task_data = response.json()
        assert task_data["task_id"] == task_id, "返回的任务ID不匹配"
        assert "status" in task_data, "响应中缺少 status 字段"
        
        print(f"✓ 获取任务状态成功，当前状态: {task_data['status']}")
        return task_data["status"]
    except Exception as e:
        print(f"✗ 获取任务状态测试失败: {e}")
        return None

def test_get_task_result(task_id):
    """测试获取任务结果"""
    print("\n测试获取任务结果...")
    try:
        # 发送 GET 请求获取任务结果
        response = requests.get(f"{BASE_URL}/api/v1/tts/tasks/{task_id}/result")
        
        # 如果任务已完成，应该返回音频文件
        if response.status_code == 200:
            assert response.headers['content-type'] in ['audio/wav', 'application/octet-stream'], \
                f"返回内容类型不正确: {response.headers.get('content-type')}"
            
            # 保存音频文件
            with open(f"test_task_{task_id}.wav", "wb") as f:
                f.write(response.content)
            
            print(f"✓ 获取任务结果成功，音频已保存为 test_task_{task_id}.wav")
            return True
        elif response.status_code == 400:
            # 任务尚未完成
            print("⚠ 任务尚未完成，无法获取结果")
            return False
        else:
            print(f"✗ 获取任务结果失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 获取任务结果测试失败: {e}")
        return False

def test_emotion_control_methods():
    """测试不同的情感控制方法"""
    print("\n=== 测试情感控制方法 ===")
    
    # 测试情感控制方法 0: 与音色参考音频相同
    print("\n测试情感控制方法 0 (与音色参考音频相同)...")
    try:
        payload = {
            "text": "这是情感控制的测试，控制方法是与音色参考音频相同。",
            "prompt_audio": TEST_PROMPT_AUDIO,
            "return_audio": False,
            "emo_control_method": 0
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/tts/tasks", json=payload)
        assert response.status_code == 200
        print("✓ 情感控制方法 0 测试成功")
    except Exception as e:
        print(f"✗ 情感控制方法 0 测试失败: {e}")
    
    # 测试情感控制方法 1: 使用情感参考音频
    print("\n测试情感控制方法 1 (使用情感参考音频)...")
    try:
        payload = {
            "text": "这是情感控制的测试，控制方法是使用情感参考音频。",
            "prompt_audio": TEST_PROMPT_AUDIO,
            "return_audio": False,
            "emo_control_method": 1,
            "emo_ref_path": TEST_PROMPT_AUDIO  # 使用相同音频作为情感参考
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/tts/tasks", json=payload)
        assert response.status_code == 200
        print("✓ 情感控制方法 1 测试成功")
    except Exception as e:
        print(f"✗ 情感控制方法 1 测试失败: {e}")
    
    # 测试情感控制方法 2: 使用情感向量控制
    print("\n测试情感控制方法 2 (使用情感向量控制)...")
    try:
        payload = {
            "text": "这是情感控制的测试，控制方法是使用情感向量控制。",
            "prompt_audio": TEST_PROMPT_AUDIO,
            "return_audio": False,
            "emo_control_method": 2,
            "emo_vec": [0, 0, 0, 0.4, 0, 0, 0, 0]  # 8个情感维度
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/tts/tasks", json=payload)
        assert response.status_code == 200
        print("✓ 情感控制方法 2 测试成功")
    except Exception as e:
        print(f"✗ 情感控制方法 2 测试失败: {e}")

def test_advanced_parameters():
    """测试高级参数设置"""
    print("\n=== 测试高级参数设置 ===")
    
    try:
        payload = {
            "text": "这是高级参数设置测试。",
            "prompt_audio": TEST_PROMPT_AUDIO,
            "return_audio": False,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_beams": 2,
            "repetition_penalty": 1.2,
            "length_penalty": 0.5,
            "max_mel_tokens": 1000,
            "max_text_tokens_per_segment": 100
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/tts/tasks", json=payload)
        assert response.status_code == 200
        print("✓ 高级参数设置测试成功")
    except Exception as e:
        print(f"✗ 高级参数设置测试失败: {e}")

def test_task_lifecycle():
    """测试完整的任务生命周期"""
    print("\n=== 测试完整的任务生命周期 ===")
    
    # 1. 创建异步任务
    task_id = test_create_task_async()
    if not task_id:
        return
    
    # 2. 轮询任务状态直到完成或失败
    max_attempts = 30  # 最多尝试30次
    attempt = 0
    while attempt < max_attempts:
        status = test_get_task_status(task_id)
        if not status:
            return
            
        if status == "completed":
            print("✓ 任务已完成")
            break
        elif status == "failed":
            print("✗ 任务执行失败")
            return
        elif status == "processing":
            print(f"... 任务处理中，第 {attempt + 1} 次检查")
        
        attempt += 1
        time.sleep(2)  # 等待2秒后再次检查
    
    if attempt >= max_attempts:
        print("⚠ 任务处理超时")
        return
    
    # 3. 获取任务结果
    test_get_task_result(task_id)

def test_error_cases():
    """测试错误情况处理"""
    print("\n=== 测试错误情况处理 ===")
    
    # 测试不存在的任务ID
    print("\n测试不存在的任务ID...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/tts/tasks/nonexistent_task_id")
        assert response.status_code == 404, f"应该返回404状态码，实际: {response.status_code}"
        print("✓ 正确处理不存在的任务ID")
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
    
    # 测试无效的参考音频文件
    print("\n测试无效的参考音频文件...")
    try:
        payload = {
            "text": "这是无效参考音频文件测试。",
            "prompt_audio": "nonexistent_audio.wav",
            "return_audio": False
        }
        response = requests.post(f"{BASE_URL}/api/v1/tts/tasks", json=payload)
        assert response.status_code == 400, f"应该返回400状态码，实际: {response.status_code}"
        print("✓ 正确处理无效的参考音频文件")
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")

def main():
    """主测试函数"""
    print("开始测试 IndexTTS API...")
    
    # 测试同步任务创建
    test_create_task_sync()
    
    # 测试情感控制方法
    test_emotion_control_methods()
    
    # 测试高级参数设置
    test_advanced_parameters()
    
    # 测试完整的任务生命周期
    test_task_lifecycle()
    
    # 测试错误情况处理
    test_error_cases()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()