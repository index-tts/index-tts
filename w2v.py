#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexTTS Flask 服务器
提供文本转语音的 REST API 服务
查看语音包命令：curl http://localhost:5000/voices
使用服务命令：curl -X POST http://localhost:5000/tts/dingzhen.wav -d "需要转换的文本" -dl -o "output.wav"
"""

import os
import uuid
import time
import datetime
import json
import numpy as np
from flask import Flask, request, jsonify, send_file, Response

from indextts.infer import IndexTTS

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 确保中文正常显示

# 全局变量存储模型
tts_model = None

def load_model():
    """加载 TTS 模型"""
    global tts_model
    print("🔍 开始执行 load_model() 函数...")
    try:
        print("开始加载 IndexTTS 模型...")
        start_time = time.time()
        from indextts.infer import IndexTTS
        print("✅ IndexTTS 导入成功")

        # 初始化模型，请根据实际情况调整到合适的参数
        tts_model = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml", is_fp16=False, device="cpu")
        print("✅ IndexTTS 初始化成功")
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {str(datetime.timedelta(seconds=load_time))}")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈
        return False


def find_voice_file(voice_file):
    """查找语音文件，支持多种路径格式"""
    # 如果已经是绝对路径或当前目录存在，直接返回
    if os.path.isabs(voice_file) or os.path.exists(voice_file):
        return voice_file
    
    # 检查 reference_voice 目录
    reference_path = os.path.join("reference_voice", voice_file)
    if os.path.exists(reference_path):
        return reference_path
    
    # 检查 test_data 目录
    test_data_path = os.path.join("test_data", voice_file)
    if os.path.exists(test_data_path):
        return test_data_path
    
    # 如果都找不到，返回原文件名（会在后续检查中报错）
    return voice_file

def list_available_voices():
    """列出所有可用的语音文件"""
    voices = []
    
    # 检查 reference_voice 目录
    reference_dir = "reference_voice"
    if os.path.exists(reference_dir):
        for file in os.listdir(reference_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                voices.append(f"reference_voice/{file}")
    
    # 检查 test_data 目录（用户注释的部分）
    # test_data_dir = "test_data"
    # if os.path.exists(test_data_dir):
    #     for file in os.listdir(test_data_dir):
    #         if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
    #             voices.append(f"test_data/{file}")
    
    # 检查当前目录（用户注释的部分）
    # for file in os.listdir("."):
    #     if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
    #         voices.append(file)
    
    return voices

@app.before_request
def before_first_request():
    """在第一个请求前加载模型"""
    global tts_model
    print("🔍 before_first_request 被调用")
    if tts_model is None:
        print("🔍 模型为空，开始加载...")
        load_model()
    else:
        print("✅ 模型已存在，跳过加载")

@app.route('/tts', methods=['POST'])
@app.route('/tts/<voice_file>', methods=['POST'])
def text_to_speech(voice_file='dingzhen.wav'):
    """文本转语音接口"""
    
    # 从多个来源获取文本内容
    text = ""
    
    # 方法1: 从自定义头部获取（支持多种头部名称）
    possible_headers = ['H', 'h', 'Text', 'text', 'Content', 'content']
    for header_name in possible_headers:
        header_value = request.headers.get(header_name, '')
        if header_value and header_value.strip():
            text = header_value.strip()
            break
    
    # 方法2: 从URL参数获取
    if not text:
        text = request.args.get('text', '')
    
    # 方法3: 从表单数据获取
    if not text:
        # 处理 curl -d "text" 的情况，在Windows下可能text作为key出现
        for key, value in request.form.items():
            if value and value.strip():
                text = value
                break
            elif key and key.strip() and not value:
                # 如果key有内容但value为空，使用key作为文本
                text = key
                break
    
    # 方法4: 从JSON获取
    if not text:
        try:
            data = request.get_json(silent=True)
            if data and 'text' in data:
                text = data['text']
        except:
            pass
    
    # 方法5: 从原始数据获取（处理curl -d的情况）
    if not text and request.data:
        try:
            # 尝试解析原始数据
            raw_data = request.data.decode('utf-8')
            if raw_data:
                # 如果是简单的文本，直接使用
                if not raw_data.startswith('{') and not raw_data.startswith('['):
                    text = raw_data
                else:
                    # 尝试解析JSON
                    data = json.loads(raw_data)
                    if 'text' in data:
                        text = data['text']
        except:
            pass
    
    # 检查下载参数 - 支持多种方式
    download = (
        request.args.get('dl', False) or  # URL参数 ?dl=1
        request.args.get('download', False) or  # URL参数 ?download=1
        'dl' in str(request.args) or  # 简单的dl参数
        'download' in str(request.args)  # 简单的download参数
    )
    
    # 检查必需参数
    if not text:
        return jsonify({
            "error": "缺少文本内容",
            "help": "请使用以下方式之一传递文本：",
            "methods": [
                "1. curl -X POST http://localhost:5000/tts/dingzhen.wav -H \"你好，世界！\"",
                "2. curl -X POST http://localhost:5000/tts/dingzhen.wav -d \"你好，世界！\"",
                "3. curl -X POST http://localhost:5000/tts/dingzhen.wav -d '{\"text\": \"你好，世界！\"}'"
            ]
        }), 400
    
    # 验证文本长度
    if len(text) > 500:
        return jsonify({"error": "文本长度不能超过500字符"}), 400
    
    # 查找语音文件
    actual_voice_path = find_voice_file(voice_file)
    
    # 检查语音文件是否存在
    if not os.path.exists(actual_voice_path):
        return jsonify({
            "error": f"语音文件 '{voice_file}' 不存在",
            "available_voices": list_available_voices()
        }), 400
    
    # 开始转换
    start_time = time.time()
    
    # 生成唯一的输出文件名
    output_filename = f"output_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join("outputs", "tasks", output_filename)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 执行TTS转换
    try:
        tts_model.infer(actual_voice_path, text, output_path=output_path)
    except Exception as e:
        return jsonify({"error": f"TTS转换失败: {str(e)}"}), 500
    
    # 检查输出文件是否生成成功
    if not os.path.exists(output_path):
        return jsonify({"error": "TTS转换失败，未生成音频文件"}), 500
    
    # 检查音频文件质量
    file_size = os.path.getsize(output_path)
    print(f"📊 生成的音频文件大小: {file_size} bytes")
    
    if file_size == 0:
        return jsonify({"error": "TTS转换失败，生成的音频文件为空"}), 500
    
    # 尝试检查音频文件信息
    try:
        import librosa
        import soundfile as sf
        info = sf.info(output_path)
        print(f"音频文件信息:")
        print(f"   - 采样率: {info.samplerate} Hz")
        print(f"   - 声道数: {info.channels}")
        print(f"   - 时长: {info.duration:.2f} 秒")
        print(f"   - 格式: {info.format}")
        
        # 检查音频数据
        audio_data, sr = librosa.load(output_path, sr=None)
        non_zero_samples = np.count_nonzero(audio_data)
        print(f"   - 非零样本数: {non_zero_samples}")
        
        if non_zero_samples == 0:
            print("警告: 音频文件包含全零数据")
        
    except Exception as e:
        print(f"无法检查音频文件信息: {e}")
    
    conversion_time = time.time() - start_time
    
    # 如果有下载参数，直接返回文件
    if download:
        try:
            # 确保文件存在且有内容
            if not os.path.exists(output_path):
                return jsonify({"error": "音频文件不存在"}), 404
            
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                return jsonify({"error": "音频文件为空"}), 500
            
            # 读取文件内容
            with open(output_path, 'rb') as f:
                file_content = f.read()
            
            # 验证读取的文件大小
            if len(file_content) != file_size:
                return jsonify({"error": "文件读取不完整"}), 500
            
            # 创建响应
            response = Response(
                file_content,
                status=200,
                mimetype='audio/wav'
            )
            
            # 设置下载头
            response.headers['Content-Disposition'] = f'attachment; filename="{output_filename}"'
            response.headers['Content-Length'] = str(len(file_content))
            response.headers['Content-Type'] = 'audio/wav'
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'no-cache'
            
            return response
            
        except Exception as e:
            return jsonify({"error": f"文件下载失败: {str(e)}"}), 500
    
    # 返回成功响应
    return jsonify({
        "success": True,
        "message": "TTS转换成功",
        "output_file": output_filename,
        "text": text,
        "voice_file": voice_file,
        "conversion_time": f"{conversion_time:.2f}秒",
        "download_url": f"/download/{output_filename}"
    })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载生成的音频文件"""
    file_path = os.path.join("outputs", "tasks", filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "文件不存在"}), 404
    
    return send_file(file_path, as_attachment=True, download_name=filename)

@app.route('/voices', methods=['GET'])
def get_voices():
    """获取可用的语音文件列表"""
    return jsonify({
        "available_voices": list_available_voices(),
        "total_count": len(list_available_voices())
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def api_info():
    """API信息接口"""
    return jsonify({
        "service": "IndexTTS API Server",
        "endpoints": {
            "POST /tts": "文本转语音（简化格式）",
            "POST /tts/<voice_file>": "文本转语音（指定语音文件）",
            "GET /voices": "获取可用语音文件列表",
            "GET /download/<filename>": "下载生成的音频文件",
            "GET /health": "健康检查"
        },
        "usage_examples": {
            "tts_with_data": 'curl -X POST http://localhost:5000/tts/dingzhen.wav -d "你好，世界！"',
            "tts_download": 'curl -X POST "http://localhost:5000/tts/dingzhen.wav?dl=1" -d "你好，世界！"',
            "tts_download_save": 'curl -X POST "http://localhost:5000/tts/dingzhen.wav?dl=1" -d "你好，世界！" -o "my_audio.wav"',
            "curl_voices": "curl http://localhost:5000/voices",
            "curl_health": "curl http://localhost:5000/health"
        },
        "note": "支持多种参数传递方式：-H 参数、-d 参数、JSON格式等"
    })

if __name__ == '__main__':
    # 在启动应用前预先加载模型
    print("🔍 主函数开始执行...")
    print("启动服务前预先加载模型...")
    if load_model():
        print("模型加载成功，启动服务器...")
        print("服务将在以下地址启动:")
        print("  - 本地访问: http://localhost:5000")
        print("  - 网络访问: http://0.0.0.0:5000")
        print("\n可用的API端点:")
        print("  - 根路径: http://localhost:5000/")
        print("  - 健康检查: http://localhost:5000/health")
        print("  - 语音列表: http://localhost:5000/voices")
        print("  - TTS转换: http://localhost:5000/tts")
        print("  - TTS转换(指定语音): http://localhost:5000/tts/<voice_file>")
        print("\n多种参数传递方式:")
        print("  1. curl -X POST http://localhost:5000/tts/dingzhen.wav -d \"你好，世界！\"")
        print("  2. curl -X POST \"http://localhost:5000/tts/dingzhen.wav?dl=1\" -d \"你好，世界！\"  # 直接下载")
        print("  3. curl -X POST \"http://localhost:5000/tts/dingzhen.wav?dl=1\" -d \"你好，世界！\" -o \"my_audio.wav\"  # 下载并保存")
        print("\n按 Ctrl+C 停止服务")
        print("-" * 50)
        
        app.run(host='0.0.0.0', port=5000, threaded=True)
    else:
        print("模型加载失败，无法启动服务器")