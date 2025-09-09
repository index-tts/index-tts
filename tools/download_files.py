import requests
import zipfile
import os
import argparse

def download_file_from_google_drive(file_id, destination):
    """
    通过文件ID下载Google Drive共享文件

    Args:
        file_id (str): Google Drive文件的ID
        destination (str): 本地保存路径
    """
    # 基本的下载URL
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    # 发起初始GET请求
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)  # 从响应中获取确认令牌（如果需要）

    if token: # 如果需要确认（大文件）
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # 将响应内容保存到文件
    save_response_content(response, destination)

def get_confirm_token(response):
    """
    从响应中检查是否存在下载确认令牌（cookie）

    Args:
        response (requests.Response): 响应对象

    Returns:
        str: 确认令牌的值（如果存在），否则为None
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'): # 确认令牌的cookie通常以这个开头
            return value
    return None

def save_response_content(response, destination, chunk_size=32768):
    """
    以流式方式将响应内容写入文件，支持大文件下载。

    Args:
        response (requests.Response): 流式响应对象
        destination (str): 本地保存路径
        chunk_size (int, optional): 每次迭代写入的块大小. Defaults to 32768.
    """
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk: # 过滤掉保持连接的空白块
                f.write(chunk)

def download_model_from_modelscope(destination,hf_cache_dir):
    """
    从ModelScope下载模型（伪代码，需根据实际API实现）
    Args:
        model_id (str): ModelScope模型ID
        destination (str): 本地保存路径
    """
    print(f"[ModelScope] Downloading models to {destination},model cache dir={hf_cache_dir}")
    from modelscope import snapshot_download
    snapshot_download("IndexTeam/IndexTTS-2", local_dir="checkpoints")
    snapshot_download("amphion/MaskGCT", local_dir="checkpoints/hf_cache/models--amphion--MaskGCT")
    snapshot_download("facebook/w2v-bert-2.0",local_dir="checkpoints/hf_cache/models--facebook--w2v-bert-2.0")
    snapshot_download("nv-community/bigvgan_v2_22khz_80band_256x",local_dir="checkpoints/hf_cache/models--nvidia--bigvgan_v2_22khz_80band_256x")
    # models--funasr--campplus modelscope目前还没有
    snapshot_download("funasr/campplus",local_dir=os.path.join(hf_cache_dir,"models--funasr--campplus"))

def download_model_from_huggingface(destination,hf_cache_dir):
    """
    从HuggingFace下载模型（伪代码，需根据实际API实现）
    Args:
        model_id (str): HuggingFace模型ID
        destination (str): 本地保存路径
    """
    print(f"[HuggingFace] Downloading models to {destination},model cache dir={hf_cache_dir}")
    from huggingface_hub import snapshot_download
    snapshot_download("IndexTeam/IndexTTS-2", local_dir=destination)
    snapshot_download("amphion/MaskGCT", local_dir=os.path.join(hf_cache_dir,"models--amphion--MaskGCT"))
    snapshot_download("facebook/w2v-bert-2.0",local_dir=os.path.join(hf_cache_dir,"models--facebook--w2v-bert-2.0"))
    snapshot_download("nvidia/bigvgan_v2_22khz_80band_256x",local_dir=os.path.join(hf_cache_dir, "models--nvidia--bigvgan_v2_22khz_80band_256x"))
    snapshot_download("funasr/campplus",local_dir=os.path.join(hf_cache_dir,"models--funasr--campplus"))

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载文件和模型工具")
    parser.add_argument('--model_source', choices=['modelscope', 'huggingface'], default=None, help='模型下载来源')
    args = parser.parse_args()

    if args.model_source:
        if args.model_source == 'modelscope':
            download_model_from_modelscope("checkpoints",os.path.join("checkpoints","hf_cache"))
        elif args.model_source == 'huggingface':
            download_model_from_huggingface("checkpoints",os.path.join("checkpoints","hf_cache"))

    # 准备样例文件
    print("Downloading example files from Google Drive...")
    file_id = "1o_dCMzwjaA2azbGOxAE7-4E7NbJkgdgO"
    destination = "example_wavs.zip"
    download_file_from_google_drive(file_id, destination)
    print(f"File downloaded to: {destination}")
    # 解压下载的zip文件到examples目录
    examples_dir = "examples"
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(examples_dir)
    print(f"File extracted to: {examples_dir}")
