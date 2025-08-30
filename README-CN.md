
<div align="center">
<img src='assets/index_icon.png' width="250"/>
</div>


<h2><center>IndexTTS: 一款工业级可控且高效的零样本文本转语音系统</h2>

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>

[**English**](./REDME.md) | [**中文简体**](./README-CN.md) 

## 👉🏻 IndexTTS 👈🏻

[[HuggingFace Demo]](https://huggingface.co/spaces/IndexTeam/IndexTTS)   [[ModelScope Demo]](https://modelscope.cn/studios/IndexTeam/IndexTTS-Demo) \
[[Paper]](https://arxiv.org/abs/2502.05512)  [[Demos]](https://index-tts.github.io)  

**IndexTTS** 是一种主要基于 XTTS 和 Tortoise 的 GPT 风格的文本转语音 （TTS） 模型。它能够使用拼音纠正汉字的发音，并通过标点符号控制任何位置的停顿。我们增强了系统的多个模块，包括扬声器条件特征表示的改进，以及 BigVGAN2 的集成以优化音频质量。经过数万小时的数据训练，我们的系统实现了最先进的性能，优于当前流行的 TTS 系统，如 XTTS、CosyVoice2、Fish-Speech 和 F5-TTS。
<span style="font-size:16px;">  
体验 **IndexTTS**：请联系 <u>xuanwu@bilibili.com</u> 了解更多详细信息。 </span>
### Contact
QQ群（二群）：1048202584 \
Discord：https://discord.gg/uT32E7KDmy  \
简历：indexspeech@bilibili.com  \
欢迎大家来交流讨论！
## 📣 Updates

- `2025/05/14` 🔥🔥我们发布了 **IndexTTS-1.5**，显著提高了模型在英语语言中的稳定性和性能。
- `2025/03/25` 🔥 我们发布了IndexTTS-1.0模型参数和推理代码。
- `2025/02/12` 🔥 我们在arXiv上提交了我们的论文，并发布了我们的演示和测试集。

## 🖥️ Method

IndexTTS的概述如下所示。

<picture>
  <img src="assets/IndexTTS.png"  width="800"/>
</picture>


主要的改善和贡献总结如下:
 - 在中文场景中，我们引入了一种字符-拼音混合建模方法。这允许快速纠正发音错误的字符。
 - **IndexTTS**结合了一个变换器条件编码器和基于BigVGAN2的语音编码解码器。这提高了训练的稳定性、声音音色的相似性和音质。
 - 我们在这里发布所有测试集，包括多音节词的测试集、主观和客观测试集。



## 模型下载
| 🤗**HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |
| [😁IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |


## 📑 Evaluation

**IndexTTS 和基线模型的单词错误率 （WER） 结果** [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| **WER**                | **test_zh** | **test_en** | **test_hard** |
|:----------------------:|:-----------:|:-----------:|:-------------:|
| **Human**              | 1.26        | 2.14        | -             |
| **SeedTTS**            | 1.002       | 1.945       | **6.243**     |
| **CosyVoice 2**        | 1.45        | 2.57        | 6.83          |
| **F5TTS**              | 1.56        | 1.83        | 8.67          |
| **FireRedTTS**         | 1.51        | 3.82        | 17.45         |
| **MaskGCT**            | 2.27        | 2.62        | 10.27         |
| **Spark-TTS**          | 1.2         | 1.98        | -             |
| **MegaTTS 3**          | 1.36        | 1.82        | -             |
| **IndexTTS**           | 0.937       | 1.936       | 6.831         |
| **IndexTTS-1.5**       | **0.821**   | **1.606**   | 6.565         |


**另一个开源测试中 IndexTTS 和基线模型的单词错误率 （WER） 结果**


|    **Model**    | **aishell1_test** | **commonvoice_20_test_zh** | **commonvoice_20_test_en** | **librispeech_test_clean** |  **avg** |
|:---------------:|:-----------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------:|
|    **Human**    |        2.0        |            9.5             |            10.0            |            2.4             |   5.1    |
| **CosyVoice 2** |        1.8        |            9.1             |            7.3             |            4.9             |   5.9    |
|    **F5TTS**    |        3.9        |            11.7            |            5.4             |            7.8             |   8.2    |
|  **Fishspeech** |        2.4        |            11.4            |            8.8             |            8.0             |   8.3    |
|  **FireRedTTS** |        2.2        |            11.0            |            16.3            |            5.7             |   7.7    |
|     **XTTS**    |        3.0        |            11.4            |            7.1             |            3.5             |   6.0    |
|   **IndexTTS**  |      1.3          |          7.0               |            5.3             |          2.1             | 3.7       |
|   **IndexTTS-1.5**  |      **1.2**     |          **6.8**          |          **3.9**          |          **1.7**          | **3.1** |


**IndexTTS 和基线模型的说话人相似度 （SS） 结果**

|    **Model**    | **aishell1_test** | **commonvoice_20_test_zh** | **commonvoice_20_test_en** | **librispeech_test_clean** |  **avg**  |
|:---------------:|:-----------------:|:--------------------------:|:--------------------------:|:--------------------------:|:---------:|
|    **Human**    |       0.846       |            0.809           |            0.820           |            0.858           |   0.836   |
| **CosyVoice 2** |     **0.796**     |            0.743           |            0.742           |          **0.837**         | **0.788** |
|    **F5TTS**    |       0.743       |          **0.747**         |            0.746           |            0.828           |   0.779   |
|  **Fishspeech** |       0.488       |            0.552           |            0.622           |            0.701           |   0.612   |
|  **FireRedTTS** |       0.579       |            0.593           |            0.587           |            0.698           |   0.631   |
|     **XTTS**    |       0.573       |            0.586           |            0.648           |            0.761           |   0.663   |
|   **IndexTTS**  |       0.744       |            0.742           |          **0.758**         |            0.823           |   0.776   |
|   **IndexTTS-1.5**  |       0.741       |            0.722           |          0.753         |            0.819           |   0.771   |



**零样本克隆语音的 MOS 分数**

| **Model**       | **Prosody** | **Timbre** | **Quality** |  **AVG**  |
|-----------------|:-----------:|:----------:|:-----------:|:---------:|
| **CosyVoice 2** |    3.67     |    4.05    |    3.73     |   3.81    |
| **F5TTS**       |    3.56     |    3.88    |    3.56     |   3.66    |
| **Fishspeech**  |    3.40     |    3.63    |    3.69     |   3.57    |
| **FireRedTTS**  |    3.79     |    3.72    |    3.60     |   3.70    |
| **XTTS**        |    3.23     |    2.99    |    3.10     |   3.11    |
| **IndexTTS**    |    **3.79**     |    **4.20**    |    **4.05**     |   **4.01**    |


## 使用说明
### 环境设置
1. 下载此存储库:
```bash
git clone https://github.com/index-tts/index-tts.git
```
2. 安装依赖:

创建新的 conda 环境并安装依赖项:
 
```bash
conda create -n index-tts python=3.10
conda activate index-tts
apt-get install ffmpeg
# or use conda to install ffmpeg
conda install -c conda-forge ffmpeg
```

安装 [PyTorch](https://pytorch.org/get-started/locally/), 例如。:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> [!NOTE]
> 如果您使用的是 Windows，则在安装时可能会遇到[错误](https://github.com/index-tts/index-tts/issues/61)`ERROR: Failed building wheel for pynini` 在这种情况下，请通过conda安装pynini 
>  
> ```bash
> # 在 conda activate index-tts后
> conda install -c conda-forge pynini==2.1.6
> pip install WeTextProcessing --no-deps
> ```

作为包安装 `IndexTTS`:
```bash
cd index-tts
pip install -e .
```

1. 下载模型:

从 `huggingface-cli`下载:

```bash
huggingface-cli download IndexTeam/IndexTTS-1.5 \
  config.yaml bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints
```

推荐给中国用户. 如果下载速度慢，可以使用镜像：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

或使用 `wget`:

```bash
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml -P checkpoints
```

> [!NOTE]
> 如果你更喜欢 `IndexTTS-1.0` 模型, 请将上述命令中的`IndexTeam/IndexTTS-1.5` 替换为 `IndexTeam/IndexTTS` 。


4. 运行测试脚本:


```bash
# 请将您的提示音频放在 'test_data' 中，并将其重命名为 'input.wav'
python indextts/infer.py
```

5. 作为令行工具使用:

```bash
# 在运行此命令之前，请确保已安装 pytorch
indextts "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！" \
  --voice reference_voice.wav \
  --model_dir checkpoints \
  --config checkpoints/config.yaml \
  --output output.wav
```

使用`--help`用于查看更多选项
```bash
indextts --help
```

#### Web 演示
```bash
pip install -e ".[webui]" --no-build-isolation
python webui.py

# use another model version:
python webui.py --model_dir IndexTTS-1.5
```

打开浏览器并访问`http://127.0.0.1:7860`以查看演示。


#### 示例代码
```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
voice="reference_voice.wav"
text="大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, output_path)
```

## 感谢
1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)

## 📚 引文

🌟 如果您觉得我们的工作有帮助，请给我们打星标并引用我们的论文。

```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025}
}
```
