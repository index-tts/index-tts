from indextts.infer import IndexTTS

if __name__ == "__main__":
    prompt_wav="tests/sample_prompt.wav"
    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, use_cuda_kernel=False)
    # 单音频推理测试
    text="晕 XUAN4 是 一 种 GAN3 觉"
    tts.infer(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'
    tts.infer(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text="There is a vehicle arriving in dock number 7?"
    tts.infer(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

    # 并行推理测试
    text="亲爱的伙伴们，大家好！每一次的努力都是为了更好的未来，要善于从失败中汲取经验，让我们一起勇敢前行,迈向更加美好的明天！"
    tts.infer_fast(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text="The weather is really nice today, perfect for studying at home.Thank you!"
    tts.infer_fast(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)
    text='''叶远随口答应一声，一定帮忙云云。
教授看叶远的样子也知道，这事情多半是黄了。
谁得到这样的东西也不会轻易贡献出来，这是很大的一笔财富。
叶远回来后，又自己做了几次试验，发现空间湖水对一些外伤也有很大的帮助。
找来一只断了腿的兔子，喝下空间湖水，一天时间，兔子就完全好了。
还想多做几次试验，可是身边没有试验的对象，就先放到一边，了解空间湖水可以饮用，而且对人有利，这些就足够了。
感谢您的收听，下期再见！
    '''.replace("\n", "")
    tts.infer_fast(audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

