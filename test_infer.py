from indextts.infer import IndexTTS

if __name__ == "__main__":
    import random

    # 30条短视频口播脚本片段
    script_fragments = [
        "今天给大家带来一个超实用的生活小技巧",
        "不知道你有没有遇到过这样的问题",
        "这个方法简单易学，效果却出奇的好",
        "我偶然间发现了这个秘密，一定要分享给你们",
        "很多人都不知道，其实只需要这样做",
        "这个技巧可以帮你节省大量时间",
        "学会这一招，再也不用担心这个问题了",
        "专业人士都在用的方法，今天教给大家",
        "用过的人都说好，真的是太神奇了",
        "这可能是你见过最简单有效的解决方案",
        "花了很长时间才总结出来的经验",
        "不要小看这个小技巧，它真的能改变你的生活",
        "很多人花大价钱解决的问题，其实可以这样简单处理",
        "这个方法我已经用了很多年，效果非常好",
        "你可能不相信，但试过之后就会爱上这个方法",
        "这个小窍门真的是太实用了，强烈推荐给大家",
        "很多人问我是怎么做到的，今天就分享给大家",
        "这个技巧真的是改变了我的生活方式",
        "如果你也有这个困扰，不妨试试这个方法",
        "这个方法简单到你可能不敢相信",
        "用了这个方法后，我再也不用担心这个问题了",
        "朋友们都惊讶于这个方法的效果",
        "这可能是解决这个问题最省钱的方式",
        "很多人都不知道这个小技巧，今天我来告诉大家",
        "这个方法不仅简单，而且效果立竿见影",
        "我以前也不知道，学会后真的是太方便了",
        "这个技巧真的是太实用了，一定要收藏",
        "如果你正在为这个问题烦恼，那这个视频一定要看完",
        "这个方法真的是太神奇了，一试就爱上",
        "学会这个技巧，你会感谢我的",
    ]

    # 脚本类型分类（可以根据需要进行分类）
    opening_lines = [0, 1, 3, 4, 7, 17, 19, 27]  # 开场白
    problem_statements = [1, 8, 13, 18, 22, 27]  # 问题陈述
    solutions = [2, 5, 6, 9, 12, 14, 20, 24, 25]  # 解决方案
    benefits = [5, 6, 8, 10, 11, 15, 16, 21, 23, 28, 29]  # 好处描述
    closing_lines = [15, 26, 28, 29]  # 结束语

    def generate_script(num_fragments=5, avoid_repetition=True):
        """
        生成随机组合的口播脚本

        参数:
        num_fragments (int): 要组合的片段数量
        avoid_repetition (bool): 是否避免重复使用片段

        返回:
        str: 组合后的口播脚本
        """
        if num_fragments > len(script_fragments) and avoid_repetition:
            num_fragments = len(script_fragments)
            print(f"警告: 请求的片段数量超过了可用片段总数，已调整为 {num_fragments}")

        # 选择片段
        if avoid_repetition:
            selected_indices = random.sample(
                range(len(script_fragments)), num_fragments
            )
        else:
            selected_indices = [
                random.randint(0, len(script_fragments) - 1)
                for _ in range(num_fragments)
            ]

        selected_fragments = [script_fragments[i] for i in selected_indices]

        # 组合脚本
        combined_script = "。".join(selected_fragments)

        return combined_script

    def generate_structured_script():
        """
        生成结构化的口播脚本，包含开场白、问题陈述、解决方案、好处描述和结束语

        返回:
        str: 结构化的口播脚本
        """
        opening = script_fragments[random.choice(opening_lines)]
        problem = script_fragments[random.choice(problem_statements)]
        solution = script_fragments[random.choice(solutions)]
        benefit = script_fragments[random.choice(benefits)]
        closing = script_fragments[random.choice(closing_lines)]

        structured_script = f"{opening}。{problem}。{solution}。{benefit}。{closing}。"

        return structured_script

    def generate_multiple_scripts(count=5, structured=False):
        """
        生成多个口播脚本

        参数:
        count (int): 要生成的脚本数量
        structured (bool): 是否生成结构化脚本

        返回:
        list: 生成的脚本列表
        """
        scripts = []
        for i in range(count):
            if structured:
                script = generate_structured_script()
            else:
                script = generate_script(random.randint(3, 6))
            scripts.append(f"脚本 {i+1}: {script}")

        return scripts

    print("随机组合的口播脚本:")
    random_scripts = generate_multiple_scripts(1, structured=False)

    # prompt_wav_path = "outputs/upload_references"
    prompt_wav_path = "/Users/wangxianchen/Desktop/tts_reference/upload_references"

    prompt_id_list = [
        "ks_曾鼎全",
        # "QY13323323_清盐",
        # "柴柴_男声",
        # "是阿殇啦",
        # "JINQQ124_东北话",
        # "简妮特",
        # "仙_男声",
    ]

    prompt_file_name = "sample1.mp3"
    # # text="晕 XUAN4 是 一 种 GAN3 觉"
    text = "各位老铁们，欢迎新进直播间的老铁们！库存真的不多了，咱们1号、2号链接赶紧拍起来！1号链接的炸鸡三兄弟配送，性价比超高，绝对值得您拥有！无论是和朋友一起分享，还是犒劳自己，都值得您拥有！而2号2号链接则更加实惠，让您省钱又省心！炸鸡三兄弟配送，价格超低，绝对值得您拥有！不管是屯个三单，五单分开用，还是一起用，都是可以的！赶紧拍，赶紧屯，让您的味蕾享受美味的同时，也能省下一大笔钱！"
    # # text_2 = "来刘炭ZHANG3吃烤肉，肉质鲜嫩多汁，秘制酱料香到上头！炭火现烤滋滋冒油，人均50吃到扶墙走！吃货们快约上姐妹冲，大口吃肉才叫爽！"

    tts = IndexTTS(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        is_fp16=True,
        use_cuda_kernel=False,
        compile=False,
        # device="cpu",
    )

    for script in random_scripts:
        for prompt_id in prompt_id_list:
            prompt_wav = prompt_wav_path + "/" + prompt_id + "/" + prompt_file_name
            print(prompt_wav)

            tts.infer_real_stream(
                audio_prompt=prompt_wav,
                text=text,
                output_path=f"./outputs/results/{prompt_id}/gen_{script[:2]}_{random.randint(1,200)}_text.wav",
                verbose=True,
                prompt_id=prompt_id,
            )

            # tts.infer_stream(
            #     audio_prompt=prompt_wav,
            #     text=text,
            #     output_path=f"./outputs/results/{prompt_id}/gen_{script[:2]}_{random.randint(1,200)}_text.wav",
            #     verbose=True,
            #     prompt_id=prompt_id,
            # )
            # tts.infer_fast(
            #     audio_prompt=prompt_wav,
            #     text=script,
            #     output_path=f"./outputs/results/{prompt_id}/gen_{script[:2]}_{random.randint(1,200)}_text.wav",
            #     verbose=False,
            #     prompt_id=prompt_id,
            # )
