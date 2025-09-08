## FACodec: Speech Codec with Attribute Factorization used for NaturalSpeech 3

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2403.03100.pdf)
[![demo](https://img.shields.io/badge/FACodec-Demo-red)](https://speechresearch.github.io/naturalspeech3/)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-pink)](https://huggingface.co/amphion/naturalspeech3_facodec)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/amphion/naturalspeech3_facodec)

## Overview

FACodec is a core component of the advanced text-to-speech (TTS) model NaturalSpeech 3. FACodec converts complex speech waveform into disentangled subspaces representing speech attributes of content, prosody, timbre, and acoustic details and reconstruct high-quality speech waveform from these attributes. FACodec decomposes complex speech into subspaces representing different attributes, thus simplifying the modeling of speech representation.

Research can use FACodec to develop different modes of TTS models, such as non-autoregressive based discrete diffusion (NaturalSpeech 3) or autoregressive models (like VALL-E).

<br>
<div align="center">
<img src="../../../imgs/ns3/ns3_overview.png" width="65%">
</div>
<br>

<br>
<div align="center">
<img src="../../../imgs/ns3/ns3_facodec.png" width="100%">
</div>
<br>

## Useage

Download the pre-trained FACodec model from HuggingFace: [Pretrained FACodec checkpoint](https://huggingface.co/amphion/naturalspeech3_facodec)

Install Amphion
```bash
git clone https://github.com/open-mmlab/Amphion.git
```

Few lines of code to use the pre-trained FACodec model
```python
from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download

fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

fa_encoder.eval()
fa_decoder.eval()

```

Inference
```python
test_wav_path = "test.wav"
test_wav = librosa.load(test_wav_path, sr=16000)[0]
test_wav = torch.from_numpy(test_wav).float()
test_wav = test_wav.unsqueeze(0).unsqueeze(0)

with torch.no_grad():

    # encode
    enc_out = fa_encoder(test_wav)
    print(enc_out.shape)

    # quantize
    vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
    
    # latent after quantization
    print(vq_post_emb.shape)
    
    # codes
    print("vq id shape:", vq_id.shape)
    
    # get prosody code
    prosody_code = vq_id[:1]
    print("prosody code shape:", prosody_code.shape)
    
    # get content code
    cotent_code = vq_id[1:3]
    print("content code shape:", cotent_code.shape)
    
    # get residual code (acoustic detail codes)
    residual_code = vq_id[3:]
    print("residual code shape:", residual_code.shape)
    
    # speaker embedding
    print("speaker embedding shape:", spk_embs.shape)

    # decode (recommand)
    recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
    print(recon_wav.shape)
    sf.write("recon.wav", recon_wav[0][0].cpu().numpy(), 16000)
```

FACodec can achieve zero-shot voice conversion with FACodecEncoderV2/FACodecDecoderV2 or FACodecRedecoder
```python
from Amphion.models.codec.ns3_codec import FACodecEncoderV2, FACodecDecoderV2

# Same parameters as FACodecEncoder/FACodecDecoder
fa_encoder_v2 = FACodecEncoderV2(...)
fa_decoder_v2 = FACodecDecoderV2(...)

encoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
decoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")

fa_encoder_v2.load_state_dict(torch.load(encoder_v2_ckpt))
fa_decoder_v2.load_state_dict(torch.load(decoder_v2_ckpt))

with torch.no_grad():
  enc_out_a = fa_encoder_v2(wav_a)
  prosody_a = fa_encoder_v2.get_prosody_feature(wav_a)
  enc_out_b = fa_encoder_v2(wav_b)
  prosody_b = fa_encoder_v2.get_prosody_feature(wav_b)

  vq_post_emb_a, vq_id_a, _, quantized, spk_embs_a = fa_decoder_v2(
      enc_out_a, prosody_a, eval_vq=False, vq=True
  )
  vq_post_emb_b, vq_id_b, _, quantized, spk_embs_b = fa_decoder_v2(
      enc_out_b, prosody_b, eval_vq=False, vq=True
  )

  vq_post_emb_a_to_b = fa_decoder_v2.vq2emb(vq_id_a, use_residual=False)
  recon_wav_a_to_b = fa_decoder_v2.inference(vq_post_emb_a_to_b, spk_embs_b)
```

or

```python
from Amphion.models.codec.ns3_codec import FACodecRedecoder

fa_redecoder = FACodecRedecoder()

redecoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_redecoder.bin")

fa_redecoder.load_state_dict(torch.load(redecoder_ckpt))

with torch.no_grad():
    enc_out_a = fa_encoder(wav_a)
    enc_out_b = fa_encoder(wav_b)

    vq_post_emb_a, vq_id_a, _, quantized_a, spk_embs_a = fa_decoder(enc_out_a, eval_vq=False, vq=True)
    vq_post_emb_b, vq_id_b, _, quantized_b, spk_embs_b = fa_decoder(enc_out_b, eval_vq=False, vq=True)

    # convert speaker
    vq_post_emb_a_to_b = fa_redecoder.vq2emb(vq_id_a, spk_embs_b, use_residual=False)
    recon_wav_a_to_b = fa_redecoder.inference(vq_post_emb_a_to_b, spk_embs_b)

    sf.write("recon_a_to_b.wav", recon_wav_a_to_b[0][0].cpu().numpy(), 16000)
```

## Q&A

Q1: What audio sample rate does FACodec support? What is the hop size? How many codes will be generated for each frame?

A1: FACodec supports 16KHz speech audio. The hop size is 200 samples, and (16000/200) * 6 (total number of codebooks) codes will be generated for each frame.

Q2: Is it possible to train an autoregressive TTS model like VALL-E using FACodec?

A2: Yes. In fact, the authors of NaturalSpeech 3 have already employ explore the autoregressive generative model for discrete token generation with FACodec. They use an autoregressive language model to generate prosody codes, followed by a non-autoregressive model to generate the remaining content and acoustic details codes.

Q3: Is it possible to train a latent diffusion TTS model like NaturalSpeech2 using FACodec?

A3: Yes. You can use the latent getted after quanzaition as the modelling target for the latent diffusion model.

Q4: Can FACodec compress and reconstruct audio from other domains? Such as sound effects, music, etc.

A4: Since FACodec is designed for speech, it may not be suitable for other audio domains. However, it is possible to use the FACodec model to compress and reconstruct audio from other domains, but the quality may not be as good as the original audio.

Q5: Can FACodec be used for content feature for some other tasks like voice conversion?

A5: I think the answer is yes. Researchers can use the content code of FACodec as the content feature for voice conversion. We hope to see more research in this direction.

## Citations

If you use our FACodec model, please cite the following paper:

```bibtex
@article{ju2024naturalspeech,
  title={NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models},
  author={Ju, Zeqian and Wang, Yuancheng and Shen, Kai and Tan, Xu and Xin, Detai and Yang, Dongchao and Liu, Yanqing and Leng, Yichong and Song, Kaitao and Tang, Siliang and others},
  journal={arXiv preprint arXiv:2403.03100},
  year={2024}
}

@article{zhang2023amphion,
      title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit}, 
      author={Xueyao Zhang and Liumeng Xue and Yicheng Gu and Yuancheng Wang and Haorui He and Chaoren Wang and Xi Chen and Zihao Fang and Haopeng Chen and Junan Zhang and Tze Ying Tang and Lexiao Zou and Mingxuan Wang and Jun Han and Kai Chen and Haizhou Li and Zhizheng Wu},
      journal={arXiv},
      year={2024},
      volume={abs/2312.09911}
}
```

