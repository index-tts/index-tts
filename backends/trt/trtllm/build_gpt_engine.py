"""Build the TRT-LLM engine for IndexTTS2 GPT.

Usage:
    python -m backends.trt.trtllm.build_gpt_engine --checkpoint_dir <ckpt> --output_dir <out>
"""

import sys

from backends.trt.trtllm.indextts2_gpt_model import IndexTTS2GPTForCausalLM

import tensorrt_llm.models as models_module

models_module.IndexTTS2GPTForCausalLM = IndexTTS2GPTForCausalLM
models_module.MODEL_MAP['IndexTTS2GPTForCausalLM'] = IndexTTS2GPTForCausalLM

if hasattr(models_module, '__all__'):
    if 'IndexTTS2GPTForCausalLM' not in models_module.__all__:
        models_module.__all__.append('IndexTTS2GPTForCausalLM')

from tensorrt_llm.commands.build import main

if __name__ == '__main__':
    sys.exit(main())
