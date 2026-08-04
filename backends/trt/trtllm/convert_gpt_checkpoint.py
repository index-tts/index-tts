"""Convert IndexTTS2 GPT checkpoint to TRT-LLM format.

Usage:
    python -m backends.trt.trtllm.convert_gpt_checkpoint --model_dir checkpoints --output_dir tllm_checkpoint --dtype float16
"""

import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from backends.trt.trtllm.indextts2_gpt_model import IndexTTS2GPTForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='checkpoints')
    parser.add_argument(
        '--gpt_variant',
        default='gpt2',
        choices=['gpt2', 'santacoder', 'starcoder', 'starcoder2', 'persimmon',
                 'kosmos-2', 'nemotron'],
    )
    parser.add_argument('--tp_size', type=int, default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size', type=int, default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['auto', 'float16', 'bfloat16', 'float32'])
    parser.add_argument("--load_model_on_cpu", action="store_true")
    parser.add_argument('--use_parallel_embedding', action="store_true", default=False)
    parser.add_argument('--embedding_sharding_dim', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_weight_only', default=False, action="store_true",
                        help='Quantize weights for GEMMs to INT4/INT8')
    parser.add_argument('--weight_only_precision', const='int8', type=str,
                        nargs='?', default='int8', choices=['int8', 'int4'])
    parser.add_argument('--output_dir', type=str, default='tllm_checkpoint')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--log_level', type=str, default='info')

    args = parser.parse_args()
    tensorrt_llm.logger.set_level(args.log_level)
    return args


def args_to_quant_config(args):
    quant_config = QuantConfig()
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            quant_config.quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            quant_config.quant_algo = QuantAlgo.W4A16
    return quant_config


def convert_and_save_hf(args):
    world_size = args.tp_size * args.pp_size

    override_fields = {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'gpt_variant': args.gpt_variant,
    }

    quant_config = args_to_quant_config(args)

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)
        model = IndexTTS2GPTForCausalLM.from_hugging_face(
            os.path.join(args.model_dir, "config.yaml"),
            args.model_dir,
            None,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            **override_fields)
        model.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del model

    execute(args.workers, [convert_and_save_rank] * world_size, args)
    release_gc()


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(exceptions) == 0, "Checkpoint conversion failed, please check error log."


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
