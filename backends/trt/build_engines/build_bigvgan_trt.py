"""Build a TensorRT engine from the BigVGAN ONNX model.

Usage:
    python build_bigvgan_trt.py [--onnx bigvgan.onnx] [--output bigvgan.engine] [--fp16]
"""

import argparse
import os
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path, engine_path, fp16=True,
                 min_T=1, opt_T=1024, max_T=4096,
                 min_batch=1, opt_batch=1, max_batch=16):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    onnx_abs = os.path.abspath(onnx_path)

    print(f"Parsing ONNX model from {onnx_abs}...")
    success = parser.parse_from_file(onnx_abs)
    if not success:
        for i in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model")

    print(f"Network has {network.num_inputs} inputs, {network.num_outputs} outputs")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input {i}: name={inp.name}, shape={inp.shape}, dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output {i}: name={out.name}, shape={out.shape}, dtype={out.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 32 << 30)

    if fp16:
        if builder.platform_has_fast_fp16:
            print("Enabling FP16 mode")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("WARNING: FP16 not supported on this platform, using FP32")

    profile = builder.create_optimization_profile()

    # mel: (B, 80, T_mel)
    profile.set_shape("mel",
                       min=(min_batch, 80, min_T),
                       opt=(opt_batch, 80, opt_T),
                       max=(max_batch, 80, max_T))

    config.add_optimization_profile(profile)

    print(f"Building TRT engine (B={min_batch}/{opt_batch}/{max_batch}, "
          f"T={min_T}/{opt_T}/{max_T})...")
    print("This may take several minutes...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"TRT engine saved to {engine_path} ({engine_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build TRT engine from BigVGAN ONNX")
    parser.add_argument("--onnx", default="bigvgan.onnx")
    parser.add_argument("--output", default="bigvgan.engine")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--min-T", type=int, default=1)
    parser.add_argument("--opt-T", type=int, default=1024)
    parser.add_argument("--max-T", type=int, default=4096)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=16)
    args = parser.parse_args()

    build_engine(args.onnx, args.output, args.fp16,
                 args.min_T, args.opt_T, args.max_T,
                 args.min_batch, args.opt_batch, args.max_batch)


if __name__ == "__main__":
    main()
