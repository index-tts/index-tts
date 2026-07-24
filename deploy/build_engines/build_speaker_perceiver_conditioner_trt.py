"""Build a TensorRT engine from the speaker perceiver conditioner ONNX model.

Usage:
    python build_speaker_perceiver_conditioner_trt.py [--onnx speaker_perceiver_conditioner.onnx] [--output speaker_perceiver_conditioner.engine]
"""

import argparse
import os
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path, engine_path, fp16=True,
                 min_T=3, opt_T=500, max_T=1500,
                 min_batch=1, opt_batch=1, max_batch=8):
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 (with perceiver layers forced FP32)")
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        perceiver_start = False
        marked = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if "conditioning_encoder.after_norm" in layer.name:
                perceiver_start = True
            if not perceiver_start:
                continue
            if layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE,
                              trt.LayerType.CONCATENATION):
                continue
            is_index_layer = False
            for j in range(layer.num_outputs):
                if layer.get_output(j).dtype in (trt.int32, trt.int64, trt.bool):
                    is_index_layer = True
                    break
            if is_index_layer:
                continue
            try:
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                marked += 1
            except Exception:
                pass
        print(f"  Marked {marked} perceiver layers as FP32")
    elif fp16:
        print("WARNING: FP16 not supported on this platform, using FP32")

    profile = builder.create_optimization_profile()

    # speech_input: (B, T, 1024)
    profile.set_shape("speech_input",
                       min=(min_batch, min_T, 1024),
                       opt=(opt_batch, opt_T, 1024),
                       max=(max_batch, max_T, 1024))
    # lengths: (B,)
    profile.set_shape("lengths",
                       min=(min_batch,),
                       opt=(opt_batch,),
                       max=(max_batch,))

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
    parser = argparse.ArgumentParser(description="Build TRT engine from speaker perceiver conditioner ONNX")
    parser.add_argument("--onnx", default="speaker_perceiver_conditioner.onnx")
    parser.add_argument("--output", default="speaker_perceiver_conditioner.engine")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--min-T", type=int, default=3)
    parser.add_argument("--opt-T", type=int, default=500)
    parser.add_argument("--max-T", type=int, default=1500)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=8)
    args = parser.parse_args()

    build_engine(args.onnx, args.output, args.fp16,
                 args.min_T, args.opt_T, args.max_T,
                 args.min_batch, args.opt_batch, args.max_batch)


if __name__ == "__main__":
    main()
