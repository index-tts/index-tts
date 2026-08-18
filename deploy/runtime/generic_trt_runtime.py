"""Generic TensorRT runtime wrapper."""

import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


class GenericTRTEngine:
    """Generic TRT engine wrapper. Call with keyword arguments matching input names."""

    def __init__(self, engine_path, device="cuda:0"):
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self._input_names = []
        self._output_names = []
        self._shape_tensor_inputs = set()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
                if self.engine.is_shape_inference_io(name):
                    self._shape_tensor_inputs.add(name)
            else:
                self._output_names.append(name)

        self._output_buffers = {}

    def __call__(self, **kwargs):
        for name in self._input_names:
            tensor = kwargs[name]
            # Shape tensors must remain on CPU
            if name in self._shape_tensor_inputs:
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                tensor = tensor.contiguous()
            else:
                tensor = tensor.contiguous()
                if not tensor.is_cuda:
                    tensor = tensor.to(self.device)
            expected_dtype = _TRT_TO_TORCH.get(
                self.engine.get_tensor_dtype(name), torch.float32
            )
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        for name in self._output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH.get(
                self.engine.get_tensor_dtype(name), torch.float32
            )
            if (name not in self._output_buffers or
                    self._output_buffers[name].shape != shape):
                self._output_buffers[name] = torch.empty(
                    shape, dtype=dtype, device=self.device
                )
            self.context.set_tensor_address(
                name, self._output_buffers[name].data_ptr()
            )

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        if len(self._output_names) == 1:
            return self._output_buffers[self._output_names[0]]
        return {name: self._output_buffers[name] for name in self._output_names}
