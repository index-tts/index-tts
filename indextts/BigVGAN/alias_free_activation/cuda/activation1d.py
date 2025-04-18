import torch
import torch.nn as nn
from torch import Tensor
from torch.library import Library, impl
from indextts.BigVGAN.alias_free_activation.cuda import load
from indextts.BigVGAN.alias_free_activation.torch.resample import (
    DownSample1d,
    UpSample1d,
)

# 加载CUDA扩展
anti_alias_activation_cuda = load.load()

# 定义库和前向算子
lib = Library("anti_alias_activation", "DEF")
lib.define(
    "fused_anti_alias_forward(Tensor input, Tensor up_ftr, Tensor down_ftr, Tensor alpha, Tensor beta) -> Tensor"
)


# 注册CUDA实现
@impl(lib, "fused_anti_alias_forward", "CUDA")
def fused_anti_alias_forward_cuda(
    input: Tensor, up_ftr: Tensor, down_ftr: Tensor, alpha: Tensor, beta: Tensor
) -> Tensor:
    return anti_alias_activation_cuda.forward(input, up_ftr, down_ftr, alpha, beta)


# 添加Meta实现，这是编译时形状推断所必需的
@impl(lib, "fused_anti_alias_forward", "Meta")
def fused_anti_alias_forward_meta(
    input: Tensor, up_ftr: Tensor, down_ftr: Tensor, alpha: Tensor, beta: Tensor
) -> Tensor:
    # 计算输出形状
    batch_size, channels, length = input.shape
    up_ratio = 2  # 假设上采样比例为2，与原始代码保持一致
    down_ratio = 2  # 假设下采样比例为2，与原始代码保持一致

    # 上采样后的长度
    up_length = length * up_ratio
    # 下采样后的长度
    down_length = up_length // down_ratio

    # 创建Meta张量作为输出
    return input.new_empty(batch_size, channels, down_length)


class FusedAntiAliasActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, up_ftr, down_ftr, alpha, beta):
        return torch.ops.anti_alias_activation.fused_anti_alias_forward(
            inputs, up_ftr, down_ftr, alpha, beta
        )

    @staticmethod
    def backward(ctx, output_grads):
        raise NotImplementedError
        return output_grads, None, None

class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        fused: bool = True,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)
        self.fused = fused

    def forward(self, x: Tensor) -> Tensor:
        if not self.fused:
            x = self.upsample(x)
            x = self.act(x)
            x = self.downsample(x)
            return x
        else:
            if self.act.__class__.__name__ == "Snake":
                beta = self.act.alpha.data
            else:
                beta = self.act.beta.data
            alpha = self.act.alpha.data

            if not self.act.alpha_logscale:
                alpha = torch.log(alpha)
                beta = torch.log(beta)

            return FusedAntiAliasActivation.apply(
                x, self.upsample.filter, self.downsample.lowpass.filter, alpha, beta
            )
