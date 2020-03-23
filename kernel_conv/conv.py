import torch
import torch.nn.functional as F

from .utils_kernels import extract_tensor_patches


class KernelConvUnfold2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='reflect'):
        super(KernelConvUnfold2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias, padding_mode)
        self.padding = [self.kernel_size[0] // 2,
                        self.kernel_size[0] // 2,
                        self.kernel_size[1] // 2,
                        self.kernel_size[1] // 2]
        self.kernel_fn = kernel_fn

    def __compute_shape(self, x):
        h = (x.shape[2] - self.kernel_size[0]) // self.stride[0] + 1
        w = (x.shape[3] - self.kernel_size[1]) // self.stride[1] + 1
        return h, w

    def forward(self, x):
        x = F.pad(input=x, pad=list(self.padding), mode=self.padding_mode)
        x_unf = F.unfold(input=x,
                         kernel_size=self.kernel_size,
                         stride=self.stride).transpose(1, 2)
        h, w = self.__compute_shape(x)
        out = self.kernel_fn(x_unf, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out.view(x.shape[0], self.out_channels, w, h)


class KernelConvConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='reflect'):
        super(KernelConvConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, dilation, groups, bias, padding_mode)

        self.padding = [self.kernel_size[0] // 2,
                        self.kernel_size[0] // 2,
                        self.kernel_size[1] // 2,
                        self.kernel_size[1] // 2]
        self.kernel_fn = kernel_fn

    def __compute_shape(self, x):
        h = (x.shape[2] - self.kernel_size[0]) // self.stride[0] + 1
        w = (x.shape[3] - self.kernel_size[1]) // self.stride[1] + 1
        return h, w

    def forward(self, x):
        x = F.pad(input=x, pad=list(self.padding), mode=self.padding_mode)
        x_unf = extract_tensor_patches(input=x, stride=tuple(self.stride), window_size=self.kernel_size).transpose(1, 2)
        h, w = self.__compute_shape(x)
        out = self.kernel_fn(x_unf, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out.view(x.shape[0], self.out_channels, w, h)


def kernel_wrapper(module, kernel):
    for name, layer in module._modules.items():
        kernel_wrapper(layer, kernel)
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            # Create replacement layer
            bias = layer.bias is not None
            kernel_conv2d = KernelConv2d(
                layer.in_channels, layer.out_channels, layer.kernel_size, kernel,
                layer.stride, layer.padding, layer.dilation, layer.groups, bias,
                layer.padding_mode
            )

            module._modules[name] = kernel_conv2d

