from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_binary_kernel(window_size):
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def _compute_zero_padding(kernel_size):
    computed: Tuple[int, ...] = tuple([(k - 1) // 2 for k in kernel_size])
    return computed[0], computed[1]


class MedianBlur(nn.Module):

    def __init__(self, kernel_size):
        super(MedianBlur, self).__init__()
        self.kernel: torch.Tensor = _compute_binary_kernel(kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    def forward(self, input):
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # map the local window to single vector
        features: torch.Tensor = F.conv2d(
            input, kernel, padding=self.padding, stride=1, groups=c)
        features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        median: torch.Tensor = torch.median(features, dim=2)[0]
        return median


def median_blur(input, kernel_size) -> torch.Tensor:
    return MedianBlur(kernel_size)(input)
