from torch.nn import Module, Sequential, AdaptiveAvgPool2d, Linear, Sigmoid, Conv2d, ReLU


class Attention(Module):

    def __init__(self, channels, channel_down_rate):
        super(Attention, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.channelwise_path = Sequential(*[
            Linear(channels, max(1, channels // channel_down_rate), bias=False), ReLU(),
            Linear(max(1, channels // channel_down_rate), channels, bias=False), Sigmoid()
        ])

        self.pointwise_path = Sequential(*[
            Conv2d(channels, 1, kernel_size=1, bias=False), Sigmoid()
        ])

    def forward(self, x):
        y = self.avg_pool(x).view(x.shape[0], x.shape[1])
        pointwise_mask = self.pointwise_path(x)
        channelwise_mask = self.channelwise_path(y)
        return x * channelwise_mask.view(x.shape[0], x.shape[1], 1, 1), pointwise_mask
