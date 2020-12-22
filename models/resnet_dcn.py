import math

import torch
import torch.nn as nn
from torchvision.models import resnet18

from .attention import Attention

BN_MOMENTUM = 0.1


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PoseResNet(nn.Module):

    def __init__(self, heads, head_conv, attention=False, no_down=False):
        self.inplanes = 512
        self.heads = heads
        self.deconv_with_bias = False
        self.attention = attention
        super(PoseResNet, self).__init__()
        self.backbone = resnet18(True)
        # self.encoder = Unet("efficientnet-b5").encoder
        self.backbone.fc = torch.nn.Sequential(*[])
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )
        if no_down:
            self.up = nn.Sequential(*[
                nn.Conv2d(64, 32, (3, 3), padding=1),
                nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2)
            ])
        else:
            self.up = None

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True))
                if 'center' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(32, classes, kernel_size=1, stride=1, padding=0, bias=True)
                if 'center' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # fc = DCN(self.inplanes, planes, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
            fc = nn.Conv2d(self.inplanes, planes, kernel_size=(3, 3), stride=1, padding=1)
            fill_fc_weights(fc)
            # ups = nn.UpsamplingBilinear2d(scale_factor=2)
            # up = nn.Conv2d(planes, planes, kernel_size=kernel - 1, stride=1, padding=(kernel-2)//2)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            if self.attention:
                layers.append(Attention(planes, 16))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(ups)
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # with torch.no_grad():
        #     x = self.encoder(x)
        m = x
        # x = x[-1]
        mask = None
        for layer in self.deconv_layers:
            x = layer(x)
            if isinstance(x, tuple):
                if mask is None:
                    mask = x[1]
                else:
                    mask = x[1]
                x = x[0] * mask + x[0]

        if self.up:
            x = self.up(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        ret["features"] = m
        if "super" in self.heads:
            ret["super"][:, 0, :, :] = (ret["center"][:, 0, :, :].sigmoid() +
                                        ret["center"][:, 2, :, :].sigmoid() +
                                        ret["center"][:, 3, :, :].sigmoid()) / 3
            ret["super"][:, 1, :, :] = (ret["center"][:, 1, :, :].sigmoid() +
                                        ret["center"][:, 5, :, :].sigmoid() +
                                        ret["center"][:, 4, :, :].sigmoid()) / 3
        if "mask" in self.heads:
            ret["mask"] = mask
        return [ret]

    def get_head_params(self):
        params = []
        if self.up:
            params += list(self.up.parameters())
        for head in self.heads:
            params += list(self.__getattr__(head).parameters())
        return params

    def get_deconv_params(self):
        params = []
        for layer in list(self.deconv_layers.children())[0:]:
            params += list(layer.parameters())
        return params

    def set_group_param(self, group):
        params_map = {0: self.get_head_params, 1: self.get_deconv_params, 2: self.parameters}
        return params_map[group]()


def get_pose_net(num_layers, heads, head_conv=256, attention=False):
    model = PoseResNet(heads, head_conv=head_conv, attention=attention)
    return model
