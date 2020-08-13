import os
import math
import logging

import torch
import torch.nn as nn
from torch.hub import load
import timm
from efficientnet_pytorch import EfficientNet
from torchsummary import torchsummary
from torchvision.models import resnet18, resnet34, resnet101, resnext50_32x4d, densenet121, shufflenet_v2_x1_0


class ResNest(nn.Module):

    def __init__(self, pretrained, num, dr_rate, cls, num_classes):
        super(ResNest, self).__init__()
        lst = torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        self.backbone = torch.hub.load('zhanghang1989/ResNeSt', f'resnest50_fast_1s1x64d', pretrained=pretrained)
        # self.backbone = EfficientNet.from_pretrained('efficientnet-b1')
        self.dr = nn.Dropout(dr_rate)
        print(self.backbone)
        self.backbone.fc = nn.Sequential(*[self.dr, nn.Linear(cls, num_classes)])
        # torchsummary.summary(self.backbone.cuda(), (3, 256, 256))

        # self.fc = nn.Sequential(*[
        #     nn.Dropout(0.2),
        #     nn.Linear(1280, num_classes)
        # ])
        # for m in self.fc.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight)

        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def features(self):
        params = {}
        params.update(dict(self.backbone.conv1.named_parameters()))
        params.update(dict(self.backbone.bn1.named_parameters()))
        params.update(dict(self.backbone.layer1.named_parameters()))
        params.update(dict(self.backbone.layer2.named_parameters()))
        params.update(dict(self.backbone.layer3.named_parameters()))
        params.update(dict(self.backbone.layer4.named_parameters()))

        return params


    def forward(self, x):
        # with torch.no_grad():
        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)
        #
        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
        #
        # x = self.backbone.avgpool(x)
        # x = self.dr(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)


        # x = self.backbone.conv1(x)
        # x = self.backbone.maxpool(x)
        # x = self.backbone.stage2(x)
        # x = self.backbone.stage3(x)
        # x = self.backbone.stage4(x)
        # x = self.backbone.conv5(x)
        # x = x.mean([2, 3])  # globalpool
        # # x = self.dr(x)
        # x = self.fc(x)

        # with torch.no_grad():
        #     x = self.backbone.conv1(x)
        #     x = self.backbone.bn1(x)
        #     x = self.backbone.relu(x)
        #     x = self.backbone.maxpool(x)
        #     x = self.backbone.layer1(x)
        #     x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
        # x = self.backbone.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.backbone(x)

        # x = self.backbone.extract_features(x)
        # x = self.backbone._avg_pooling(x).view(x.shape[0], -1)
        # x = self.fc(x)

        return [{"y": x}]
