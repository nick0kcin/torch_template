import timm
import torch.nn as nn


class EffNet(nn.Module):

    def __init__(self, num):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model(f'tf_efficientnet_b{num}', drop_path_rate=0.4, pretrained=True)
        self.dr = nn.Dropout(0)
        self.backbone.classifier = nn.Sequential(*[self.dr])

    def features(self):
        params = {}
        params.update(dict(self.backbone.conv_stem.named_parameters()))
        params.update(dict(self.backbone.bn1.named_parameters()))
        params.update(dict(self.backbone.blocks.named_parameters()))
        params.update(dict(self.backbone.act1.named_parameters()))
        params.update(dict(self.backbone.conv_head.named_parameters()))
        params.update(dict(self.backbone.bn2.named_parameters()))
        params.update(dict(self.backbone.act2.named_parameters()))
        return params

    def forward(self, x):
        x = self.backbone(x)

        return [{"y": x}]
