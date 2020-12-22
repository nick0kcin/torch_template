from segmentation_models_pytorch import Unet
import torch
import torch.nn as nn


def dict_forward_class(forward_fn):
    def forward(self, X):

        stages = [
            nn.Identity(),
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ]

        features = []
        with torch.no_grad():
            for i in range(self.encoder._depth):
                stages[i].eval()
                X = stages[i](X)
                features.append(X)

        # stages[-1].eval()
        X = stages[-1](X)
        features.append(X)

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.decoder.center(head)
        if len(self.decoder.blocks) == 5:
            del self.decoder.blocks[-1]
            del self.decoder.blocks[-1]
        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        decoder_output = x #self.decoder(*features)

        # p = 0.1 * decoder_output.shape[-2] * decoder_output.shape[-1] / (73 * 64) #0.1
        p = 0.5
        if self.training:
            mask = torch.bernoulli((1-p)*torch.ones((1,1,decoder_output.shape[-2], decoder_output.shape[-1]), device=decoder_output.device))
            masks = self.segmentation_head(decoder_output * mask)
        else:
            masks = self.segmentation_head(decoder_output * (1-p))
        masks = torch.nn.functional.upsample(masks, scale_factor=4)
        #
        if self.classification_head is not None:
            labels = self.classification_head(head)
            return [{"mask": masks, "label": labels}]

        return [{"mask": masks}]
    return forward

class VaeUnet(Unet):
    def __init__(self, **kwargs):
        super(VaeUnet, self).__init__(**kwargs)
        self.mean_path = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=True), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1, bias=True))
        self.std_path = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=True), nn.ReLU(), nn.Conv2d(16, 1, 3, padding=1, bias=True))
        self.epoch = -1

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.decoder.center(head)
        if len(self.decoder.blocks) == 5:
            del self.decoder.blocks[-1]
            del self.decoder.blocks[-1]
        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        decoder_output = x  # self.decoder(*features)

        mean = self.mean_path(decoder_output)
        log_std = self.std_path(decoder_output)

        if self.training and self.epoch > 2:
            x = mean + torch.exp(log_std) * torch.normal(torch.zeros_like(log_std), torch.ones_like(log_std))
        else:
            x = mean

        masks = self.segmentation_head(x)
        masks = torch.nn.functional.upsample(masks, scale_factor=4)
        # kl = 0.5 * (torch.exp(log_std * 2) + mean ** 2 - 2 * log_std)
        if self.training:
            a = (torch.exp(log_std * 2) + 1e-12) / (mean ** 2 + 1e-09)
            k1 = 0.63576
            k2 = 1.87320
            k3 = 1.48695
            kl = k1 * (k2 + k3 * torch.log(a)).sigmoid() - 0.5 * torch.log(1 + 1 / a) - k1
            return [{"mask": masks, "kl": -0.1 * kl}]
        return [{"mask": masks}]



Unet.forward = dict_forward_class(Unet.forward)
