from torchvision.transforms.transforms import Lambda
from numpy.random import random, uniform, randint
import math
import torch


def random_erase_factory( n, p, scale, ratio):
    def random_erase(img):
        area = img.shape[-1] * img.shape[-2]
        for i in range(n):
            if random() < p:
                current_area = uniform(scale[0], scale[1]) * area
                current_ratio = uniform(ratio[0], ratio[1])
                h = int(round(math.sqrt(current_area * current_ratio)))
                w = int(round(math.sqrt(current_area / current_ratio)))
                if w < img.shape[-1] and h < img.shape[-2]:
                    i = randint(0, img.shape[-2] - h)
                    j = randint(0, img.shape[-1] - w)
                    img[:, i:i+h, j:j+w] = torch.empty([img.shape[0], h, w], dtype=torch.float32).normal_()
        return img
    return random_erase


def random_erase_transform(n=10, p=0.5, scale=(0.001, 0.01), ratio=(0.5,2)):
    return Lambda(random_erase_factory(n, p, scale, ratio))







