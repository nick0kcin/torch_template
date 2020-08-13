from numpy.random import random, uniform, randint, choice
import math
import torch
import cv2
import numpy as np
import os
from albumentations.augmentations.transforms import Lambda
from albumentations import  ImageOnlyTransform


def random_lines_factory(min_lines = 5, max_lines = 80, min_length = 20, max_length = 100, p =0.5):
    def random_lines(img, **kwargs):
        n = randint(min_lines, max_lines)
        mask = np.zeros_like(img)
        for i in range(n):
            length = randint(min_length, max_length)
            x1, y1 = randint(0, img.shape[0] - 1), randint(0, img.shape[1] - 1)
            angle = randint(0, 2 * math.pi)
            x2 = int(x1 + length * math.cos(angle))
            y2 = int(y1 + length * math.sin(angle))
            try:
                mask = cv2.line(np.zeros(img.shape).astype(np.uint8).copy(), (x1, y1), (x2, y2), (255, 255, 255), randint(1, 3))
            except TypeError:
                print("a")
        img = img & ~mask if random() < p else img | mask
        return img
    return random_lines



def random_mmm_factory(p=0.5):
    def random_mmm(img, **kwargs):
        c1 = randint(0, img.shape[0] - 1), randint(0, img.shape[1] - 1)
        tf = randint(1, 3)
        mask = np.zeros_like(img)
        # t_size = cv2.getTextSize("mm", 0, fontScale=tf / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, colors[rect["category_id"]], -1, cv2.LINE_AA)  # filled
        try:
            mask = cv2.putText(np.zeros(img.shape).astype(np.uint8).copy(), "mm", (c1[0], c1[1] - 2), 0, 2 / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        except TypeError:
            print("b")
        img = img & ~mask if random() < p else img | mask
        return img
    return random_mmm


def random_microscope_factory():
    def random_microscope(img, **kwargs):
        mask = np.zeros_like(img)
        dx = randint(0, img.shape[0] // 4)
        dy = randint(0, img.shape[1] // 4)
        try:
            mask = cv2.circle(np.zeros(img.shape).astype(np.uint8).copy(), (img.shape[0] // 2, img.shape[1] // 2),
                   randint(img.shape[0] // 3, int(img.shape[0] / 1.5) ), (255,255,255), -1)
        except TypeError:
            print("c")
        img &= mask
        return img
    return random_microscope


def random_lines(p=0.5):
    return Lambda(random_lines_factory(), p=p)


def random_mm(p=0.5):
    return Lambda(random_mmm_factory(), p=p)


def random_microscope(p=0.5):
    return Lambda(random_microscope_factory(), p=p)


def random_invert_channels(p=0.5):
    return Lambda(lambda x: x[:,:,::-1], p=p)


class AdvancedHairAugmentation(ImageOnlyTransform):

    def __init__(self, hairs: int = 5, hairs_folder: str = "" , always_apply=False, p=0.5):
        self.hairs = hairs
        self.hairs_folder = hairs_folder
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        n_hairs = randint(0, self.hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, choice(hair_images)))
            if np.max(hair.shape) > img.shape[0] :
                hair = cv2.resize(hair, (0,0), fx=0.5, fy=0.5)
            hair = cv2.flip(hair, choice([-1, 0, 1]))
            hair = cv2.rotate(hair, choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = randint(0, img.shape[0] - hair.shape[0])
            roi_wo = randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg, dtype=cv2.CV_64F)
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img
