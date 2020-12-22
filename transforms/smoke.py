import os
import cv2
import numpy as np
from albumentations import ImageOnlyTransform
from numpy.random import randint, choice


class AdvancedSmokeAugmentation(ImageOnlyTransform):

    def __init__(self, min_width, max_width, min_height, max_height,
                 smokes=5, smoke_folder: str = "", always_apply=False, p=0.5):
        self.smokes = smokes
        self.smoke_folder = smoke_folder
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        n_smokes = randint(0, self.smokes)

        if not n_smokes:
            return img

        height, width, _ = img.shape  # target image width and height
        smoke_images = [im for im in os.listdir(self.smoke_folder) if ('png' in im) or ("jpg" in im)]

        for _ in range(n_smokes):
            smoke = cv2.imread(os.path.join(self.smoke_folder, choice(smoke_images)))
            smoke_width = randint(self.min_width, self.max_width)
            smoke_height = randint(self.min_height, self.max_height)
            smoke = cv2.resize(smoke, (smoke_width, smoke_height))
            smoke = cv2.flip(smoke, choice([-1, 0, 1]))
            smoke = cv2.rotate(smoke, choice([0, 1, 2]))

            h_height, h_width, _ = smoke.shape  # hair image width and height
            roi_ho = randint(0, img.shape[0] - smoke.shape[0])
            roi_wo = randint(0, img.shape[1] - smoke.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]
            x, y = np.meshgrid(range(h_width), range(h_height))
            w = (x - h_width / 2) ** 2 + (y - h_height / 2) ** 2
            w = np.log(1 + w[:, :, None])
            w /= w.max()

            img2gray = cv2.cvtColor(smoke, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            smoke_fg = cv2.bitwise_and(smoke, smoke, mask=mask)

            dst = cv2.add(((1 - w) * img_bg).astype(np.uint8), (w * smoke_fg).astype(np.uint8), dtype=cv2.CV_64F)
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img
