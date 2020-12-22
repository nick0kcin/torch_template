import cv2
from albumentations import ImageOnlyTransform
from albumentations import Solarize, ToGray


class OTSUSolarize(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.5):
        self. solarize = Solarize(p=1, always_apply=True)
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        thr, _ = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        self.solarize = Solarize(thr, p=1, always_apply=True)
        img = ToGray(p=1)(image=img, force_apply=True)["image"]
        img = self.solarize(image=img, force_apply=True)["image"]
        img = cv2.medianBlur(img, 3)
        return img


class GraySolarize(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.5):
        self. solarize = Solarize(p=1, always_apply=True)
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        img = ToGray(p=1)(image=img, force_apply=True)["image"]
        img = self.solarize(image=img, force_apply=True)["image"]
        img = cv2.medianBlur(img, 3)
        return img
