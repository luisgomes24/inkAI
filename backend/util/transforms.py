import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class Transforms:
    def __init__(self):
        self.transforms = A.Compose(
            [
                A.SmallestMaxSize(max_size=160),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.4
                ),
                # A.RandomCrop(height=150, width=256),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(128, 128),
                ToTensorV2(),
            ]
        )

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]
