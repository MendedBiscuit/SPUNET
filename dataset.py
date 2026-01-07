import os
import cv2
import torch
import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms


class SpunetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(
            self.mask_dir, self.images[idx]
        )

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = mask.astype(np.float32) / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

