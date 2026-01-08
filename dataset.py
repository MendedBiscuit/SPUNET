import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class SpunetDataset(Dataset):
    """
    Prepare training images and masks for loading

    -- Parameters --
    img_dir : str
    mask_dir : str
    transform : Albumentations sequential 

    """
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

        # Ensure mask is read as 2 channels
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Ensure inputs are binary
        mask = mask.astype(np.float32) / 255.0
        
        if self.transform:
            # Applies albumentations sequential to mask AND image
            # Includes normalisation
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Only perform normalisation
            norm = A.Compose([
                    A.Resize(544, 544),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                    ])
            augmented = norm(image = image, mask = mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

