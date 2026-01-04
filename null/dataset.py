import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".jpg", "_mask.gif")
        )
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"].unsqueeze(0)

        return image, mask


class DataModule(pl.LightningDataModule):
    def __init__(self, image_dir, mask_dir, transform, train_size=0.9, batch_size=16):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.train_size = train_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = Dataset(self.image_dir, self.mask_dir, self.transform)
        training_size = math.floor(len(dataset) * self.train_size)
        val_size = len(dataset) - training_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [training_size, val_size]
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=16,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=16,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, self.batch_size)