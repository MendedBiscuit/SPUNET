import os
import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from model import UNet
from dataset import SpunetDataset
import albumentations as A 
from albumentations.pytorch import ToTensorV2

EPOCHS = 50
BATCH_SIZE = 8

TRAIN_IMG = "./img/train/train_img"
TRAIN_MASK = "./img/train/train_mask"
VAL_IMG = "./img/val/val_img"
VAL_MASK = "./img/val/val_mask"

TRAIN_TRANSFORM = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
])

def main():
    train_ds = SpunetDataset(TRAIN_IMG, TRAIN_MASK, TRAIN_TRANSFORM)
    val_ds = SpunetDataset(VAL_IMG, VAL_MASK, TRAIN_TRANSFORM)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )

    model = UNet(encoder_name="resnet34", in_channels=3, classes=1, t_max=EPOCHS)

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        precision="16-mixed",
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

main()
