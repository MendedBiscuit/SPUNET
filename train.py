import os
import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from model import UNet
from dataset import SpunetDataset
import albumentations as A 
from albumentations.pytorch import ToTensorV2

# Training Parameters

EPOCHS = 256
BATCH_SIZE = 32

TRAIN_IMG = "./img/train/train_img"
TRAIN_MASK = "./img/train/train_mask"
VAL_IMG = "./img/val/val_img"
VAL_MASK = "./img/val/val_mask"

TRAIN_TRANSFORM = A.Compose([
                A.Resize(544, 544),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
])
VALID_TRANSFORM = A.Compose([
                A.Resize(544, 544),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
])


def rename_model_file():
    """
    Renames the model file to "model.ckpt
    """
    version = max([int(x.split("_")[-1]) for x in os.listdir("./lightning_logs/")])
    directory = f"./lightning_logs/version_{version}/checkpoints/"
    old = os.path.join(directory, f"epoch={EPOCHS-1}-step={EPOCHS*2}.ckpt")
    new = os.path.join(directory, "model.ckpt")
    os.rename(old, new)


def main():
    """
    Functionality for being able to run the training   
    """

    # Prepare training and validation data with appropriate transforms 
    train_ds = SpunetDataset(TRAIN_IMG, TRAIN_MASK, transform=TRAIN_TRANSFORM)
    val_ds = SpunetDataset(VAL_IMG, VAL_MASK, transform=VALID_TRANSFORM)

    # Load data for training, change num_workers to increase CPU/GPU load
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )

    # Configure model
    model = UNet(encoder_name="resnet34", in_channels=3, classes=1, t_max=EPOCHS)

 
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        precision="16-mixed",
    )

    # Lightning module responsible for running the training loop
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    rename_model_file()

main()
