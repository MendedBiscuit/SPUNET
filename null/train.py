import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from testing.dataset import DataModule
from testing.model import UNet

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_DIR = "./data/train/"
MASK_DIR = "./data/train_masks/"


def main():
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    datamodule = DataModule(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transform=transform,
    )

    model = UNet(in_channels=3, out_channels=1)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="unet_models/",
        filename="unet_implementation-epoch-{epoch:02d}-val_loss-{val_loss:.2f}-val_acc-{val_acc:.2f}-val_f1-{val_f1:.2f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=2, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=TensorBoardLogger("unet_lightning_logs/", name="unet_implmentation"),
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()