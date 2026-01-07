import torch
import matplotlib.pyplot as plt
import pytorch_lightning as L
from torch.utils.data import DataLoader
from dataset import SpunetDataset
from model import UNet

CHECKPOINT = "lightning_logs/version_2/checkpoints/epoch=199-step=200.ckpt"
DATA = "./img/train/train_img"
MASK = "./img/train/train_mask"


def predict_and_visualize():
    model = UNet.load_from_checkpoint(CHECKPOINT)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = SpunetDataset(DATA, MASK)
    test_dataloader = DataLoader(test_dataset, batch_size=7, shuffle=False)

    batch = next(iter(test_dataloader))
    images, masks = batch

    with torch.no_grad():
        logits = model(images.to(device))
        pr_masks = (logits.sigmoid() > 0.5).float()
        # pr_masks = logits.sigmoid()

    for idx in range(min(len(images), 7)):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        img_display = images[idx].permute(1, 2, 0).cpu().numpy()
        img_display = (img_display - img_display.min()) / (
            img_display.max() - img_display.min() + 1e-8
        )
        plt.imshow(img_display)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(masks[idx].squeeze().cpu().numpy(), cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        pred_display = pr_masks[idx].squeeze().cpu().numpy()
        plt.imshow(pred_display, cmap="gray")
        plt.title(f"Prediction\n(Max Prob: {pred_display.max():.2f})")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    predict_and_visualize()
