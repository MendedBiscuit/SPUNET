import torch
import matplotlib.pyplot as plt
import pytorch_lightning as L
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset import SpunetDataset
from model import UNet

CHECKPOINT = "lightning_logs/version_4/checkpoints/model.ckpt"
DATA = "./img/train/train_img"
MASK = "./img/train/train_mask"
OUTPUT = "./output/"
NUM_PREDS = 15
PREDICT_TRANSFORM = A.Compose([
    A.Resize(544, 544),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def predict_and_visualize(visualise=False, save_img=False):
    model = UNet.load_from_checkpoint(CHECKPOINT)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = SpunetDataset(DATA, MASK, transform=PREDICT_TRANSFORM)
    test_dataloader = DataLoader(test_dataset, batch_size=NUM_PREDS, shuffle=False)

    batch = next(iter(test_dataloader))
    images, masks = batch

    with torch.no_grad():
        logits = model(images.to(device))
        pr_masks = (logits.sigmoid() > 0.5).float()
        # pr_masks = logits.sigmoid()

    if visualise:
        for idx in range(min(len(images), NUM_PREDS)):
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
    
    if save_img:
        px = 1/plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(544*px, 544*px), frameon=False)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        pred_display = pr_masks[idx].squeeze().cpu().numpy()
        ax.imshow(pred_display, cmap="gray", aspect='auto')

        plt.savefig(
            OUTPUT + f"pred{idx+1}.png", 
            format="png", 
            bbox_inches=None,
            pad_inches=0,
            dpi=plt.rcParams['figure.dpi']
        )
        plt.close(fig)


if __name__ == "__main__":
    predict_and_visualize(visualise=True, save_img=False)
