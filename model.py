import torch
import torch.nn as nn
import pytorch_lightning as L
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler


class UNet(L.LightningModule):
    def __init__(
        self, encoder_name="resnet34", in_channels=3, classes=1, t_max=50, **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            **kwargs,
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)

        self.register_buffer("std", torch.tensor(params["std"]).view(1, 4, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 4, 1, 1))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image):
        normalized_image = (image[:, :4, :, :] - self.mean) / self.std
        return self.model(normalized_image)

    def shared_step(self, batch, stage):
        image, mask = batch

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.to(torch.int64), mask.to(torch.int64), mode="binary"
        )

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        return out["loss"]


    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(out)
        return out["loss"]
    

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)


    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()


    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.t_max, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

