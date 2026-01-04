
import torch
import pytorch_lightning as L
import segmentation_models_pytorch as smp

class UNet(L.LightningModule):
    def __init__(self, arch="Unet", encoder="resnet34", lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.create_model(arch, encoder_name=encoder, in_channels=3, classes=1)
        
        self.loss_fn = smp.losses.DiceLoss(mode='binary')
        self.metrics = smp.utils.metrics.BinaryDiceCoefficient()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = (torch.sigmoid(logits) > 0.5).float()
        dice = self.metrics(preds, y)

        self.log_dict({
            'val_loss': loss,
            'val_dice': dice,
        }, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

