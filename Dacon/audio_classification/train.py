import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, log_loss, f1_score

from model import audio_resnet34

class build_fn(pl.LightningModule):
    def __init__(self, hparams, train_loader=None, val_loader=None):
        super().__init__()
        self.hparams = hparams
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = audio_resnet34()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        x, labels = batch
        output = self(x)
        loss = self.loss_fn(output, labels)

        logits = nn.functional.softmax(output, dim=-1)

        y_true = list(labels.detach().cpu().numpy())
        y_pred = list(logits.detach().cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        loss = 0.0
        y_true = []
        y_pred = []

        for i in outputs:
            loss += i['loss'].item()
            y_true += i['y_true']
            y_pred += i['y_pred']

        loss = loss / len(outputs)

        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', accuracy_score(y_true, np.argmax(y_pred,
                                                                axis=-1)), on_epoch=True, prog_bar=True, logger=True)
        self.log(state+'_f1', f1_score(y_true, np.argmax(y_pred, axis=-1),
                                       average='weighted'), on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def train_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                             patience=2,
                                                                             mode='min', verbose=True),
                     'interval': 'epoch',
                     'monitor': 'val_loss'}
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
