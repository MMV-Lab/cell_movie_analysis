import os
import numpy as np
import random
from glob import glob
import torch
import monai
import pytorch_lightning as pl
from datahandler import Simple2D_Dataset
from torch.utils.data import DataLoader


class SegModel(pl.LightningModule):

    def __init__(
        self,
        data_path: str = None,
        batch_size: int = 4,
        lr: float = 1e-3,
        val_ratio: float = 0.2,
        train: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.net = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=1,
            out_channels=2,
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
            #num_res_units=2
        )

        if train:
            self.data_path = data_path
            self.batch_size = batch_size
            self.learning_rate = lr

            self.criterion = monai.losses.DiceCELoss(softmax=True, to_onehot_y=True)
            # torch.nn.CrossEntropyLoss() 

            # split train/test
            all_files = glob(data_path + "/*_IM.tiff")
            all_basenames = [os.path.basename(fn)[:-8] for fn in all_files]

            num_train = int(np.floor((1-val_ratio) * len(all_files)))
            random.shuffle(all_basenames)
            self.train_basename = all_basenames[:num_train]
            self.valid_basename = all_basenames[num_train:]

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        out = self(img)
        loss = self.criterion(out, mask)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        out = self(img)
        loss = self.criterion(out, mask)
        self.log('val_loss', loss)
        return loss

    """
    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val}
        return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}
    """

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self):
        return DataLoader(
            Simple2D_Dataset(self.data_path, self.train_basename),
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            Simple2D_Dataset(self.data_path, self.train_basename),
            batch_size=self.batch_size,
            shuffle=False
        )