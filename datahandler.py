import os
import random
import numpy as np
from typing import Union, List
from pathlib import Path
from tifffile import imread
from glob import glob

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

class Simple2D_Dataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path] = None,
        file_basename: List = [],
        patch_size: tuple = (640, 640),
        inference: bool = False,
    ):
        """ data_path assumes _IM.tiff and _GT.tiff """

        self.img_list = []
        self.inference = inference
        if inference:
            self.name_list = []
            filenames = glob(data_path + "/*.tiff")
            for raw_fn in filenames: 
                # read image
                im = np.squeeze(imread(raw_fn).astype(np.float32))
                im = (im - im.mean()) / im.std()
                self.img_list.append(
                    torch.from_numpy(
                        np.expand_dims(im, axis=0)
                    )
                )
                self.name_list.append(os.path.basename(raw_fn)[:-5])
        else:
            self.patch_size = patch_size
            self.gt_list = []

            # pre-load all images
            for fn in file_basename:
                # parse filename
                raw_fn = Path(data_path) / f"{fn}_IM.tiff"
                gt_fn = Path(data_path) / f"{fn}_GT.tiff"

                # read image
                im = np.squeeze(imread(raw_fn).astype(np.float32))
                im = (im - im.mean()) / im.std()
                self.img_list.append(
                    torch.from_numpy(np.expand_dims(im, axis=0))
                )
                gt = np.squeeze(imread(gt_fn).astype(np.uint8))
                gt[gt > 0] = 1
                self.gt_list.append(
                    torch.from_numpy(np.expand_dims(gt,axis=0))
                )

    def data_transform(self, img, gt):
        # rotation
        if random.random() > 0.5:
            deg = random.randint(1, 180)
            img = TF.rotate(img, angle=deg, interpolation=InterpolationMode.BILINEAR)
            gt = TF.rotate(gt, angle=deg, interpolation=InterpolationMode.NEAREST)

        # flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            gt = TF.hflip(gt)

        # crop
        dy = self.patch_size[0]
        dx = self.patch_size[1]
        py = random.randint(0, img.shape[-2]-dy)
        px = random.randint(0, img.shape[-1]-dx)   
        img = TF.crop(img, top=py, left=px, height=dy, width=dx)
        gt = TF.crop(gt, top=py, left=px, height=dy, width=dx)

        return img, gt

    def prepare_test_image(self, img):
        # this is a placeholder, not processing is needed at the moment
        return img

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        if self.inference:
            img = self.prepare_test_image(self.img_list[idx])
            fn = self.name_list[idx]
            return img, fn
        else:
            gt = self.gt_list[idx]
            img, gt = self.data_transform(img, gt)

            return img, gt



"""
class Simple2D_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()

        self.data_dir = data_dir
    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        pass
    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
    def train_dataloader(self):
        train_split = Dataset(...)
        return DataLoader(train_split)
    def val_dataloader(self):
        val_split = Dataset(...)
        return DataLoader(val_split)
    def test_dataloader(self):
        test_split = Dataset(...)
        return DataLoader(test_split)
    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
"""

