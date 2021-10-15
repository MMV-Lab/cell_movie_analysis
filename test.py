import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datahandler import Simple2D_Dataset
from tifffile import imsave
from models import SegModel
from glob import glob
import os
import numpy as np
from aicsimageio import imread

timelapse_path = "/mnt/data/timelapse/"
out_parent_path = "/mnt/data/timelapse_seg/"
ckpt_path = "/home/nodeadmin/research/tcell/lightning_logs/version_4/checkpoints/epoch=99999-step=99999.ckpt"
hparams_path = "/home/nodeadmin/research/tcell/lightning_logs/version_4/hparams.yaml"

class ModelRuntime(SegModel):
    def __init__(self,**kwargs):
        super().__init__(train=False)
        self.out_path = None

    def test_step(self, batch, batch_idx):
        img, fn = batch
        assert len(fn)==1, "only batch=1 is supported currently"
        out = self(img)
        seg = out.cpu().numpy()
        seg = seg[0,1,:,:]
        imsave(self.out_path + fn[0] + "_segmentation.tiff", seg)


net = ModelRuntime.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    hparams_file=hparams_path,
    map_location=None,
)


trainer = pl.Trainer(
    gpus=1,
    precision=16,
    distributed_backend='ddp',
)

filenames = glob(timelapse_path + "/*.tiff")
for fn in filenames:
    fn_base = os.path.basename(fn)
    well_name = fn_base[:-5]

    # prepare file path
    out_path = out_parent_path + well_name + "/"
    os.makedirs(out_path, exist_ok=True)
    split_path = timelapse_path + well_name + "/"
    os.makedirs(split_path, exist_ok=True)

    # split the timelapse file
    img = np.squeeze(imread(fn))
    for tt in range(img.shape[0]):
        im_single = img[tt, :, :]
        imsave(split_path + f"/img_{tt}.tiff", im_single)

    net.out_path = out_path
    test_dataloader = DataLoader(Simple2D_Dataset(split_path, inference=True))
    trainer.test(model=net, dataloaders=test_dataloader)
