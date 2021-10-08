import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datahandler import Simple2D_Dataset
from tifffile import imsave
from models import SegModel

data_path = "/mnt/data/well_C1/"
out_path = "/mnt/data/well_C1_seg/"
ckpt_path = "/home/nodeadmin/research/tcell/lightning_logs/version_3/checkpoints/epoch=99999-step=99999.ckpt"
hparams_path = "/home/nodeadmin/research/tcell/lightning_logs/version_3/hparams.yaml"

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
net.out_path = out_path

trainer = pl.Trainer(
    gpus=1,
    precision=16,
    distributed_backend='ddp',
)

test_dataloader = DataLoader(Simple2D_Dataset(data_path, inference=True))

trainer.test(model=net, dataloaders=test_dataloader)
