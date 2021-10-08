import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from models import SegModel


model = SegModel(
    data_path="/mnt/data/tcell_train/",
    lr=5e-4,
    batch_size=64,
    val_ratio=0.25
)

####
# ckpt_path = "/home/nodeadmin/research/tcell/lightning_logs/version_3/checkpoints/epoch=99999-step=99999.ckpt"
# hparams_path = "/home/nodeadmin/research/tcell/lightning_logs/version_3/hparams.yaml"

"""
model = SegModel.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    hparams_file=hparams_path,
    map_location="cuda:0",
)
"""
# set finetuning parameters
model.learning_rate = 1e-4
model.batch_size = 64
###

seed_everything(42, workers=True)

"""
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
)
"""

# defining training hyperparameters
trainer = pl.Trainer(
    gpus=1,
    precision=16,
    distributed_backend='ddp',
    max_epochs=100000,
    #callbacks=[early_stopping],
)
trainer.logger._default_hp_metric = False
trainer.fit(model=model)
