import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from models import SegModel

training_data_path = "/mnt/data/syn/"

model = SegModel(
    data_path=training_data_path,
    lr=5e-4,
    batch_size=64,
    val_ratio=0.25
)

# ### set finetuning parameters
# model.learning_rate = 1e-4
# model.batch_size = 64

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
    max_epochs=10000,
    #callbacks=[early_stopping],
)
trainer.logger._default_hp_metric = False
trainer.fit(model=model)
