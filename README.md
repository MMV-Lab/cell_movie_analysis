# cell_movie_analysis

## instruction for using this repository to do cell segmentation and tracking

## instruction for using this repository to test deep learning applications on single/multiple GPUs on a local machine or a cluster

### step 1: installaiton 

* create a new conda environment
* activate the new conda environment and install packages using **requirement.txx**

"""bash
pip install -r requirement.txt
"""

### step 2: genearte synthetic training data for testing

"""bash
python generate_syn.py
"""

make sure your change `out_path` to the directory you want to save your data

### step 3: run model training with different configurations

"""bash
python train.py
"""

Make sure to change the `training_data_path` to the directory your saved your synthetic data.

To try out different training settings (e.g., single gpu or multi-gpu, etc.), you can change the parameters in `trainer = pl.Trainer()` (near the bottom of the script) following the documentation here: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.html?highlight=trainer

More details about how to do training on computing cluster (e.g. SLURM) can be found here: https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html

