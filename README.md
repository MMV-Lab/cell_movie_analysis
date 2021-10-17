# cell_movie_analysis

## instruction for using this repository to do cell segmentation and tracking

## instruction for using this repository to test training deep learning models on single/multiple GPUs on a local machine or a cluster

### step 1: installaiton 

* create a new conda environment (python=3.8)
* activate the new conda environment
* install PyTorch (this is for LTS version 1.8.2 on CUDA 11.1 on Linux OS, check [official website](https://pytorch.org/get-started/locally/) for details if you need to install in a different setting):

```bash
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

* install packages using **requirement.txx**

```bash
pip install -r requirement.txt
```

### step 2: genearte synthetic training data for testing

```bash
python generate_syn.py
```

make sure your change `out_path` to the directory you want to save your data

### step 3: run model training with different configurations

```bash
python train.py
```

Make sure to change the `training_data_path` to the directory your saved your synthetic data.

To try out different training settings (e.g., single gpu or multi-gpu, etc.), you can change the parameters in `trainer = pl.Trainer()` (near the bottom of the script) following the documentation here: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.html?highlight=trainer

More details about how to do training on computing cluster (e.g. SLURM) can be found here: https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html


## instruction for using this repository to test deploying a deep learning model as a tool for data processing on single/multiple GPUs on a local machine or a cluster


### step 1: installaiton 

* create a new conda environment (python=3.8)
* activate the new conda environment
* install PyTorch (this is for LTS version 1.8.2 on CUDA 11.1 on Linux OS, check [official website](https://pytorch.org/get-started/locally/) for details if you need to install in a different setting):

```bash
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

* install packages using **requirement.txx**

```bash
pip install -r requirement.txt
```

### step 2: genearte synthetic images for testing

```bash
python generate_syn_for_test.py
```

make sure your change `out_path` to the directory you want to save your data

### step 3: Apply a trained model on these new synthetic images

```bash
python test_on_syn.py
```

Make sure to check the `data_path` to the directory you saved your test data and change `out_path` to the path your want to save the results

This script is very simple. You may need to take this and change accordingly for different applications, such as ImJoy.

There are two more advanced ways to run a Pytorch model in production, see 
https://pytorch-lightning.readthedocs.io/en/latest/common/production_inference.html