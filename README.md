# 🧬 Protien GAN

This repository attempts to implement "Generative Modeling for Protein Structures" by Namrata Anand and Po-Ssu Huang. 
  
The processed dataset is available on [Kaggle](https://www.kaggle.com/collinarnett/protein-maps)
```
@incollection{NIPS2018_7978,
title = {Generative modeling for protein structures},
author = {Anand, Namrata and Huang, Possu},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {7494--7505},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7978-generative-modeling-for-protein-structures.pdf}
```    
 
## Installation

Install Kaggle CLI by following their installation guide found [here](https://www.kaggle.com/docs/api). Then run the following command to download the dataset:

```
kaggle datasets download -d collinarnett/protein-maps
```

### Requirements

Before getting started with training Protein GAN requires:

 - Python (>= 3.6)


Run the following command to install the nessary dependencies:
```
pip install -r requirements.txt 
```

## Usage

```
usage: train.py [-h] [--config CONFIG] [--batch_size BATCH_SIZE]
                [--manual_seed MANUALSEED] [--num_epochs NUM_EPOCHS] [--lr LR]
                [--data DATA_PATH] [--image_size IMAGE_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to config json file
  --batch_size BATCH_SIZE
                        Number of maps shown to the generator and
                        discriminator per step.
  --manual_seed MANUALSEED
                        Set random seed for reproducibility. Default is 666.
  --num_epochs NUM_EPOCHS
                        Number of epochs to train the GAN for.
  --lr LR               Learning rate.
  --data DATA_PATH      Dataset location.
  --image_size IMAGE_SIZE
                        Image size.
```

### Example

The simplist way to run Protien GAN is to list all your parameters as arguments:

```
python train.py --batch_size 13 --manual_seed 666 --num_epochs 5 --lr 0.0001 --data dataset.hdf5 --image_size 64
```

However you can also specify a json config file for running experiments.

```
python train.py --config config.json
```

The json must be in the following format:

```json
{
   "batch_size": 13,
   "num_epochs": 5,
   "lr": 0.0001,
   "manualSeed": 666,
   "image_size": 64,
   "data_path": "dataset.hdf5"
}
```

### Monitoring

You can connect to the tensorboard by running the following command

```
tensorboard --logdir='runs'
```
