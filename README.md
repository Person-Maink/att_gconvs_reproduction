## Reproduction of Attentive Group Equivariant Convolutional Networks

This repository contains the a reproduction of paper:
 
 [Attentive Group Equivariant Convolutional Networks](https://arxiv.org/abs/2002.03830) <br/>**David W. Romero, Erik J. Bekkers, Jakub M. Tomczak & Mark Hoogendoorn**, ICML 2020. 

Further implementation details, reproduction philosophies, and reproduction results can be found at our [blog post](https://hackmd.io/hVXYdGK8S3-6VlYP6v2daw). 

## Folder structure
The folder structure is as follows:

* `attgconv` contains the main PyTorch library. 

* `demo` includes some short jupyter notebook demo's on how to use the the code.

* `experiments` contains the experiments described in the paper.

* `experiments\experiment_config` contains the experiemental configuration files for Hydra, which can be run.

* `saved` contains the output model files, model training value, and graphs of the training process. 

## Dependencies

This code as based on PyTorch and has been tested with the following library versions:

* torch==1.4.0

* numpy==1.17.4

* scipy==1.3.2

* matplotlib==3.1.1

* jupyter==1.0.0 

* hydra-core == 1.3.2

The exact specification of our environment is provided in the file `environment.yml`. An appropriate environment can be easily created via:
```
conda env create -f environment.yml
```
or constructed manually with conda via:
```
conda create --yes --name torch
conda activate torch
# Please check your cudatoolkit version and replace it in the following line
conda install conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install numpy==1.17.4 scipy==1.3.2 matplotlib==3.1.1 jupyter==1.0.0 --yes
```

## Experiments
For the sake of reproducibility, we provide the parameters used in the corresponding baselines hardcoded by default. If you wish to vary these parameters
for your own experiments, please modify the corresponding `experiments\experimental_config\*.yaml` file, and write the correct values in the `main.py` file when running.

Upon running `main.py` with the correct input arguments, the results of the entire table from the paper (Tables 1, 2, and 3) will be reproduced in a serial fashion, and all data saved into `saved`. 

The parameters of the models, the setup of the experiments, the datasets, can all be changed through hydra, and the yaml config files. 


### Datasets
The utilized datasets have been uploaded to a repository for reproducibility. Please extract the files in the corresponding `experiments/experiment_i/data` folder.
For our experiments in **CIFAR-10**, we make use of the dataset provided in `torchvision`.

**Rot-MNIST:** 

The dataset can be downloaded from: https://drive.google.com/file/d/1PcPdBOyImivBz3IMYopIizGvJOnfgXGD/view?usp=sharing

**PCAM**: 

We use an `ImageFolder` structure for our experiments. A file containing the entire dataset in this format can be downloaded from: https://drive.google.com/file/d/1THSEUCO3zg74NKf_eb3ysKiiq2182iMH/view?usp=sharing

Code used to transform the `.h5` dataset to this format is provided in `experiments/pcam/data/`.

## License

The code and scripts in this repository are distributed under MIT license. See LICENSE file.
