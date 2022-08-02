# PU-Flow: a Point Cloud Upsampling Network with Normalizing Flows

by Aihua Mao, Zihui Du, Junhui Hou, Yaqi Duan, Yong-jin Liu and Ying He

## Introduction

Official PyTorch implementation of our TVCG paper: https://xxx.xx

## Environment

First clone the code of this repo:
```bash
git clone --recursive https://github.com/unknownue/PU-Flow
```
Then other settings can be either configured manually or set up with docker.

### Manual configuration

The code is implemented with CUDA 11.1, Python 3.8, PyTorch 1.8.0.

Other require libraries:

- pytorch-lightning==1.2.2 for training
- tensorflow==1.14.0 for dataset loading
- torchdiffeq form https://github.com/rtqichen/torchdiffeq
- pytorch3d from https://github.com/facebookresearch/pytorch3d
- knn_cuda from https://github.com/unlimblue/KNN_CUDA

### Docker configuration

If you are familiar with Docker, you can use provided [Dockerfile](docker/Dockerfile) to configure all setting automatically.

### Additional configuration for training

If you want to train the network, you also need to build the kernel of emd like followings:
```bash
cd metric/emd/
python setup.py install --user
```

## Datasets
All training and evaluation data can be downloaded from this [link](https://drive.google.com/drive/folders/1jaKC-bF0yfwpdxfRtuhoQLMhCjiMVPiz?usp=sharing), including:
- Training data from PUGeo dataset (tfrecord_x4_normal.zip), PU-GAN dataset and PU1K dataset. Put training data as list in [here](data/filelist.txt).
- Testing models of input 2K/5K points and corresponding ground truth 8K/20K points.
- Training and testing meshes for further evaluation.

We include some [pretrained x4 models](pretrain/) in this repo.

## Training & Upsampling & Evaluation
Train the model on specific dataset:
```bash
python modules/discrete/train_pu1k.py      # Train the discrete model on PU1K Dataset
python modules/discrete/train_pugeo.py     # Train the discrete model on PUGeo Dataset
python modules/discrete/train_pugan.py     # Train the discrete model on PU-GAN Dataset
python modules/continuous/train_interp.py  # Train the continuous model on PU1K Dataset
```

Upsampling point clouds as follows:
```bash
# For discrete model
python modules/discrete/upsample.py \
    --source=path/to/input/directory \
    --target=path/to/output/directory \
    --checkpoint=pretrain/puflow-x4-pugeo.pt \
    --up_ratio=4

# For continuous model
python modules/continuous/upsample.py \
    --source=path/to/input/directory \
    --target=path/to/output/directory \
    --checkpoint=pretrain/puflow-x4-cnf-pu1k.pt \
    --up_ratio=4
```

Evaluation as follows:
```bash
# Build files for evaluation (see build.sh for more details)
bash evaluation/build.sh

# Evaluate on PU1K dataset
cd evaluation/
cp path/to/output/directory/**.xyz ./result/
bash eval_pu1k.sh
python evaluate.py --pred ./result/ --gt=../data/PU1K/test/input_2048/gt_8192 --save_path=./result/

# Evaluate on PU-GAN dataset
cd evaluation/
cp path/to/output/directory/**.xyz ./result/
bash eval_pugan.sh
python evaluate.py --pred ./result/ --gt=../data/PU-GAN/GT --save_path=./result/
```

## Citation(TODO)

If this work is useful for your research, please consider citing:

```bibtex
@article{unknownue2022puflow,
    title={PU-Flow: a Point Cloud Upsampling Network with Normalizing Flows},
    author={Aihua Mao and Zihui Du and Junhui Hou and Yaqi Duan and Yong-jin Liu and Ying He},
    journal={IEEE Transactions on Visualization and Computer Graphics},
    volume={},
    number={},
    pages={},
    year={2022},
    doi={}
}
```

