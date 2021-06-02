
# PU-Flow: a Point Cloud Upsampling Network with Normalizing Flows

Official PyTorch implementation for paper: https://xxx.xx

## Environment

First clone the code of this repo:
```bash
git clone --recursive https://github.com/unknownue/PU-Flow
```
Then other settings can be either configured manually or set up with docker.

### Manual configuration

The code is implemented with CUDA 11.1, Python 3.8, PyTorch 1.8.0.
Other require libraries:

- pytorch-lightning==1.2.2
- tensorflow==1.14.0 for dataset loading
- knn_cuda from https://github.com/unlimblue/KNN_CUDA
- pointnet2lib from https://github.com/erikwijmans/Pointnet2_PyTorch.git

### Docker configuration

If you are familiar with Docker, you can use provided [Dockerfile](docker/Dockerfile) to configure all setting automatically.

### Additional configuration

If you want to train the network, you also need to build the kernel of PytorchEMD like followings:
```bash
cd metric/PytorchEMD/
python setup.py install --user
cp build/lib.linux-x86_64-3.6/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```

## Datasets
All training and evaluation data can be downloaded from this [link](https://drive.google.com/drive/folders/1jaKC-bF0yfwpdxfRtuhoQLMhCjiMVPiz?usp=sharing), including:
- Training data from PUGeo dataset (tfrecord_x4_normal.zip). Put training data as list in [here](data/filelist.txt).
- Testing models of input 5K points and corresponding ground truth 20K points.
- Training and testing meshes for further evaluation.

We include a [pretrained x4 model](pretrain/puflow-x4-final.pt) in this repo.

## Training & Evaluation
Train the model as followings:
```bash
python train_interpflow.py
```

Upsampling point clouds as followings:
```bash
python evaluate_pflow.py \
    --source=path/to/input/directory \
    --target=path/to/output/directory \
    --checkpoint=pretrain/puflow-x4-final.pt \
    --up_ratio=4 \
    --num_out=20000
```