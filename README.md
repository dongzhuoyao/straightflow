
# StraightFlow

RectifiedFlow on Cifar10
```
CUDA_VISIBLE_DEVICES=0  python train_vanilla.py --config=configs/rflow_cifar10.py 
```

Flow Matching on Cifar10

```
CUDA_VISIBLE_DEVICES=0  python train_vanilla.py 
```

```
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=9233 --use_env  train_vanilla.py  --config=configs/fm_cifar10.py 
```




## Prepare ./assets following U-ViT repo

[https://github.com/baofff/U-ViT](https://github.com/baofff/U-ViT)

fid_stats, a dummy file
pretrained_weights, for initialization and fine-tunig
stable-diffusion, need the Encoder-Decoder weight

## Environment Preparation

```
python 3.10
torch2.0
```


```
conda create -n straightflow  python=3.10
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning torchdiffeq  matplotlib h5py timm diffusers accelerate loguru blobfile ml_collections
pip install hydra-core wandb einops scikit-learn --upgrade
pip install einops sklearn ml_collections
pip install transformers==4.23.1 pycocotools # for text-to-image task

```







# Acknowledgement

This codebase is developed based on U-ViT, if you find this repo useful, please consider citing the following paper:

```
@inproceedings{bao2022all,
  title={All are Worth Words: A ViT Backbone for Diffusion Models},
  author={Bao, Fan and Nie, Shen and Xue, Kaiwen and Cao, Yue and Li, Chongxuan and Su, Hang and Zhu, Jun},
  booktitle = {CVPR},
  year={2023}
}
```
