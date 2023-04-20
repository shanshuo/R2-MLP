# R2-MLP: Round-Roll MLP for Multi-View 3D Object Recognition

This folder contains the PyTorch code for our paper [R2-MLP: Round-Roll MLP for Multi-View 3D Object Recognition](https://arxiv.org/abs/2211.11085) by [Shuo Chen](https://shanshuo.github.io/), [Tan Yu](https://sites.google.com/site/tanyuspersonalwebsite/home), and [Ping Li](https://pltrees.github.io/).

If you use this code for a paper, please cite:


```
@misc{Chen2022R2MLP,
  author        = {Shuo Chen and
                   Tan Yu and
                   Ping Li},
  title         = {{R2-MLP:} Round-Roll {MLP} for Multi-View 3D Object Recognition},
  year          = {2022},
  eprint        = {2211.11085},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

This work is built upon our BMVC 2021 paper [MVT: Multi-view Vision Transformer for 3D Object Recognition](https://arxiv.org/abs/2110.13083) and its corresponding [GitHub repository](https://github.com/shanshuo/MVT), which also focuses on the same task of multi-view 3D object recognition.


## Requests
PyTorch 1.7.0+


## Data preparation
Download ModelNet40 dataset (20 view setting) and extract it to the current folder:

```bash
wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar
tar -xvf modelnet40v2png_ori4.tar
```


## Training
Training on 2 V100 GPUs use the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_addr 127.0.0.5 --master_port 24500 --nproc_per_node=2 --use_env main.py --model resmlp_36_224 --epochs 1000 --batch-size 8 --lr 0.003 --data-set M10v2o4 --view-num 20 --output_dir output --no-repeated-aug --amp --native-amp
```

**Note:** If you change `--view-num`, please remember to change `timm/models/mlp_mixer.py` **line 158** and **line 255** accordingly.


## Evaluation
[Download](https://surfdrive.surf.nl/files/index.php/s/IcEre2sVE35j3jd) the R2-MLP-36 model trained on ModelNet10 with the 20 view setting.
Run the following command to evaluate the model:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model=resmlp_36_224 --resume=r2mlp_m10_20view.pth --data-set=M10v2o4 --num_workers=4 --view-num=20 --batch-size=8
```


## Acknowledgments
This repo is based on [MVT](https://github.com/shanshuo/MVT), [Deit](https://github.com/facebookresearch/deit) and [SOS](https://github.com/ntuyt/SOS). We thank the authors for their work.
