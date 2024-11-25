# SemiSAM

Official repository of BIBM'24 paper "SemiSAM: Enhancing Semi-Supervised Medical Image Segmentation via SAM-Assisted Consistency Regularization".

Under construction. Coming soon.


## Introduction

*  Semi-supervised learning has attracted much attention due to its less dependence on acquiring abundant annotations from experts compared to fully supervised methods, which is especially important for medical image segmentation which typically requires intensive pixel/voxel-wise labeling by domain experts. Although semi-supervised methods can improve the performance by utilizing unlabeled data, there are still gaps between fully supervised methods under extremely limited annotation scenarios.

*  We propose a simple yet efficient strategy to explore the usage of the Segment Anything Model (SAM) for enhancing semi-supervised medical image segmentation. Please refer to [the paper](https://arxiv.org/pdf/2312.06316.pdf) for more details.

## :computer: Usage

1. Clone the repo
```
git clone https://github.com/YichiZhang98/SemiSAM
cd SemiSAM
```
2. Data & Checkpoint preparation.
Put the data in `data/2018LA_Seg_Training Set` and SAM checkpoint in `ckpt/sam_med3d.pth`.

5. Train
```
cd code
python train_LA_semisam_mt.py
```

4. Test
```
python test_LA.py
```


## :books: Citation

If you find this paper useful, please consider citing:
```
@article{SemiSAM,
  title={SemiSAM: Enhancing Semi-Supervised Medical Image Segmentation via SAM-Assisted Consistency Regularization},
  author={Zhang, Yichi and Yang, Jin and Liu, Yuchen and Cheng, Yuan and Qi, Yuan},
  journal={arXiv preprint arXiv:2312.06316},
  year={2023}
}
```

* This code is adapted from [UA-MT](https://github.com/yulequan/UA-MT) and [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D). We thank all the authors for their contribution. 
