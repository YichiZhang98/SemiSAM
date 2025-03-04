# SemiSAM  [[paper](https://arxiv.org/pdf/2312.06316.pdf)] / SemiSAM+  [[paper](https://arxiv.org/pdf/2502.20749)] 

Official repository of our paper "SemiSAM: Enhancing Semi-Supervised Medical Image Segmentation via SAM-Assisted Consistency Regularization" and "SemiSAM+: Rethinking Semi-Supervised Medical Image Segmentation in the Era of Foundation Models".


## Introduction

*  Semi-supervised learning (SSL) has attracted much attention due to its less dependence on acquiring abundant annotations from experts compared to fully supervised methods, which is especially important for medical image segmentation which typically requires intensive pixel/voxel-wise labeling by domain experts. Although semi-supervised methods can improve the performance by utilizing unlabeled data, there are still gaps between fully supervised methods under extremely limited annotation scenarios.

*  Other than existing model-centric advancements of SSL aims to exploit more efficient utilization of unlabeled data for better performance, SemiSAM/SemiSAM+ represent a new paradigm to exploit pre-trained knowledge of SAM-like foundation models to assist in the learning procedure.

*  In SemiSAM, we propose a simple yet efficient strategy to explore the usage of the Segment Anything Model (SAM) for enhancing semi-supervised medical image segmentation. As an extension, SemiSAM+ incorporates substantial methodological advancements with extensive experiments to show strong efficiency as a plug-and-play method applicable to different specialist SSL and generalist foundation models.


## :computer: Usage of SemiSAM

1. Clone the repo
```
git clone https://github.com/YichiZhang98/SemiSAM
cd SemiSAM
```
2. Put the data in `data/2018LA_Seg_Training Set` and SAM checkpoint in `ckpt/sam_med3d.pth`.

3. Train the model
```
cd code
python train_LA_semisam_mt.py
```

4. Test the model
```
python test_LA.py
```

## :computer: Usage of SemiSAM+

Coming soon.


## :books: Citation

If you find this paper useful, please consider citing:
```
@inproceedings{SemiSAM,
  title={SemiSAM: Enhancing Semi-Supervised Medical Image Segmentation via SAM-Assisted Consistency Regularization},
  author={Zhang, Yichi and Yang, Jin and Liu, Yuchen and Cheng, Yuan and Qi, Yuan},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2024}
}

@article{SemiSAM+,
  title={SemiSAM+: Rethinking Semi-Supervised Medical Image Segmentation in the Era of Foundation Models},
  author={Zhang, Yichi and Lv, Bohao and Xue, Le and Zhang, Wenbo and Liu, Yuchen and Fu, Yu and Cheng, Yuan and Qi, Yuan},
  journal={arXiv preprint arXiv:2502.20749},
  year={2025}
}

```

* Our code is adapted from [UA-MT](https://github.com/yulequan/UA-MT) and [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D). We thank all the authors for their contribution. 
