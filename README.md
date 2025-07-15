# SemiSAM  [[paper](https://arxiv.org/pdf/2312.06316.pdf)] / SemiSAM+  [[paper](https://arxiv.org/pdf/2502.20749)] 

Official repository of our paper "SemiSAM: Enhancing Semi-Supervised Medical Image Segmentation via SAM-Assisted Consistency Regularization" (BIBM'24) and "SemiSAM+: Rethinking Semi-Supervised Medical Image Segmentation in the Era of Foundation Models" (MedIA'25).


![image](https://github.com/YichiZhang98/SemiSAM/blob/main/Highlight.png)

## Introduction

*  Semi-supervised learning (SSL) has attracted much attention due to its less dependence on acquiring abundant annotations from experts compared to fully supervised methods, which is especially important for medical image segmentation which typically requires intensive pixel/voxel-wise labeling by domain experts. Although semi-supervised methods can improve the performance by utilizing unlabeled data, there are still gaps between fully supervised methods under extremely limited annotation scenarios.

*  Other than existing model-centric advancements of SSL aims to exploit more efficient utilization of unlabeled data for better performance, SemiSAM/SemiSAM+ represent a new paradigm to exploit pre-trained knowledge of SAM-like foundation models to assist in the learning procedure.

*  In SemiSAM, we propose a simple yet efficient strategy to explore the usage of the Segment Anything Model (SAM) for enhancing semi-supervised medical image segmentation. As an extension, SemiSAM+ incorporates substantial methodological advancements with extensive experiments to show strong efficiency as a plug-and-play method applicable to different specialist SSL and generalist foundation models.





## :computer: Usage

### Requirements

Our implementation is based on PyTorch 1.14.0, CUDA 11.8 and Python 3.8. Some important required packages include TensorBoardX, torchio, prefetch_generator, SimpleITK, etc. All experiments in our paper were conducted on an NVIDIA A100 GPU. As the generalist foundation model is not trainable, the computation cost is similar to specialist SSL model.


### Pre-trained Weights of Generalist Models

The current version supports three generalist models for training: SAM-Med3D and SAM-Med3D-turbo as universal medical generalist models, and SegAnyPET as PET modality-specific generalist model. The pre-trained weights can be accessed from [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) and [SegAnyPET](https://github.com/YichiZhang98/SegAnyPET). Put the checkpoint in `ckpt/[model].pth` for usage in the training procedure.


### Data Preparation

The processed dataset used in this work can be accessed refer to [LA](https://github.com/yulequan/UA-MT/tree/master/data) for SemiSAM and [BraTS19](https://github.com/HiLab-git/SSL4MIS/tree/master/data/BraTS2019) for SemiSAM+. Put the data in `data/[name]` for usage. The `[code/code_semisam+]/dataloaders` folders contain the code for image pre-processing, which can be used for developing on your own dataset.



### Training and Testing


[SemiSAM] The implementation of SSL specialist model is adapted from [UA-MT](https://github.com/yulequan/UA-MT).

```
# Training
cd code
python train_LA_semisam_mt.py
# Testing
python test_LA.py
```


[SemiSAM+] The implementation of SSL specialist model is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS/tree/master/data/BraTS2019).


```
# Training
cd code_semisam+
python SemiSAM_train_MT_3D.py --prompt [unc/point/mask]
python SemiSAM_train_UAMT_3D.py --prompt [unc/point/mask]
python SemiSAM_train_DAN_3D.py --prompt [unc/point/mask]
# Testing
python test_LA.py
```






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
  journal={Medical Image Analysis},
  year={2025}
}

```

* Our code is adapted from [UA-MT](https://github.com/yulequan/UA-MT), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS/tree/master/data/BraTS2019) and [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D). We thank all the authors for their contribution. 
