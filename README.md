# Empowering Low-Light Image Enhancer through Customized Learnable Priors (ICCV2023)
[Paper link](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Empowering_Low-Light_Image_Enhancer_through_Customized_Learnable_Priors_ICCV_2023_paper.pdf), [arXiv](https://arxiv.org/pdf/2309.01958.pdf)

Naishan Zheng*, Man Zhou*, Yanmeng Dong, Xiangyu Rui, Jie Huang, Chongyi Li, Feng Zhao

*Equal Contribution

University of Science and Technology of China, Xi’an Jiaotong University, S-Lab, Nanyang Technological University   


## How to test on LOL

1. Update the paths of image sets and pre-trained models.
 ```
Updating the paths in configure files of /CUE/options/test/learnedPrior/LearnablePrior.yml
```

2. Run the testing commands.
 ```
python test.py -opt /CUE/options/test/learnedPrior/LearnablePrior.yml
```

## How to train CUE

**Some steps require replacing your local paths.**

1. Training the learnable noise prior.
```
python train.py -opt /CUE/options/train/learnedPrior/MAE_refl_hog.yml
```

2. Training the learnable illumination prior.
```
python train.py -opt /CUE/options/train/learnedPrior/UNet_illu_bil.yml
```

3. Training the enhancement network.
```
python train.py -opt /CUE/options/train/learnedPrior/LearnablePrior.yml
```
