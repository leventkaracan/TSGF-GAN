# Trainable self-guided filter for multi-focus image fusion (TSGF-GAN)

![TSGF-GAN](https://github.com/leventkaracan/TSGF-GAN/assets/2334419/0999507c-323a-4a7d-89b9-eb07578ef8b3)

## Introduction

TSGF-GAN proposes a new GAN-based multi-focus Image Fusion (MFIF) model leveraging a trainable guided filter module to improve fusion quality by predicting more accurate focus maps. The self-guided adaptive filtering enhances predicted focus maps and succeeds in superior multi-focus fusion results. The proposed approach outperforms existing GAN-based MFIF methods and achieves highly competitive performance with state-of-the-art methods. 

For a comprehensive understanding and deeper insights, we invite you to explore the [paper](https://ieeexplore.ieee.org/abstract/document/10325460).


## Installation

TSGF-GAN is coded with PyTorch.

It requires the following installations:

```
python 3.8.3
pytorch (1.7.1)
cuda 11.1
```


## Training Data

Given a dataset root path in which there are folders containing input multi-focus images and corresponding all-in-focus images, you can train your own model.

We follow the [MFIF-GAN](https://github.com/ycwang-libra/MFIF-GAN) to generate training data from [Pascal VOC12](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) dataset.

## Test Datasets

You may find the test data under the datasets folder. Please refer to the related papers if you use them in your research.

### [Lytro](https://github.com/xingchenzhang/MFIFB)
```M. Nejati, S. Samavi, S. Shirani, "Multi-focus Image Fusion Using Dictionary-Based Sparse Representation", Information Fusion, vol. 25, Sept. 2015, pp. 72-84. ```

### [MFFW](https://github.com/xingchenzhang/MFIFB)
```Xu, S., Wei, X., Zhang, C., Liu, J., & Zhang, J. (2020). MFFW: A new dataset for multi-focus image fusion. arXiv preprint arXiv:2002.04780.```

### [MFI-WHU](https://github.com/HaoZhang1018/MFI-WHU)

```Zhang, H., Le, Z., Shao, Z., Xu, H., & Ma, J. (2021). MFF-GAN: An unsupervised generative adversarial network with adaptive and gradient joint constraints for multi-focus image fusion. Information Fusion, 66, 40-53.```

 
## Training TSGF-GAN

You can train MiT-MFIF using the following script. 

`python main.py --root_traindata  ./mfif_dataset/  --model_save_dir ./models/  --model_name mfif`

## Testing TSGF-GAN

You can test TSGF-GAN using the following script. You can reach the pre-trained model under the "model" directory.

`python test.py --root_testdata  ./datasets --test_dataset LytroDataset --root_result ./results  --root_model ./models/ --model_name tsgf-gan_best`

## Evaluation

To evaluate the TSGF-GAN, we utilize the following Matlab implementations.

 [https://github.com/zhengliu6699/imageFusionMetrics](https://github.com/zhengliu6699/imageFusionMetrics)
 
 [https://github.com/xytmhy/Evaluation-Metrics-for-Image-Fusion](https://github.com/xytmhy/Evaluation-Metrics-for-Image-Fusion)


## Implementation Notes

In our code, some code pieces are adapted from the [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).

## Results

We have included the results for three datasets (Lytro, MFFW, MFI-WHU) in the "results" folder.

## Contact

Feel free to reach out to [me](mailto:leventkaracan87@gmail.com) with any questions regarding MiT-MFIF or to explore collaboration opportunities in solving diverse computer vision and image processing challenges. For additional details about my research please visit [my personal webpage](https://leventkaracan.github.io/).

## Citing TSGF-GAN

```
@ARTICLE{karacan2023tsgfgan,
  author={Karacan, Levent},
  journal={IEEE Access}, 
  title={Trainable Self-Guided Filter for Multi-Focus Image Fusion}, 
  year={2023},
  volume={11},
  number={},
  pages={139466-139477},
  doi={10.1109/ACCESS.2023.3335307}}
```

