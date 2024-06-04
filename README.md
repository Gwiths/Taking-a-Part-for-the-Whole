# Paper Title : Taking a Part for the Whole: An Archetype-agnostic Framework for Voice-Face Association

## Overview
The paper presents an Archetype-agnostic framework for enhancing voice-face association accuracy. It introduces the AaSM method for feature calibration independent of modal archetypes and a Bilateral Connection Re-gauging scheme to refine data pair calibration. The framework is shown to improve data utilization and achieve competitive performance in cross-modal cognitive tasks.

For further reading, visit the full paper [here](https://dl.acm.org/doi/abs/10.1145/3581783.3611938).



## Environment

- Ubutu 20.04
- Python 3.8.12
- Pytorch 1.4.0 
- CUDA 10.1

## Requirements

```txt
easydict==1.9
librosa==0.8.0
lmdb==0.94
matplotlib==3.5.0
numpy==1.20.3
opencv_python==4.5.4.60
PyYAML==6.0
Pillow==8.4.0
scipy==1.7.3
scikit_learn==1.0.1
tensorboardX==2.1
torch==1.4.0
torchaudio==0.4.0
torchvision==0.5.0
tqdm==4.62.3
```



## Preliminary

1. Get dataset

   - Download the [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html), [VGGFace](https://www.dropbox.com/s/bqsimq20jcjz1z9/VGG_ALL_FRONTAL.zip?dl=0) datasets and unzip them to the path specified in the `image_data_dir/audio_data_dir` section of the `cfg.yaml` file.

2. Get our trained model

   - Download the trained [model](https://www.dropbox.com/s/kllvfxyoq0bjfcb/checkpoint_best.pth.tar?dl=0). You can configure the model path in `testing.trained_model_path` of the `cfg.yaml` file

3.  Get test list

      - The test list contains three evaluation scenarios : matching, verification and retrieval.

      - Download [test list](https://www.dropbox.com/s/ht5g2hjzjs2q0hb/gen_list.zip?dl=0) or generate your own test list. You can also customize the path to the test list by modifying the `list_dir` key in the configuration file.

## Training

Run the setup script:
```shell
python3 train.py --cfg config/cfg.yaml
```

## Testing

```shell
python3 test.py --cfg config/cfg.yaml
```

## Expected results(%):
|             |     | 1:2 Matching(U) | 1:2Matching(G) | Verification(U) | Verification(G) | Retrieval |
|-------------|-----|-----------------|----------------|-----------------|-----------------|-----------|
| Vox-CM      | V2F | 87.87           | 78.90           | 88.42           | 79.17           | 6.77      |
|             | F2V | 88.63           | 78.58          | 89.02           | 78.95           | 7.16      |
| AVSpeech-CM | V2F | 80.29           | 68.78          | 80.69           | 68.63           | 11.80      |
|             | F2V | 79.98           | 67.81          | 81.16           | 69.05           | 13.43     |

## Citation

If you find our work useful for your research, please consider citing it as follows:
```latex
@inproceedings{10.1145/3581783.3611938,
author = {Chen, Guancheng and Liu, Xin and Xu, Xing and Cheung, Yiu-ming and Li, Taihao},
title = {Taking a Part for the Whole: An Archetype-agnostic Framework for Voice-Face Association},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3611938},
doi = {10.1145/3581783.3611938},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {7056–7064},
numpages = {9},
keywords = {archetype-agnostic, instance equilibrium, re-gauging, voice-face association},
location = {, Ottawa ON, Canada, },
series = {MM '23}
}
```


