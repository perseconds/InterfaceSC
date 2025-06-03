# InterfaceSC
Pytorch Implementation of "Learning-Based Interface for Semantic Communication with Bit Importance Awareness"
# Installation
Python 3.8 and Pytorch 2.4.1<br>
CUDA 12.1
# Usage
## Train Source Mapper/Demapper
``` 
python train_source.py --trainset {CIFAR10/SVHN/ImageNet32}
```
## Train Channel Mapper/Demapper
``` 
python train_channel.py --trainset {CIFAR10/SVHN/ImageNet32} --channel {awgn/rayleigh} --C {output dimension}
```
e.g. cbr = 1/24, channel = AWGN, trainset = CIFAR10
``` 
python train_channel.py --trainset CIFAR10 --channel awgn --C 4
```
## Evaluation
``` 
python train_channel.py --trainset {CIFAR10/SVHN/ImageNet32} --channel {awgn/rayleigh} --C {output dimension}
```
# Acknowledgement
The model is partially built upon the [WITT](https://github.com/KeYang8/WITT) and [CompressAI](https://github.com/InterDigitalInc/CompressAI/). We thank the authors for sharing their code.
