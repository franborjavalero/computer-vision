# VGG16 + mixup on CIFAR10

In this assigment, we achieve more than 90% of accuracy on the CIFAR10 test set using VGG10 model applying the data augmentation technique mixup on the training set.

## Usage
The script below replicates the experiment:
```
./vgg.sh
```

## Results
The accuracy and cost plots generated during the training are in [this folder](https://github.com/franborjavalero/computer-vision/tree/master/basic-implementations/plots).

## References
- [Paper](https://arxiv.org/pdf/1409.1556.pdf)
- [PyTorch code](https://github.com/facebookresearch/mixup-cifar10/blob/master/models/vgg.py)