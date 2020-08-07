# Advanced topologies

In this assigment, we experiment with two deep neural models based on Convolutional Neural Networks.

## Wide ResNet on CIFAR10

In this assigment, we achieve 92.50% of accuracy on the CIFAR10 test set using the Wide ResNet depth=20 and k=10.

### Usage
The script below replicates the experiment:
```
./wide-restnets.sh
```

### Results
The accuracy and cost plots generated during the training are in [this folder](https://github.com/franborjavalero/computer-vision/tree/master/advanced-topologies/plots/wide_resnet).

### Reference
- [Paper](https://arxiv.org/pdf/1605.07146.pdf)

## DenseNet on CIFAR10

In this assigment, we achieve 93.70% of accuracy on the CIFAR10 test set using the Wide DenseNet-121.

### Usage
The script below replicates the experiment:
```
./dense-nets.sh
```

### Results
The accuracy and cost plots generated during the training are in [this folder](https://github.com/franborjavalero/computer-vision/tree/master/advanced-topologies/plots/densenet).

### Reference
- [Paper](https://arxiv.org/pdf/1608.06993.pdf)