# Bilinear

In this assigment, we use a bilinear model based on Convolutional Neural Network that uses VGG-16 model pre-trained on ImageNet. This approach achieve in the first phase (pre-trained weights frozen) a 69.10 \% accuracy and 75.50 % accuracy on the second phase (all weights unfrozen).

## Usage
The script below replicates the experiment:
```
./biliner-cnn.sh
```

## Results
The accuracy and cost plots generated during the training are in [this folder](https://github.com/franborjavalero/computer-vision/tree/master/bilinear/plots).

## References
- [Paper](https://pdfs.semanticscholar.org/3a30/7b7e2e742dd71b6d1ca7fde7454f9ebd2811.pdf)