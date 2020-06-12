import argparse
from math import ceil, pow, floor
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from utils import export_metrics_values

"""
    Densely Connected Convolutional Networks
        - Paper: https://arxiv.org/pdf/1608.06993.pdf
        - PyTorch reference code: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
        - PyTorch reference code: https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py 
"""

class DenseLayer(nn.Module):

  """
    Bottleneck:
      Dense connectivity. To further improve the information flow between layers we propose a different connectivity pattern: 
      they introduce direct connections from any layer to all subsequent layers. 
  """
  
  def __init__(self, num_input_features, growth_rate, bn_size=4):
    super(DenseLayer, self).__init__()
    self.bn1 = nn.BatchNorm2d(num_input_features)
    self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
    self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
  
  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x))) # composite function: BN-RELU-CONV
    out = self.conv2(F.relu(self.bn2(out)))
    out = torch.cat([x, out], dim=1) # x and out have the same dimension, but different number of channels
    return out

class TransitionLayer(nn.Module):
  """
    The layers between two adjacent blocks are referred to as transition layers and change feature-map sizes via convolution and pooling
  """
  
  def __init__(self, num_input_features, num_output_features):
    super(TransitionLayer, self).__init__()
    self.bn = nn.BatchNorm2d(num_input_features)
    self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
  
  def forward(self, x):
    out = self.conv(F.relu(self.bn(x)))
    out = F.avg_pool2d(out, 2)
    return out

class DenseNet(nn.Module):
  
  def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), bn_size=4, num_classes=10, reduction=0.5):
    super(DenseNet, self).__init__()
    
    num_init_features = growth_rate * 2
    self.growth_rate = growth_rate
    self.bn_size = bn_size
    
    # First convolution
    self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False) # k=3 instead 7, reduced stride and pandding
    
    # Dense blocks and transtion layer
    num_features = num_init_features
    num_blocks_config = len(block_config) - 1
    self.dense = []
    for (idx, num_dense_layers) in enumerate(block_config):
      # Dense layer
      self.dense.append(
        self._make_dense_layers(
          num_layers=num_dense_layers, 
          num_input_features=num_features, 
        )
      )
      num_features +=  (num_dense_layers * growth_rate)
      # Transiton layer
      if idx < num_blocks_config:
        num_features_reduction = int(floor(num_features * reduction)) # apply reduction
        self.dense.append(TransitionLayer(num_input_features=num_features, num_output_features=num_features_reduction))
        num_features = num_features_reduction
    
    self.dense = nn.Sequential(*self.dense)
    self.bnt = nn.BatchNorm2d(num_features)
    self.classifier = nn.Linear(num_features, num_classes)

    # Official init from torch repo.
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    out = self.conv0(x)
    out = F.avg_pool2d(F.relu(self.bnt(self.dense(out))), 4)
    out = out.flatten(start_dim=1)
    out = self.classifier(out)
    return out  

  def _make_dense_layers(self, num_layers, num_input_features):
    layers = []
    for i in range(num_layers):
      num_input_features_ = num_input_features + (i * self.growth_rate)
      layers.append(DenseLayer(num_input_features_, self.growth_rate, bn_size=self.bn_size))
    return nn.Sequential(*layers)  


def main():
    
    """ arguments. """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./dataset/cifar10/")
    parser.add_argument("--metrics_path", type=str, default="./metrics/cifar10/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--growth_rate", type=int, default=12)
    parser.add_argument("--block_config", nargs='+', type=int, default=[6, 12, 24, 16])
    parser.add_argument("--bn_size", type=int, default=4)
    parser.add_argument("--reduction", type=float, default=0.5)
    parser.add_argument("--initial_lr", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--milestones", nargs='+', type=int, default=[150, 225])
    parser.add_argument("--max_grad_norm", type=int, default=1)
    args = parser.parse_args()

    
    """ reproducibility. """
    
    device = "cpu"
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = "cuda:0"
    
    """ prepare data. """

    dataset_path = args.dataset_path
    download_ = False
    if not os.path.isdir(dataset_path):
      os.makedirs(dataset_path)
      download_ = True
    
    train_transforms_ = torchvision.transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms_ = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(dataset_path, train=True, download=download_, transform=train_transforms_)
    test = datasets.CIFAR10(dataset_path, train=False, download=download_, transform=test_transforms_)

    batch_size = args.batch_size
    num_traning_samples = len(train)
    num_test_samples = len(test)
    num_batches_train = ceil(num_traning_samples / batch_size)
    num_batches_test = ceil(num_test_samples / batch_size)
    train_set = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_set = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    """ define network. """

    num_classes = 10
    growth_rate = args.growth_rate
    block_config = args.block_config
    bn_size = args.bn_size
    reduction = args.reduction

    model_name = f"densenet_growth_rate={growth_rate}_block_config={block_config}_bn_size={bn_size}_reduction={reduction}"

    net = DenseNet(
        growth_rate=growth_rate, 
        block_config=block_config, 
        bn_size=bn_size, 
        num_classes=num_classes, 
        reduction=reduction,
    ).to(device)
    

    """ training process. """
    
    initial_lr = args.initial_lr
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    momentum = args.momentum
    gamma = args.gamma
    milestones = args.milestones
    # max_grad_norm = args.max_grad_norm

    metrics_path = args.metrics_path
    if not os.path.isdir(metrics_path):
      os.makedirs(metrics_path)
    file_metrics = os.path.join(metrics_path, f"{model_name}.csv")

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    acc_epoch_train = np.empty(num_epochs)
    acc_epoch_test = np.empty(num_epochs)
    loss_epoch_train = np.empty(num_epochs)
    loss_epoch_test = np.empty(num_epochs)
    
    for idx_epoch in range(0, num_epochs, 1):
        
      net.train()
      num_corrected_predictions_train = 0
      total_loss_train = 0
      
      for (batch_x, batch_y) in train_set:
        batch_x_ = batch_x.view(-1, 3, 32,32).to(device)
        batch_y_ = batch_y.to(device)
        optimizer.zero_grad()
        predictions = net(batch_x_)
        loss = criterion(predictions, batch_y_)
        loss.backward()
        # clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        _, predicted_labels = predictions.max(1)
        num_corrected_predictions_train += batch_y_.eq(predicted_labels).sum().item()
        total_loss_train += loss
      acc_train = (num_corrected_predictions_train / num_traning_samples)
      loss_train = (total_loss_train / num_batches_train)
      acc_epoch_train[idx_epoch] = acc_train
      loss_epoch_train[idx_epoch] = loss_train
      
      """ evaluation process. """
      net.eval()
      num_corrected_predictions_test = 0
      total_loss_test = 0
      for (batch_x_test, batch_y_test) in test_set:
        batch_x_ = batch_x_test.view(-1, 3, 32,32).to(device)
        batch_y_ = batch_y_test.to(device)
        with torch.no_grad():
          predictions = net(batch_x_)
          loss = criterion(predictions, batch_y_)
        _, predicted_labels = predictions.max(1)
        num_corrected_predictions_test += batch_y_.eq(predicted_labels).sum().item()
        total_loss_test += loss
      acc_test = (num_corrected_predictions_test / num_test_samples)
      loss_test = (total_loss_test / num_batches_test)
      acc_epoch_test[idx_epoch] = acc_test
      loss_epoch_test[idx_epoch] = loss_test

      scheduler.step()

      print(f"Epoch {idx_epoch}:\n \tloss train: {loss_train:.3f}\n \tloss test: {loss_test:.3f}\n \tacc train: {acc_train*100:.3}\n \tacc test: {acc_test*100:.3f}\n")
        
    print(f"Best test-acc: {np.max(acc_epoch_test)*100:.3}")

    export_metrics_values(file_metrics, acc_epoch_train, acc_epoch_test, loss_epoch_train, loss_epoch_test)


if __name__ == "__main__": 
    main() 
