import argparse
from math import ceil, pow
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
  Wide Residual Networks:
  - Paper: https://arxiv.org/pdf/1605.07146.pdf
  - PyTorch reference code 1 : https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
  - PyTorch reference code 2 : https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
"""

def learning_rate_scheduler(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init * pow(0.2, optim_factor)

class BasicWideBlock(nn.Module):
  
  """
    Basic-wide block optionally with dropout
  """
  
  def __init__(self, in_planes, out_planes, stride=1, dropout_rate=None, bias=True):
    super(BasicWideBlock, self).__init__()
    self.dropout_rate = dropout_rate
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)
    self.bn2 = nn.BatchNorm2d(out_planes)
    self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
    # Adapt input map for the shorcut (when required)
    if stride > 1 or in_planes != out_planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias),
      )
    else:
      self.shortcut = nn.Sequential()
    if self.dropout_rate:
       self.dropout = nn.Dropout(p=dropout_rate)
  
  def forward(self, x):
    if self.dropout_rate:
      out = self.dropout(self.conv1(F.relu(self.bn1(x))))
    else:
      out = self.conv1(F.relu(self.bn1(x)))
    out = self.conv2(F.relu(self.bn2(out)))
    out += self.shortcut(x)
    return out

class WideResNet(nn.Module):
  
  """
    Block type = B(3,3)
  """
  
  def _make_wide_layer(self, block, in_planes, out_planes, n, stride, dropout_rate=None):
    layers = []
    in_planes_ = in_planes
    strides = [1 for _ in range(n)]
    strides[0] = stride
    for i in range(n):
      layers.append(block(in_planes_, out_planes, stride=strides[i], dropout_rate=dropout_rate))
      in_planes_ = out_planes
    return nn.Sequential(*layers)

  def __init__(self, block, num_classes, num_layers=40, k=1, dropout_rate=None):
    """
      k: widen factor
      num_layers: depth network
    """
    super(WideResNet, self).__init__()
    num_channels = [16, 16*k, 32*k, 64*k]
    assert((num_layers - 4) % 6 == 0)
    n = ((num_layers - 4) % 6 == 0)
    self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3, padding=1, stride=1, bias=True) # [:, 16, 32, 32]
    self.conv2 = self._make_wide_layer(block, num_channels[0], num_channels[1], n, 1, dropout_rate=dropout_rate) # [:, 16*k, 32, 32]
    self.conv3 = self._make_wide_layer(block, num_channels[1], num_channels[2], n, 2, dropout_rate=dropout_rate) # [:, 32*k, 16, 16]
    self.conv4 = self._make_wide_layer(block, num_channels[2], num_channels[3], n, 2, dropout_rate=dropout_rate) # [:, 64*k, 8, 8]
    self.bn = nn.BatchNorm2d(num_channels[3])
    self.linear = nn.Linear(num_channels[3], num_classes)

  def forward(self, x):
    # x: [:, 3, 32, 32]
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = F.relu(self.bn(out))
    out = F.avg_pool2d(out, 8)
    out = out.flatten(start_dim=1)
    out = self.linear(out)
    return out


def main():
    
    """ arguments. """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./dataset/cifar10/")
    parser.add_argument("--metrics_path", type=str, default="./metrics/cifar10/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0)
    parser.add_argument("--initial_lr", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
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

    num_layers = args.num_layers
    k = args.k
    dropout_rate = args.dropout_rate
    num_classes = 10
    
    net = WideResNet(BasicWideBlock, num_classes, num_layers=num_layers, k=k, dropout_rate=dropout_rate).to(device)

    """ training process. """
    
    initial_lr = args.initial_lr
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    momentum = args.momentum
    # max_grad_norm = args.max_grad_norm

    model_name = f"wide_resnet_num_layers={num_layers}_k={k}_dropout={dropout_rate}"
    metrics_path = args.metrics_path
    if not os.path.isdir(metrics_path):
      os.makedirs(metrics_path)
    file_metrics = os.path.join(metrics_path, f"{model_name}.csv")

    criterion = nn.CrossEntropyLoss()

    acc_epoch_train = np.empty(num_epochs)
    acc_epoch_test = np.empty(num_epochs)
    loss_epoch_train = np.empty(num_epochs)
    loss_epoch_test = np.empty(num_epochs)
    
    for idx_epoch in range(0, num_epochs, 1):
        
      net.train()
      num_corrected_predictions_train = 0
      total_loss_train = 0
      optimizer = optim.SGD(net.parameters(), lr=learning_rate_scheduler(initial_lr, idx_epoch), weight_decay=weight_decay, momentum=momentum)
      
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

      print(f"Epoch {idx_epoch}:\n \tloss train: {loss_train:.3f}\n \tloss test: {loss_test:.3f}\n \tacc train: {acc_train*100:.3f}\n \tacc test: {acc_test*100:.3f}\n")
        
    print(f"Best test-acc: {np.max(acc_epoch_test)*100:.3f}")

    export_metrics_values(file_metrics, acc_epoch_train, acc_epoch_test, loss_epoch_train, loss_epoch_test)


if __name__ == "__main__": 
    main() 
