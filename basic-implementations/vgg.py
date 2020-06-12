import argparse
from math import ceil, pow, floor
import numpy as np
import os
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from utils import export_metrics_values

"""
  VGG:
    - Paper: https://arxiv.org/pdf/1409.1556.pdf
    - PyTorch reference code: https://github.com/facebookresearch/mixup-cifar10/blob/master/models/vgg.py
"""

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
  
  def __init__(self, vgg_name):
    super(VGG, self).__init__()
    self.features = self._make_layers(cfg[vgg_name])
    self.classifier = nn.Linear(512, 10)

  def forward(self, x):
    out = self.features(x)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    return out

  def _make_layers(self, cfg):
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [
              nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
              nn.BatchNorm2d(x),
              nn.ReLU(inplace=True)
            ]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
  """Returns mixed inputs, pairs of targets, and lambda"""
  if alpha > 0:
      lam = np.random.beta(alpha, alpha)
  else:
      lam = 1
  batch_size = x.size()[0]
  if use_cuda:
      index = torch.randperm(batch_size).cuda()
  else:
      index = torch.randperm(batch_size)
  mixed_x = lam * x + (1 - lam) * x[index, :]
  y_a, y_b = y, y[index]
  return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    
    """ arguments. """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./dataset/cifar10/")
    parser.add_argument("--metrics_path", type=str, default="./metrics/cifar10/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=int, choices=[11, 13, 16, 19], default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--initial_lr", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=125)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--milestones", nargs='+', type=int, default=[80])
    args = parser.parse_args()
    
    """ reproducibility. """
    
    device = "cpu"
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_cuda = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = "cuda:0"
        use_cuda = True
    
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

    model_name = f"VGG{args.model}"
    net = VGG(model_name).to(device)
    
    """ training process. """
    
    initial_lr = args.initial_lr
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    momentum = args.momentum
    gamma = args.gamma
    milestones = args.milestones

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
        inputs, targets_a, targets_b, lam = mixup_data(batch_x_, batch_y_, 1, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        predictions = net(inputs)
        loss = mixup_criterion(criterion, predictions, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        _, predicted_labels = predictions.max(1)
        num_corrected_predictions_train += (lam * predicted_labels.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted_labels.eq(targets_b.data).cpu().sum().float())
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
