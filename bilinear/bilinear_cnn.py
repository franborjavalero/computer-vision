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
import torchvision.models as models
from utils import export_metrics_values
import PIL

"""
  Bilinear CNN Models For Fine-Grained Visual Recognition:
    - Paper: https://pdfs.semanticscholar.org/3a30/7b7e2e742dd71b6d1ca7fde7454f9ebd2811.pdf
    - Reference code: https://github.com/dasguptar/bcnn.pytorch/blob/master/bcnn/model.py
    - Reference code: https://github.com/HaoMood/bilinear-cnn/blob/master/src/bilinear_cnn_fc.py
"""

class BilinearCNN(nn.Module):
  
  def __init__(self, num_classes=20, freeze=True, dim_classifier=512, ouput_vgg=15, depth_vgg=512):
    super(BilinearCNN, self).__init__()
    self.dim_classifier = dim_classifier
    self.ouput_vgg = ouput_vgg
    self.depth_vgg = depth_vgg
    # Ignores the last pooling operation, for this reason the outpute size is double
    self.feature_extractor = nn.Sequential(*list(models.vgg16(pretrained=True).features)[:-1]) 
    self.classifier = nn.Linear(dim_classifier ** 2, num_classes)
    self.dropout = nn.Dropout(p=0.5)
    self.freeze = freeze
    # Freeze petrained model
    if self.freeze:
      for param in self.feature_extractor.parameters():
        param.requires_grad = False
    # Initialize properly the FC layers
    nn.init.kaiming_normal_(self.classifier.weight.data)
    if self.classifier.bias is not None:
      nn.init.constant_(self.classifier.bias.data, val=0)

  def forward(self, x, threshold=1e-5):
    """ forward pretained model: vgg. """
    out = self.feature_extractor(x)
    """ outer product. """
    out = out.view(-1, self.depth_vgg, self.ouput_vgg ** 2)
    out = torch.bmm(out, out.permute(0, 2, 1)) # bilinear vector: [batch_size, depth_vgg, depth_vgg]
    out = out.view(-1, self.depth_vgg ** 2)
    out = torch.div(out, self.ouput_vgg ** 2) # divide by feature map size (normalization)
    out = torch.sign(out) * torch.sqrt(out + threshold) # signed square root (normalization)
    out = nn.functional.normalize(out, p=2, dim=1) # l2 normalization
    out = self.dropout(out)
    """ linear projection. """
    out = self.classifier(out)
    return out
  
  def unfreeze_vgg(self):
    if self.freeze:
      for param in self.feature_extractor.parameters():
        param.requires_grad = True
      self.freeze = False
    
  def freeze_vgg(self):
    if not self.freeze:
      for param in self.feature_extractor.parameters():
        param.requires_grad = False
      self.freeze = True


class CustomTensorDataset(torch.utils.data.TensorDataset):
  
  # https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset

  def __init__(self, X, y, transform=None):
    super(CustomTensorDataset, self).__init__()
    assert X.size(0) == y.size(0)
    self.X = X
    self.y = y
    self.transform = transform

  def __getitem__(self, index):
    x_ = self.X[index]
    if self.transform:
        x_ = self.transform(x_)
    y_ = self.y[index]
    return x_, y_

  def __len__(self):
    return self.y.size(0)

def main():
    
    """ arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./dataset/car_identification/")
    parser.add_argument("--metrics_path", type=str, default="./metrics/car_identification/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--p1_initial_lr", type=float, default=1)
    parser.add_argument("--p1_num_epochs", type=int, default=55)
    parser.add_argument("--p1_momentum", type=float, default=0.9)
    parser.add_argument("--p1_weight_decay", type=float, default=1e-8)
    parser.add_argument("--p1_gamma", type=float, default=0.1)
    parser.add_argument("--p1_milestones", nargs='+', type=int, default=None)
    parser.add_argument("--p2_initial_lr", type=float, default=1)
    parser.add_argument("--p2_num_epochs", type=int, default=55)
    parser.add_argument("--p2_momentum", type=float, default=0.9)
    parser.add_argument("--p2_weight_decay", type=float, default=1e-8)
    parser.add_argument("--p2_gamma", type=float, default=0.1)
    parser.add_argument("--p2_milestones", nargs='+', type=int, default=None)
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
    url_dataset = "wget https://www.dropbox.com/s/sakfqp6o8pbgasm/data.tgz"
    if not os.path.isdir(dataset_path):
      os.makedirs(dataset_path)
      compressed_file = os.path.join(dataset_path, "data.tgz")
      os.system(f'wget {url_dataset}')
      os.system(f'mv data.tgz {compressed_file}')
      os.system(f'tar xvzf {compressed_file} -C {dataset_path}')
    
    X_train = torch.tensor(np.load(os.path.join(dataset_path, 'x_train.npy'))).permute(0, 3, 1, 2).float() / 255.
    X_test = torch.tensor(np.load(os.path.join(dataset_path, 'x_test.npy'))).permute(0, 3, 1, 2).float() / 255.
    y_train = torch.tensor(np.load(os.path.join(dataset_path, 'y_train.npy'))) - 1
    y_test = torch.tensor(np.load(os.path.join(dataset_path, 'y_test.npy'))) - 1

    train_transforms = torchvision.transforms.Compose([
      transforms.ToPILImage(),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.RandomCrop(size=250),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    test_transforms = torchvision.transforms.Compose([
      transforms.ToPILImage(),
      # torchvision.transforms.CenterCrop(size=250),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    batch_size = args.batch_size
    train = CustomTensorDataset(X_train, y_train, transform=train_transforms)
    test = CustomTensorDataset(X_test, y_test, transform=test_transforms)
    num_traning_samples = len(train)
    num_test_samples = len(test)
    num_batches_train = ceil(num_traning_samples / batch_size)
    num_batches_test = ceil(num_test_samples / batch_size)
    train_set = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_set = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    """ define network. """
    model = BilinearCNN().to(device)

    """ training method. """   
    metrics_path = args.metrics_path
    def training_method(model_name, net, initial_lr, weight_decay, momentum, milestones, gamma, num_epochs, improve_acc=0.000):
      best_model = net
      best_acc = improve_acc
      if not os.path.isdir(metrics_path):
        os.makedirs(metrics_path)
      file_metrics = os.path.join(metrics_path, f"{model_name}.csv")
      optimizer = optim.SGD(net.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=momentum)
      if milestones:
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
          batch_x_ = batch_x.view(-1, 3, 250, 250).to(device)
          batch_y_ = batch_y.to(device)
          optimizer.zero_grad()
          predictions = net(batch_x_)
          loss = criterion(predictions, batch_y_)
          loss.backward()
          optimizer.step()
          predictions = predictions.detach().cpu().numpy()
          predicted_labels = np.argmax(predictions, axis=1)
          batch_y_ = batch_y_.to('cpu').numpy()
          num_corrected_predictions_train += np.sum(predicted_labels == batch_y_)
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
          batch_x_ = batch_x_test.view(-1, 3, 250, 250).to(device)
          batch_y_ = batch_y_test.to(device)
          with torch.no_grad():
            predictions = net(batch_x_)
            loss = criterion(predictions, batch_y_)
          predictions = predictions.detach().cpu().numpy()
          predicted_labels = np.argmax(predictions, axis=1)
          batch_y_ = batch_y_.to('cpu').numpy()
          num_corrected_predictions_test += np.sum(predicted_labels == batch_y_)
          total_loss_test += loss
        acc_test = (num_corrected_predictions_test / num_test_samples)
        loss_test = (total_loss_test / num_batches_test)
        acc_epoch_test[idx_epoch] = acc_test
        loss_epoch_test[idx_epoch] = loss_test
        if acc_epoch_test[idx_epoch] > best_acc:
          best_model = net
          best_acc = acc_epoch_test[idx_epoch]
        if milestones:
          scheduler.step()
        print(f"Epoch {idx_epoch}:\n \tloss train: {loss_train:.3f}\n \tloss test: {loss_test:.3f}\n \tacc train: {acc_train*100:.3}\n \tacc test: {acc_test*100:.3f}\n")  
      print(f"Best test-acc: {np.max(acc_epoch_test)*100:.3}")
      export_metrics_values(file_metrics, acc_epoch_train, acc_epoch_test, loss_epoch_train, loss_epoch_test)
      return best_model, best_acc
    
    """ phase 1. """
    model_name = "bilinear_VGG16_phase1"
    initial_lr = args.p1_initial_lr
    num_epochs = args.p1_num_epochs
    momentum = args.p1_momentum
    weight_decay = args.p1_weight_decay
    milestones = args.p1_milestones
    gamma = args.p1_gamma
    model_p1, acc_p1 = training_method(model_name, model, initial_lr, weight_decay, momentum, milestones, gamma, num_epochs)

    """ phase 2. """
    model_p1.unfreeze_vgg()
    model_name = "bilinear_VGG16_phase2"
    initial_lr = args.p2_initial_lr
    num_epochs = args.p2_num_epochs
    momentum = args.p2_momentum
    weight_decay = args.p2_weight_decay
    milestones = args.p2_milestones
    gamma = args.p2_gamma
    model_p2, acc_p2 = training_method(model_name, model_p1, initial_lr, weight_decay, momentum, milestones, gamma, num_epochs, improve_acc=acc_p1)


if __name__ == "__main__": 
    main() 
