#!/bin/bash

dataset_path="./../dataset/cifar10"
metrics_path="./../metrics/cifar10/"
seed=42
batch_size=128
num_layers=28
k=10
dropout_rate=0.3
initial_lr=0.1
num_epochs=200
weight_decay=5e-4
momentum=0.9
gamma=0.2
# max_grad_norm=1

python3 wide_resnet.py \
--dataset_path ${dataset_path} \
--metrics_path ${metrics_path} \
--seed ${seed} \
--batch_size ${batch_size} \
--num_layers ${num_layers} \
--k ${k} \
--dropout_rate ${dropout_rate} \
--initial_lr ${initial_lr} \
--num_epochs ${num_epochs} \
--weight_decay ${weight_decay} \
--momentum ${momentum} \
# --max_grad_norm ${max_grad_norm}

