#!/bin/bash

dataset_path="./../dataset/cifar10"
metrics_path="./../metrics/cifar10/"
seed=42
batch_size=64
initial_lr=0.1
num_epochs=125
momentum=0.9
weight_decay=5e-4
gamma=0.1
model=16


python3 vgg.py \
--dataset_path ${dataset_path} \
--metrics_path ${metrics_path} \
--model ${model} \
--seed ${seed} \
--batch_size ${batch_size} \
--initial_lr ${initial_lr} \
--num_epochs ${num_epochs} \
--weight_decay ${weight_decay} \
--gamma ${gamma} \
--milestones 80 \
--momentum ${momentum}