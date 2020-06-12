#!/bin/bash

dataset_path="./../dataset/cifar10/"
metrics_path="./../metrics/cifar10/"
seed=42
batch_size=64
growth_rate=12
bn_size=4
reduction=0.5
initial_lr=0.1
num_epochs=300
weight_decay=5e-4
momentum=0.9
gamma=0.1
# max_grad_norm=1

python3 dense_nets.py \
--dataset_path ${dataset_path} \
--metrics_path ${metrics_path} \
--seed ${seed} \
--batch_size ${batch_size} \
--growth_rate ${growth_rate} \
--block_config 6 12 24 16 \
--bn_size ${bn_size} \
--reduction ${reduction} \
--initial_lr ${initial_lr} \
--num_epochs ${num_epochs} \
--weight_decay ${weight_decay} \
--momentum ${momentum} \
--gamma ${gamma} \
--milestones 150 225 \
# --max_grad_norm ${max_grad_norm}
