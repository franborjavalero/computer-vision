#!/bin/bash

dataset_path="./../dataset/car_identification/"
metrics_path="./../metrics/car_identification/"
seed=42
batch_size=64
momentum=0.9
gamma=0.1

p1_initial_lr=1
p1_num_epochs=200
p1_weight_decay=1e-8

p2_initial_lr=0.01
p2_num_epochs=25
p2_weight_decay=1e-5

python3 bilinear_cnn.py \
--dataset_path ${dataset_path} \
--metrics_path ${metrics_path} \
--seed ${seed} \
--batch_size ${batch_size} \
--p1_initial_lr ${p1_initial_lr} \
--p1_num_epochs ${p1_num_epochs} \
--p1_momentum ${momentum} \
--p1_weight_decay ${p1_weight_decay} \
--p2_initial_lr ${p2_initial_lr} \
--p2_num_epochs ${p2_num_epochs} \
--p2_momentum ${momentum} \
--p2_weight_decay ${p2_weight_decay}
