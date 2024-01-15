#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import os

project_dir = os.path.abspath('.')
data_dir = '/home/stijani/projects/dataset'
dataset = 'cifar100'

config = {
    # experiment specifics
    'dataset_name': dataset,
    'run_summary': 'code-test',
    'filename_acc': os.path.join(project_dir, 'experiments', dataset, 'niid-1', 'hyper-parameter-tuning/acc/test_23_03_2023.csv'), 
    'filename_loss': os.path.join(project_dir, 'experiments', dataset, 'niid-1', 'hyper-parameter-tuning/loss/test_23_03_2023.csv'),

    # local
    'model-arch': 'lenet',
    #'model-arch': 'resnet18',
    'local_steps': 15, # tunned (7 aggr, 15 for feprox, 15 for fedavg: all at 1000-clients-frac-0.1)
    'beta': 0.99, # best value after tunning is 0.99
    'mu': 0.01,
    'local_bs': 64,
    # 'use_unbias_step': False, # (the unbiase step serves as our momemtum)
    # 'use_aggregated_unbiased_grad': True,
    'bs_unbiase_step': 1024,
    'local_lr': 0.01,
    'momentum': 0.9,
    'use_aug': True,
    'niid': None,

    # global
    'comm_rounds': 1000,
    'stopping_rounds': 1000,
    'num_clients': 10,
    'frac': 1,
    'use_saved_weights': False, #True,
    'use_proxy_data': False,
    # 'train_with_only_classical_momentum': False, # if this is True, we are just doing FedAvg(but with momentum)
    'init_weight_path': '/home/stijani/projects/phd/paper-2/phd-paper2-code/experiments/cifar100/initialization_weights/lenet.pt',

    # hardware - gpu
    'use_gpu': True,
    'gpu_id': 2,

    # data set
    'proxy_data_perc': 0.1, #0.05,
    'train-features': os.path.join(data_dir, dataset, 'train_features.npy'),
    'train-labels': os.path.join(data_dir, dataset, 'train_labels.npy'),
    'test-features': os.path.join(data_dir, dataset, 'test_features.npy'),
    'test-labels': os.path.join(data_dir, dataset, 'test_labels.npy'),
    'proxy_features': os.path.join(data_dir, 'tiny-imagenet/tinyimagenet_32x32_ch_last_np/train_features.npy'),
    'proxy_labels': os.path.join(data_dir, 'tiny-imagenet/tinyimagenet_32x32_ch_last_np/train_labels.npy')
}