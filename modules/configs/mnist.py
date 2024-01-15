#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import os

project_dir = os.path.abspath('.')
data_dir = '/home/stijani/projects/dataset'
dataset = 'mnist'

config = {
    # experiment specifics
    'dataset_name': dataset,
    'run_summary': 'code-test',
    'filename_acc': os.path.join(project_dir, 'experiments', dataset, 'niid-1', 'hyper-parameter-tuning/acc/proxy_data_perc.csv'), 
    'filename_loss': os.path.join(project_dir, 'experiments', dataset, 'niid-1', 'hyper-parameter-tuning/loss/proxy_data_perc.csv'),

    # local
    'model-arch': 'mlp',
    'local_steps': 10, # tunned for all algo
    'beta': 0.99, #0.99, # best value after tunning is 0.99
    'local_bs': 64,
    #'use_unbias_step': False, # (the unbiase step serves as our momemtum)
    #'use_aggregated_unbiased_grad': True,
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
    'use_saved_weights': True,
    'use_proxy_data': False,
    'train_with_only_classical_momentum': False, # if this is True, we are just doing FedAvg(but with momentum)
    'init_weight_path': '/home/stijani/projects/phd/paper-2/phd-paper2-code/experiments/mnist/initialization_weights/mlp.pt',

    # hardware - gpu
    'use_gpu': True,
    'gpu_id': 2,

    # data set
    'proxy_data_perc': 0.25,
    'train-features': os.path.join(data_dir, dataset, 'train_features.npy'),
    'train-labels': os.path.join(data_dir, dataset, 'train_labels.npy'),
    'test-features': os.path.join(data_dir, dataset, 'test_features.npy'),
    'test-labels': os.path.join(data_dir, dataset, 'test_labels.npy'),
    'proxy_features': os.path.join(data_dir, 'mnist-fashion/train_features.npy'),
    'proxy_labels': os.path.join(data_dir, 'mnist-fashion/train_labels.npy')
}