import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from gradiance.local_update_per_client import train_net_gradiance


def local_train_net_gradiance(
    nets, 
    selected, 
    global_model, 
    args, 
    net_dataidx_map, 
    test_dl, 
    device, 
    aggregated_unbiased_grads=None,
    logger=None
    ):
    avg_acc = 0.0
    all_clients_unbiased_step_grads = []
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        # get the data indexes for this client
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 1000, dataidxs, noise_level, net_id, args.n_parties-1)
            # get dataloader for the unbiased step
            unbiased_train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 1024, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 1000, dataidxs, noise_level)
            # get dataloader for the unbiased step
            unbiased_train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 1024, dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 1024)
        #n_epoch = args.epochs

        trainacc, testacc, unbiased_grad_dict = train_net_gradiance(net_id, net, global_model, train_dl_local, test_dl, unbiased_train_dl_local, args.num_local_steps, args.lr, args.optimizer, args.mu, args.beta_, aggregated_unbiased_grads, logger, device=device)
        all_clients_unbiased_step_grads.append(unbiased_grad_dict)
        logger.info("net %d final test acc %f" % (net_id, testacc))

        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
    return all_clients_unbiased_step_grads