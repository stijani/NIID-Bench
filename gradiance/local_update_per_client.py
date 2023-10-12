#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import sys
import torch
from torch import nn
from itertools import cycle
from gradiance.optimizer import GradianceOptimizer
import torch.optim as optim
from utils import *


def get_parameter_grads(model):
    """
    Extracts the gradients of the parameters in each of the layers of a model
    and returns them as a dictionary sharing keys with the model's state dictionary
    :param trained_model: a pytorch model
    :return: a python dictionary
    """
    grad_dict = {}
    layer_names = model.state_dict().keys()
    with torch.no_grad():
        for name, layer_param in zip(layer_names, model.parameters()): 
            grad_dict[name] = copy.deepcopy(layer_param.grad)
    return copy.deepcopy(grad_dict)


def train_net_gradiance(
    net_id, 
    net, 
    global_net, 
    train_dataloader, 
    test_dataloader,
    unbiased_train_dataloader, 
    num_local_steps, 
    lr, 
    args_optimizer, 
    mu, 
    beta, 
    aggregated_unbiased_grads, 
    logger, 
    device="cpu"
    ):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    train_acc, train_loss = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_loss = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    criterion = nn.CrossEntropyLoss().to(device)
    cnt = 0 # TODO compare this to local_steps, stop the training loop once cnt == local_steps 
    # global_weight_collector = list(global_net.to(device).parameters())
    lu = LocalUpdate(net_id, 
                        net, 
                        num_local_steps, # local_updates
                        lr, 
                        beta, 
                        train_dataloader, 
                        test_dataloader, 
                        aggregated_unbiased_grads, 
                        unbiased_train_dataloader, 
                        device,
                        logger
                        )
    net, unbiased_grad_dict = lu.update_weights()
    #net.to(device) ######################################### works
    train_acc, train_loss = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_loss = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, unbiased_grad_dict


class LocalUpdate(object):
    def __init__(
        self, 
        cliend_idx, 
        net, 
        num_local_steps, 
        lr, 
        beta, 
        train_dataloader, 
        test_dataloader, 
        aggregated_unbiased_grads, 
        unbiased_train_dataloader=None, 
        device="cpu", 
        logger=None
        ):
        self.cliend_idx = cliend_idx
        self.net = net
        self.num_local_steps = num_local_steps
        self.lr = lr
        self.beta = beta
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.unbiased_train_dataloader = unbiased_train_dataloader
        self.aggregated_unbiased_grads = aggregated_unbiased_grads
        self.device = device
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss().to(device)
        
    def unbiased_update_step(self):
        """
        Compute unbiased first step update for a client and return the parameters
        :param global_model_clone -> current state of the global model as recieved from server
        :return gradient of the trainable parameters
        """
        #global_model_clone = self.net.clone()
        global_model_clone = copy.deepcopy(self.net)
        global_model_clone.train()
        images, labels = next(iter(self.unbiased_train_dataloader))
        images, labels = images.to(self.device), labels.to(self.device)
        global_model_clone.zero_grad()
        log_probs = global_model_clone(images)
        loss = self.criterion(log_probs, labels)
        loss.backward()
        gradients_dict = {} # to stroee grads against layer names
        for name, param in global_model_clone.named_parameters():
            if param.requires_grad:
                gradients_dict[name] = param.grad.clone()
        return gradients_dict

    # def update_weights(self):
    #     # unbiased update step
    #     unbiased_grad_dict = self.unbiased_update_step()
    #     batch_loss = []
               
    #     trainloader = cycle(self.train_dataloader)
    #     #model = copy.deepcopy(self.net)
    #     model = self.net
        
    #     if self.aggregated_unbiased_grads:
    #         self.optimizer = GradianceOptimizer(model.parameters(), self.aggregated_unbiased_grads, self.lr, self.beta)
    #     else:
    #         self.optimizer = optim.SGD(model.parameters(), lr=self.lr)
    #     # model = model.to(self.device) #############
    #     model.train()   
    #     for step in range(self.num_local_steps):
    #         # loop through the data for number of local steps
    #         images, labels = next(trainloader)
    #         images, labels = images.to(self.device), labels.to(self.device)
    #         model.zero_grad()
    #         log_probs = model(images)
    #         loss = self.criterion(log_probs, labels)
    #         loss.backward()
    #         self.optimizer.step()
    #         batch_loss.append(loss.item())
    #     self.logger.info('Epoch: %d Loss: %f' % (step, sum(batch_loss)/len(batch_loss)))
    #     train_acc = compute_accuracy(model, self.train_dataloader, device=self.device)
    #     test_acc, conf_matrix = compute_accuracy(model, self.test_dataloader, get_confusion_matrix=True, device=self.device)

    #     logger.info('>> Training accuracy: %f' % train_acc)
    #     logger.info('>> Test accuracy: %f' % test_acc)

    #     #model.to('cpu')
    #     logger.info(' ** Training complete **')
    #     return copy.deepcopy(model), copy.deepcopy(unbiased_grad_dict)
    #     #return model, copy.deepcopy(unbiased_grad_dict)

    def update_weights(self):
        # unbiased update step
        unbiased_grad_dict = self.unbiased_update_step()
               
        trainloader = cycle(self.train_dataloader)
        
        if self.aggregated_unbiased_grads:
            self.optimizer = GradianceOptimizer(self.net.parameters(), self.aggregated_unbiased_grads, self.lr, self.beta)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        self.net.train()
        sum_local_loss = 0.0   
        for step in range(self.num_local_steps):
            # loop through the data for number of local steps
            images, labels = next(trainloader)
            images, labels = images.to(self.device), labels.to(self.device)
            self.net.zero_grad()
            log_probs = self.net(images)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            self.optimizer.step()
            sum_local_loss += loss.item()
        logger.info(f'Client: {self.cliend_idx} | Averaged Local Loss: {round(sum_local_loss/self.num_local_steps, 2)}')
        train_acc, train_loss = compute_accuracy(self.net, self.train_dataloader, device=self.device)
        test_acc, conf_matrix, test_loss = compute_accuracy(self.net, self.test_dataloader, get_confusion_matrix=True, device=self.device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
        logger.info(' ** Training complete **')
        return self.net, copy.deepcopy(unbiased_grad_dict)