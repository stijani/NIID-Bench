#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import copy
import numpy as np
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from custom_image_dataset import CustomImageDataset
#from data_sharding.create_ext_data_ import ExtData
#from options import config

# my modules
from data_sharding.create_private_shards import Shard
#import config_ as config

# def get_dataset_():
#     X_train = np.load(config['train-features'])
#     y_train = np.load(config['train-labels'])
#     X_test = np.load(config['test-features'])
#     y_test = np.load(config['test-labels'])

#     shards = Shard(list(X_train), list(y_train), config['num_clients'])
#     all_client_data_pry = shards.niid_sharding(config['niid'])
    
#     ext_data = ExtData(all_client_data_pry,
#                   list(X_train),
#                   list(y_train),
#                   config['proxy_data_perc']
#                   )
#     all_client_data_sec = ext_data.create_ext_data_for_all_client()
    
#     # features, targets = ext.add_ext_single_client(client, perc_ext)
#     return {'data_pry':all_client_data_pry, 
#             'data_sec':all_client_data_sec, 
#             'data_test': (X_test, y_test)}


def add_proxy_data(data_pry, data_sec, proxy_perc):
    if data_sec is None:
        return
    features_pry, labels_pry = data_pry
    features_sec, labels_sec = data_sec

    # quantity of extension samples to use as % of the number of private shard samples
    idx = int(proxy_perc * len(labels_pry))
    final_features = list(features_pry) + list(features_sec)[:idx]
    final_labels = list(labels_pry) + list(labels_sec)[:idx]
    final_features, final_labels = shuffle_pairwise(final_features, final_labels)
    return np.array(final_features), np.array(final_labels)
    
    
def shuffle_pairwise(list1, list2):
    """"
    Takes in 2 list and shuffles them pairwise
    """
    zipped_list = list(zip(list1, list2))
    random.shuffle(zipped_list)
    shuffled_list1, shuffled_list2 = zip(*zipped_list)
    return list(shuffled_list1), list(shuffled_list2)


def apply_transform(data):
    apply_transform = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                        )
    features, labels = data
    transformed_data = CustomImageDataset(features, labels, apply_transform)
    

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


#def get_dataset_(args):


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(len(w)))
    return w_avg


# def average_weights_resnet(model_list):
#     model_params_list = [model.state_dict() for model in model_list]
#     num_models = len(model_params_list)
#     #print("#######",  [param.data.shape for idx, weights in enumerate(model_params_list) for param in weights if param.requires_grad is not None and idx < 2])
#     for param in model_params_list[0]:
#         if param.requires_grad:
#             avg_param = torch.mean(torch.stack([model_param.data for model_params in model_params_list for model_param in model_params]), dim=0)
#             param.data = avg_param
#     return model_params_list[0]

# def average_weights_resnet(model_list):
#     model_params_list = [model.state_dict() for model in model_list]
#     num_models = len(model_params_list)
#     averaged_state_dict = {key: value for key, value in model_params_list[0].items}
#     for name, params in zip(model_params_list[0].keys(), zip(*model_params_list)):
#         if params[0].requires_grad:
#             stacked_params = torch.stack([param.data for param in params])
#             avg_param = torch.mean(stacked_params, dim=0)
#             averaged_state_dict[name] = avg_param
#     return averaged_state_dict

def average_weights_resnet(model_list):
    model_params_list = [model.parameters() for model in model_list]
    averaged_params = []
    for params in zip(*model_params_list):
        #if params[0].requires_grad:
        stacked_params = torch.stack(params)
        avg_param = torch.mean(stacked_params, dim=0)
        averaged_params.append(avg_param)
    return averaged_params


def average_weights_resnet_and_assign(model_list):
    new_global_model = copy.deepcopy(model_list[0])
    model_params_list = [copy.deepcopy(list(model.parameters())) for model in model_list]
    averaged_params = []
    for params in zip(*model_params_list):
        #if params[0].requires_grad:
        stacked_params = torch.stack([param.data for param in params])
        avg_param = torch.mean(stacked_params, dim=0)
        averaged_params.append(avg_param)

    for param, averaged_param in zip(new_global_model.parameters(), averaged_params):
        param.data = averaged_param.data

    return new_global_model
    


def average_gradients_resnet(local_unbiased_models_params):
    #num_models = len(models)
    #model_grads_list = models #[]
    # for model in models:
    #     model_grads = [param.grad.data.clone() for param in model.parameters() if param.requires_grad] #and param.grad is not None]
    #     #print("########### utils_.py 188", model_grads )
    #     model_grads_list.append(model_grads)

    averaged_grads = []
    for params in zip(*local_unbiased_models_params):
        grads = [param.grad for param in params]
        averaged_grads.append(torch.mean(torch.stack(grads), dim=0))

    return averaged_grads

def load_new_params(model, averaged_params):
    for model_param, averaged_param in zip(model.parameters(), averaged_params):
        model_param.data.copy_(averaged_param)
    return model

# def average_gradients(*model_params_list):
#     num_models = len(model_params_list)
#     averaged_grads = []
#     for grads in zip(*model_params_list):
#         if grads[0] is not None:
#             stacked_grads = torch.stack(grads)
#             avg_grad = torch.mean(stacked_grads, dim=0)
#             averaged_grads.append(avg_grad)
#     return averaged_grads

def average_aggr_grads(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for i in range(1, len(w)):
        w_avg += w[i]
    #w_avg = torch.div(w_avg, float(len(w)))
    w_avg = [torch.div(i, float(len(w))) for i in w_avg]
    return w_avg


def get_parameter_grads1(model):
    """
    Extracts the gradients of the parameters in each of the layers of a model
    and returns them as a dictionary sharing keys with the model's state dictionary
    :param trained_model: a pytorch model
    :return: a python dictionary
    """
    grad_dict = {}
    layer_names = model.state_dict().keys()
    layer_params = model.parameters()
    with torch.no_grad():
        for name, layer_param in zip(layer_names, layer_params): 
            print('########source########', layer_param.grad)
            grad_dict[name] = copy.deepcopy(layer_param)
    return copy.deepcopy(grad_dict)


def get_parameter_grads(model):
    """
    Extracts the gradients of the parameters in each of the layers of a model
    and returns them as a dictionary sharing keys with the model's state dictionary
    :param trained_model: a pytorch model
    :return: a python dictionary
    """
    grad_dict = {}
    layer_names = model.state_dict().keys()
    #layer_params = model.parameters()
    with torch.no_grad():
        for name, layer_param in zip(layer_names, model.parameters()): 
            #print('########source122########', layer_param.grad)
            #grad_dict[name] = copy.deepcopy(layer_param)
            grad_dict[name] = copy.deepcopy(layer_param.grad)
    return copy.deepcopy(grad_dict)


def get_parameter_grads_testing(model):
    """
    Extracts the gradients of the parameters in each of the layers of a model
    and returns them as a dictionary sharing keys with the model's state dictionary
    :param trained_model: a pytorch model
    :return: a python dictionary
    """
    # grad_dict = {}
    # layer_names = model.state_dict().keys()
    # #layer_params = model.parameters()
    # with torch.no_grad():
    #     for name, layer_param in zip(layer_names, model.parameters()): 
    #         #print('########source122########', layer_param.grad)
    #         #grad_dict[name] = copy.deepcopy(layer_param)
    #         grad_dict[name] = copy.deepcopy(layer_param.grad)
    grads_of_trainable_params = [copy.deepcopy(p.grad.data) for p in model.parameters() if p.grad is not None]
    return copy.deepcopy(grads_of_trainable_params)


def get_dummy_grads(model, device):
    dummy_data = torch.randn(1, 3, 32, 32)
    dummy_data = dummy_data.to(device)
    out = model(dummy_data)
    loss = out.mean()
    loss.backward()
    grad_dict = {}
    for name, param in model.named_parameters():
        grad_dict[name] = param.grad
    return copy.deepcopy(grad_dict)
    


def exp_details(config):
    print('\nEXPERIMENTAL DETAILS:\n')
    print(f'    Model     : {config["run_summary"]}')
    print(f'    Model     : {config["model-arch"]}')
    print(f'    Local lr  : {config["local_lr"]}')
    print(f'    Beta  : {config["beta"]}')
    print(f'    Local steps  : {config["local_steps"]}')
    print(f'    Proxy Data Perc  : {config["proxy_data_perc"]}')
    print(f'    BS Unbiase Step  : {config["bs_unbiase_step"]}')
    print(f'    Use Momentum  : {config["momentum"]}')

    print('\n    FEDRATED PARAMETERS:')
    print(f'    NON IID-NESS  : {config["niid"]}')
    print(f'    Number of Clients  : {config["num_clients"]}')
    print(f'    Local Batch Size  : {config["local_bs"]}')
    print(f'    Unbiase Step Batch Size  : {config["bs_unbiase_step"]}')
    print(f'    Communication Rounds  : {config["comm_rounds"]}')
    print(f'    Number of Clients  : {config["num_clients"]}')
    print(f'    Fraction of Clients  : {config["frac"]}')
    print(f'    Use Save Weights  : {config["use_saved_weights"]}')
    print(f'    Use Proxy Data  : {config["use_proxy_data"]}')
    print(f'    Use Proxy Data  : {config["use_proxy_data"]}')
    print(f'    Use Proxy Data  : {config["use_aug"]}')    

    print('\n    DATA PATHS:')
    print(f'    Train Data  : {config["train-features"]}')
    print(f'    Test Data  : {config["test-features"]}')
    print(f'    Proxy Data  : {config["proxy_features"]}')

    print('\n    HARDWARE - GPU:')
    print(f'    GPU ID  : {config["gpu_id"]}')
    return
