import sys
import copy
from math import *
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from algorithms.gradiance.update_local import local_train_net_gradiance
sys.path.append('/home/stijani/projects/phd/paper-2/phd-paper2-code/modules')
from custom_image_dataset import CustomImageDataset


# def global_train_net_gradiance(
#     net,
#     selected,
#     global_model,
#     args,
#     net_dataidx_map,
#     test_dl,
#     device,
#     aggregated_unbiased_grads=None,
#     logger=None,
#     # all_dataset_dict
#     ):
#     trained_state_dicts = {}
#     all_clients_unbiased_step_grads = []
#     for client_id in selected:
#         #if client_id not in selected:
#             #continue
#         client_net = copy.deepcopy(net)
#         client_net.to(device)
#         use_aug = True
#         train_dl_local = DataLoader(CustomImageDataset(net_dataidx_map[client_id], args.dataset, args.use_aug), batch_size=args.batch_size, shuffle=True)
#         unbiased_train_dl_local = DataLoader(CustomImageDataset(net_dataidx_map[client_id], args.dataset, use_aug), batch_size=1024, shuffle=True)
#         trained_state_dict, unbiased_grad_dict = local_train_net_gradiance(client_id, client_net, global_model, train_dl_local, test_dl, unbiased_train_dl_local, args.num_local_steps, args.lr, args.optimizer, args.mu, args.beta_, aggregated_unbiased_grads, logger, device=device)
#         all_clients_unbiased_step_grads.append(unbiased_grad_dict)
#         trained_state_dicts[client_id] = trained_state_dict
#         del client_net
        
#     return trained_state_dicts, all_clients_unbiased_step_grads

def global_train_net_gradiance(
    nets,
    selected,
    global_model,
    args,
    net_dataidx_map,
    test_dl,
    device,
    aggregated_unbiased_grads=None,
    logger=None,
    # all_dataset_dict
    ):
    trained_state_dicts = {}
    all_clients_unbiased_step_grads = []
    for net_id in nets:
        if net_id not in selected:
            continue
        net = copy.deepcopy(nets[net_id])
        net.to(device)
        use_aug = True
        train_dl_local = DataLoader(CustomImageDataset(net_dataidx_map[net_id], args.dataset, args.use_aug), batch_size=args.batch_size, shuffle=True)
        unbiased_train_dl_local = DataLoader(CustomImageDataset(net_dataidx_map[net_id], args.dataset, use_aug), batch_size=1024, shuffle=True)
        trained_state_dict, unbiased_grad_dict = local_train_net_gradiance(net_id, net, global_model, train_dl_local, test_dl, unbiased_train_dl_local, args.num_local_steps, args.lr, args.optimizer, args.mu, args.beta_, aggregated_unbiased_grads, logger, device=device)
        all_clients_unbiased_step_grads.append(unbiased_grad_dict)
        trained_state_dicts[net_id] = trained_state_dict
    return trained_state_dicts, all_clients_unbiased_step_grads
