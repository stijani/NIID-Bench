import copy
from math import *
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from algorithms.gradiance.update_local import local_train_net_gradiance


def global_train_net_gradiance(
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
    trained_state_dicts = {}
    all_clients_unbiased_step_grads = []
    for net_id in nets:
        if net_id not in selected:
            continue

        net = copy.deepcopy(nets[net_id])
        # get the data indexes for this client
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            # get dataloader for the unbiased step
            unbiased_train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 1024, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            # get dataloader for the unbiased step
            unbiased_train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, 1024, 32, dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 1024)

        trained_state_dict, unbiased_grad_dict = local_train_net_gradiance(net_id, net, global_model, train_dl_local, test_dl, unbiased_train_dl_local, args.num_local_steps, args.lr, args.optimizer, args.mu, args.beta_, aggregated_unbiased_grads, logger, device=device)
        all_clients_unbiased_step_grads.append(unbiased_grad_dict)
        trained_state_dicts[net_id] = trained_state_dict
    return trained_state_dicts, all_clients_unbiased_step_grads
