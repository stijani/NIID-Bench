from math import *
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from algorithms.fednova.update_local import local_train_net_fednova


def global_train_net_fednova(
    nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"
):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info(
            "Training network %s. n_training: %d" % (str(net_id), len(dataidxs))
        )
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == "space":
            train_dl_local, test_dl_local, _, _ = get_dataloader(
                args.dataset,
                args.datadir,
                args.batch_size,
                32,
                dataidxs,
                noise_level,
                net_id,
                args.n_parties - 1,
            )
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(
                args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level
            )
        # train_dl_global, test_dl_global, _, _ = get_dataloader(
        #     args.dataset, args.datadir, args.batch_size, 32
        # )
        # n_epoch = args.epochs

        a_i, d_i = local_train_net_fednova(
            args,
            net_id,
            net,
            global_model,
            train_dl_local,
            test_dl,
            args.num_local_steps,
            args.lr,
            args.optimizer,
            device=device,
        )

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
    #     logger.info("net %d final test acc %f" % (net_id, testacc))
    #     avg_acc += testacc

    # avg_acc /= len(selected)
    # if args.alg == "local_training":
    #     logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list


