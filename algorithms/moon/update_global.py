import sys
from math import *
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from algorithms.moon.update_local import local_train_net_moon
sys.path.append('/home/stijani/projects/phd/paper-2/phd-paper2-code/modules')
from custom_image_dataset import CustomImageDataset


def global_train_net_moon(
    nets,
    selected,
    args,
    net_dataidx_map,
    test_dl=None,
    global_model=None,
    prev_model_pool=None,
    comm_round=None,
    device="cpu",
):
    avg_acc = 0.0
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info(
            "Training network %s. n_training: %d" % (str(net_id), len(dataidxs))
        )
        net.to(device)

        # noise_level = args.noise
        # if net_id == args.n_parties - 1:
        #     noise_level = 0

        # if args.noise_type == "space":
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(
        #         args.dataset,
        #         args.datadir,
        #         args.batch_size,
        #         32,
        #         dataidxs,
        #         noise_level,
        #         net_id,
        #         args.n_parties - 1,
        #     )
        # else:
        #     noise_level = args.noise / (args.n_parties - 1) * net_id
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(
        #         args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level
        #     )
        # train_dl_global, test_dl_global, _, _ = get_dataloader(
        #     args.dataset, args.datadir, args.batch_size, 32
        # )
        # n_epoch = args.epochs

        train_dl_local = DataLoader(CustomImageDataset(net_dataidx_map[net_id], args.dataset, args.use_aug), batch_size=args.batch_size, shuffle=True)

        prev_models = []
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        local_train_net_moon(
            net_id,
            net,
            global_model,
            prev_models,
            train_dl_local,
            test_dl,
            args.num_local_steps,
            args.lr,
            args.optimizer,
            args.mu,
            args.temperature,
            args,
            comm_round,
            device=device,
        )
        # logger.info("net %d final test acc %f" % (net_id, testacc))
        # avg_acc += testacc

    # avg_acc /= len(selected)
    # if args.alg == "local_training":
    #     logger.info("avg test acc %f" % avg_acc)
    ### global_model.to('cpu')
    nets_list = list(nets.values())
    return nets_list


def get_partition_dict(
    dataset,
    partition,
    n_parties,
    init_seed=0,
    datadir="./data",
    logdir="./logs",
    beta=0.5,
):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
    ) = partition_data(dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map