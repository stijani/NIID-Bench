import numpy as np
import json
import time
from os.path import join
import torch
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
import datetime
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *


from algorithms.fedavg.update_global import global_train_net
from algorithms.fedprox.update_global import global_train_net_fedprox
from algorithms.gradiance.update_global import global_train_net_gradiance
from algorithms.fednova.update_global import global_train_net_fednova
from algorithms.scaffold.update_global import global_train_net_scaffold
from algorithms.moon.update_global import global_train_net_moon

from algorithms.fedavg.update_local import local_train_net
from algorithms.gradiance.utils import get_avg_of_unbiased_grads


def save_metrics(args, metric, filename="test_acc.csv"):
    args_json = json.dumps({k: v for k, v in vars(args).items()})
    metric = [args_json] + metric
    to_csv(metric, join(args.metric_dir, args.exp_category, filename))


def custom_logger(dir_="./exp_metrics/gradiance", filename="log.log"):
    filename = join(dir_, filename)
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    # Create a handler for logger1 (e.g., write logs to a file)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)

    # Create a formatter for logger1
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to logger1
    logger.addHandler(handler)
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mlp", help="neural network used in training"
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="dataset used for training"
    )
    parser.add_argument("--net_config", type=lambda x: list(map(int, x.split(", "))))
    parser.add_argument(
        "--partition", type=str, default="homo", help="the data partitioning strategy"
    )
    parser.add_argument(
        "--niid", type=int, default=0, help="number of unique labels per client, 0 means iid"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of local epochs"
    )  # no longer used
    parser.add_argument(
        "--num_local_steps", type=int, default=10, help="number of local epochs"
    )
    parser.add_argument(
        "--n_parties",
        type=int,
        default=2,
        help="number of workers in a distributed cluster",
    )
    parser.add_argument(
        "--alg",
        type=str,
        default="fedavg",
        help="fl algorithms: fedavg/fedprox/scaffold/fednova/moon",
    )
    parser.add_argument(
        "--use_projection_head",
        type=bool,
        default=False,
        help="whether add an additional header to model or not (see MOON)",
    )
    parser.add_argument(
        "--out_dim",
        type=int,
        default=256,
        help="the output dimension for the projection layer",
    )
    parser.add_argument("--loss", type=str, default="contrastive", help="for moon")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="the temperature parameter for contrastive loss",
    )
    parser.add_argument(
        "--comm_round",
        type=int,
        default=50,
        help="number of maximum communication roun",
    )
    parser.add_argument(
        "--is_same_initial",
        type=int,
        default=1,
        help="Whether initial all the models with the same parameters in fedavg",
    )
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dropout_p",
        type=float,
        required=False,
        default=0.0,
        help="Dropout probability. Default=0.0",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        required=False,
        default="/home/stijani/data",
        help="Data directory",
    )
    parser.add_argument(
        "--metric_dir",
        type=str,
        required=False,
        default="./exp_metrics",
        help="Directory where train/test metric files are stored",
    )
    parser.add_argument(
        "--exp_title",
        type=str,
        required=False,
        default="testing-gradiance-with-beta-0.99",
        help="desciption of an experiment",
    )
    parser.add_argument(
        "--exp_category",
        type=str,
        required=False,
        default="hyper-parame-tunning",
        help="the category to which an experiment belong",
    )
    parser.add_argument(
        "--reg", type=float, default=1e-5, help="L2 regularization strength"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default="./logs/",
        help="Log directory path",
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        required=False,
        default="./models/",
        help="Model directory path",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="The parameter for the dirichlet distribution for data partitioning",
    )
    parser.add_argument("--beta_", type=float, default=0.99, help="fpr gradiance only")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to run the program"
    )
    parser.add_argument(
        "--log_file_name", type=str, default=None, help="The log file name"
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="the optimizer")
    parser.add_argument(
        "--mu", type=float, default=0.001, help="the mu parameter for fedprox"
    )
    parser.add_argument(
        "--noise", type=float, default=0, help="how much noise we add to some party"
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="level",
        help="Different level of noise or different space of noise",
    )
    parser.add_argument(
        "--rho", type=float, default=0, help="Parameter controlling the momentum SGD"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1,
        help="Sample ratio for each communication comm_round",
    )
    parser.add_argument(
        "--use_aug",
        type=bool,
        default=True,
        help="whether to use data augmentation",
    )
    parser.add_argument(
        "--local_data_path",
        type=str,
        help="path where local copies of datasets are loaded",
    )
    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {"mnist", "cifar10", "svhn", "fmnist"}:
        n_classes = 10
    elif args.dataset == "celeba":
        n_classes = 2
    elif args.dataset == "cifar100":
        n_classes = 100
    elif args.dataset == "tinyimagenet":
        n_classes = 200
    elif args.dataset == "femnist":
        n_classes = 62
    elif args.dataset == "emnist":
        n_classes = 47
    elif args.dataset in {"a9a", "covtype", "rcv1", "SUSY"}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model + add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == "moon":
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(
                    args.model + add, args.out_dim, n_classes, net_configs
                )
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == "covtype":
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == "a9a":
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == "rcv1":
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == "SUSY":
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16, 8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(
                            input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10
                        )
                    elif args.dataset in ("mnist", "femnist", "fmnist"):
                        net = SimpleCNNMNIST(
                            input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10
                        )
                    elif args.dataset == "celeba":
                        net = SimpleCNN(
                            input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2
                        )
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", "femnist"):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == "celeba":
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                elif args.model == "lenet":
                    net = LeNet()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for k, v in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def view_image(train_dataloader):
    for x, target in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


if __name__ == "__main__":
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = (
            "experiment_arguments-%s.json"
            % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
        )
    else:
        argument_path = args.log_file_name + ".json"
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    mkdirs(join(args.metric_dir, args.exp_category))
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = "experiment_log-%s" % (
            datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
        )
    log_path = args.log_file_name + ".log"
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.DEBUG,
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)
    logger_custom = custom_logger(
        dir_=join(args.metric_dir, args.exp_category),
        filename=f"{args.alg}_{args.log_file_name}.log",
    )
    logger_custom.info(args)
    # logger_custom.info(f"{args.log_file_name}.log")

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    # (
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     net_dataidx_map,
    #     traindata_cls_counts,
    # ) = partition_data(
    #     args.dataset,
    #     args.datadir,
    #     args.logdir,
    #     args.partition,
    #     args.n_parties,
    #     beta=args.beta,
    # )

    # n_classes = len(np.unique(y_train))

    ############################################
    net_dataidx_map = {}
    train_feature_pth = os.path.join(args.local_data_path, args.dataset, 'train_features.npy')
    train_lab_pth = os.path.join(args.local_data_path, args.dataset, 'train_labels.npy')
    test_feature_pth = os.path.join(args.local_data_path, args.dataset, 'test_features.npy')
    test_lab_pth = os.path.join(args.local_data_path, args.dataset, 'test_labels.npy')
    train_dataset_dict, test_dl_global, n_classes, all_train_y = get_data_partition(args,
                                                                                    train_feature_pth,
                                                                                    train_lab_pth,
                                                                                    test_feature_pth,
                                                                                    test_lab_pth
                                                                                    )
    # net_dataidx_map = train_dataset_dict

    # modify the key in data partition to conform with nets
    for idx, key in enumerate(train_dataset_dict.keys()):
        net_dataidx_map[idx] = train_dataset_dict[key]
        # print("############", idx)

    # traindata_cls_counts = record_net_data_stats(all_train_y, net_dataidx_map, args.logdir)
    ############################################

    # train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(
    #     args.dataset, args.datadir, args.batch_size, 32
    # )

    # print("len train_dl_global:", len(train_ds_global))
    # logger.info(f"len train_dl_global: {len(train_ds_global)}")

    # data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    # train_all_in_list = []
    # test_all_in_list = []
    # if args.noise > 0:
    #     for party_id in range(args.n_parties):
    #         dataidxs = net_dataidx_map[party_id]

    #         noise_level = args.noise
    #         if party_id == args.n_parties - 1:
    #             noise_level = 0

    #         if args.noise_type == "space":
    #             (
    #                 train_dl_local,
    #                 test_dl_local,
    #                 train_ds_local,
    #                 test_ds_local,
    #             ) = get_dataloader(
    #                 args.dataset,
    #                 args.datadir,
    #                 args.batch_size,
    #                 32,
    #                 dataidxs,
    #                 noise_level,
    #                 party_id,
    #                 args.n_parties - 1,
    #             )
    #         else:
    #             noise_level = args.noise / (args.n_parties - 1) * party_id
    #             (
    #                 train_dl_local,
    #                 test_dl_local,
    #                 train_ds_local,
    #                 test_ds_local,
    #             ) = get_dataloader(
    #                 args.dataset,
    #                 args.datadir,
    #                 args.batch_size,
    #                 32,
    #                 dataidxs,
    #                 noise_level,
    #             )
    #         train_all_in_list.append(train_ds_local)
    #         test_all_in_list.append(test_ds_local)
    #     train_all_in_ds = data.ConcatDataset(train_all_in_list)
    #     train_dl_global = data.DataLoader(
    #         dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True
    #     )
    #     test_all_in_ds = data.ConcatDataset(test_all_in_list)
    #     test_dl_global = data.DataLoader(
    #         dataset=test_all_in_ds, batch_size=32, shuffle=False
    #     )

    if args.alg == "fedavg":
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(
            args.net_config, args.dropout_p, args.n_parties, args
        )
        global_models, global_model_meta_data, global_layer_type = init_nets(
            args.net_config, 0, 1, args
        )
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        test_acc_per_round, test_loss_per_round = [], []

        for comm_round in range(args.comm_round):
            start_time = time.time()
            logger.info("in comm comm_round:" + str(comm_round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[: int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if comm_round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            global_train_net(
                nets,
                selected,
                args,
                net_dataidx_map,
                test_dl=test_dl_global,
                device=device,
            )
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [
                len(net_dataidx_map[r]) / total_data_points for r in selected
            ]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            # logger.info("global n_training: %d" % len(train_dl_global))
            # logger.info("global n_test: %d" % len(test_dl_global))

            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(
            #     global_model, train_dl_global, device=device
            # )
            test_acc, conf_matrix, test_loss = compute_accuracy(
                global_model, test_dl_global, get_confusion_matrix=True, device=device
            )

            # train_acc_per_round.append(train_acc)
            test_acc_per_round.append(test_acc)
            # train_loss_per_round.append(train_loss)
            test_loss_per_round.append(test_loss)

            # logger.info(">> Global Model Train accuracy: %f" % train_acc)
            logger.info(">> Global Model Test accuracy: %f" % test_acc)

            elapsed_time = round(time.time() - start_time)

            logger_custom.info(
                f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}"
            )
            print(f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}")

        # save metrics to file
        save_metrics(args, test_acc_per_round, filename="test_acc.csv")
        save_metrics(args, test_loss_per_round, filename="test_loss.csv")
        # save_metrics(args, train_acc_per_round, filename="train_acc.csv")
        # save_metrics(args, train_loss_per_round, filename="train_loss.csv")

    elif args.alg == "gradiance":
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(
            args.net_config, args.dropout_p, args.n_parties, args
        )
        global_models, global_model_meta_data, global_layer_type = init_nets(
            args.net_config, 0, 1, args
        )
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        test_acc_per_round, test_loss_per_round = [], []
        for comm_round in range(args.comm_round):
            start_time = time.time()
            logger.info("in comm comm_round:" + str(comm_round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[: int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if comm_round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
                # set the aggregate unbiased grads to None since it's yet to be computed
                aggregated_unbiased_grads = None

            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            trained_state_dicts, all_clients_unbiased_step_grads = global_train_net_gradiance(
                nets,
                selected,
                global_model,
                args,
                net_dataidx_map,
                test_dl=test_dl_global,
                # test_dl=test,
                device=device,
                aggregated_unbiased_grads=aggregated_unbiased_grads,
                logger=logger,
            )
            # aggregate and update the unbiased grads
            aggregated_unbiased_grads = get_avg_of_unbiased_grads(
                all_clients_unbiased_step_grads
            )

            ### global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [
                len(net_dataidx_map[r]) / total_data_points for r in selected
            ]

            for idx in range(len(selected)):
                # net_para = nets[selected[idx]].cpu().state_dict()
                # net_para = nets[selected[idx]].state_dict()
                net_para = trained_state_dicts[selected[idx]]
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            # logger.info("global n_training: %d" % len(train_dl_global))
            # logger.info("global n_test: %d" % len(test_dl_global))

            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, test_loss = compute_accuracy(
                global_model, test_dl_global, get_confusion_matrix=True, device=device
            )

            # train_acc_per_round.append(train_acc)
            test_acc_per_round.append(test_acc)
            # train_loss_per_round.append(train_loss)
            test_loss_per_round.append(test_loss)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info(">> Global Model Test accuracy: %f" % test_acc)

            elapsed_time = round(time.time() - start_time)

            # logger_custom.info(f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss} | Train Acc: {train_acc} | Train Loss: {train_loss}")
            logger_custom.info(
                f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}"
            )
            print(f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}")
        # save metrics to file
        save_metrics(args, test_acc_per_round, filename="test_acc.csv")
        save_metrics(args, test_loss_per_round, filename="test_loss.csv")
        # save_metrics(args, train_acc_per_round, filename="train_acc.csv")
        # save_metrics(args, train_loss_per_round, filename="train_loss.csv")

    elif args.alg == "fedprox":
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(
            args.net_config, args.dropout_p, args.n_parties, args
        )
        global_models, global_model_meta_data, global_layer_type = init_nets(
            args.net_config, 0, 1, args
        )
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        test_acc_per_round, test_loss_per_round = [], []

        for comm_round in range(args.comm_round):
            start_time = time.time()
            logger.info("in comm comm_round:" + str(comm_round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[: int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if comm_round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            global_train_net_fedprox(
                nets,
                selected,
                global_model,
                args,
                net_dataidx_map,
                test_dl=test_dl_global,
                device=device,
            )
            ### global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [
                len(net_dataidx_map[r]) / total_data_points for r in selected
            ]

            for idx in range(len(selected)):
                # net_para = nets[selected[idx]].cpu().state_dict()
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            # logger.info("global n_training: %d" % len(train_dl_global))
            # logger.info("global n_test: %d" % len(test_dl_global))

            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(
            #     global_model, train_dl_global, device=device
            # )
            test_acc, conf_matrix, test_loss = compute_accuracy(
                global_model, test_dl_global, get_confusion_matrix=True, device=device
            )

            # train_acc_per_round.append(train_acc)
            test_acc_per_round.append(test_acc)
            # train_loss_per_round.append(train_loss)
            test_loss_per_round.append(test_loss)

            # logger.info(">> Global Model Train accuracy: %f" % train_acc)
            logger.info(">> Global Model Test accuracy: %f" % test_acc)

            elapsed_time = round(time.time() - start_time)

            logger_custom.info(
                f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}"
            )
            print(f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}")

        # save metrics to file
        save_metrics(args, test_acc_per_round, filename="test_acc.csv")
        save_metrics(args, test_loss_per_round, filename="test_loss.csv")
        # save_metrics(args, train_acc_per_round, filename="train_acc.csv")
        # save_metrics(args, train_loss_per_round, filename="train_loss.csv")

    elif args.alg == "scaffold":
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(
            args.net_config, args.dropout_p, args.n_parties, args
        )
        global_models, global_model_meta_data, global_layer_type = init_nets(
            args.net_config, 0, 1, args
        )
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        test_acc_per_round, test_loss_per_round = [], []

        for comm_round in range(args.comm_round):
            start_time = time.time()
            logger.info("in comm comm_round:" + str(comm_round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[: int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if comm_round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            global_train_net_scaffold(
                nets,
                selected,
                global_model,
                c_nets,
                c_global,
                args,
                net_dataidx_map,
                test_dl=test_dl_global,
                device=device,
            )
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [
                len(net_dataidx_map[r]) / total_data_points for r in selected
            ]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            # logger.info("global n_training: %d" % len(train_dl_global))
            # logger.info("global n_test: %d" % len(test_dl_global))

            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(
            #     global_model, train_dl_global, device=device
            # )
            test_acc, conf_matrix, test_loss = compute_accuracy(
                global_model, test_dl_global, get_confusion_matrix=True, device=device
            )

            # train_acc_per_round.append(train_acc)
            test_acc_per_round.append(test_acc)
            # train_loss_per_round.append(train_loss)
            test_loss_per_round.append(test_loss)

            # logger.info(">> Global Model Train accuracy: %f" % train_acc)
            logger.info(">> Global Model Test accuracy: %f" % test_acc)

            elapsed_time = round(time.time() - start_time)

            logger_custom.info(
                f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}"
            )
            print(f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}")

        # save metrics to file
        save_metrics(args, test_acc_per_round, filename="test_acc.csv")
        save_metrics(args, test_loss_per_round, filename="test_loss.csv")
        # save_metrics(args, train_acc_per_round, filename="train_acc.csv")
        # save_metrics(args, train_loss_per_round, filename="train_loss.csv")

    elif args.alg == "fednova":
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(
            args.net_config, args.dropout_p, args.n_parties, args
        )
        global_models, global_model_meta_data, global_layer_type = init_nets(
            args.net_config, 0, 1, args
        )
        global_model = global_models[0]

        d_list = [
            copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)
        ]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        # data_sum = 0
        # for i in range(args.n_parties):
        #     data_sum += len(traindata_cls_counts[i])
        # portion = []
        # for i in range(args.n_parties):
        #     portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        test_acc_per_round, test_loss_per_round = [], []

        for comm_round in range(args.comm_round):
            start_time = time.time()
            logger.info("in comm comm_round:" + str(comm_round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[: int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if comm_round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = global_train_net_fednova(
                nets,
                selected,
                global_model,
                args,
                net_dataidx_map,
                test_dl=test_dl_global,
                device=device,
            )
            total_n = sum(n_list)
            # print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    # if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    # else:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n

            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i] / total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                # print(updated_model[key])
                if updated_model[key].type() == "torch.LongTensor":
                    updated_model[key] -= (coeff * d_total_round[key]).type(
                        torch.LongTensor
                    )
                elif updated_model[key].type() == "torch.cuda.LongTensor":
                    updated_model[key] -= (coeff * d_total_round[key]).type(
                        torch.cuda.LongTensor
                    )
                else:
                    # print(updated_model[key].type())
                    # print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)

            # logger.info("global n_training: %d" % len(train_dl_global))
            # logger.info("global n_test: %d" % len(test_dl_global))

            # global_model.to(device)
            # train_acc, train_loss = compute_accuracy(
            #     global_model, train_dl_global, device=device
            # )
            test_acc, conf_matrix, test_loss = compute_accuracy(
                global_model, test_dl_global, get_confusion_matrix=True, device=device
            )

            # train_acc_per_round.append(train_acc)
            test_acc_per_round.append(test_acc)
            # train_loss_per_round.append(train_loss)
            test_loss_per_round.append(test_loss)

            # logger.info(">> Global Model Train accuracy: %f" % train_acc)
            logger.info(">> Global Model Test accuracy: %f" % test_acc)

            elapsed_time = round(time.time() - start_time)

            logger_custom.info(
                f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}"
            )
            print(f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}")

        # save metrics to file
        save_metrics(args, test_acc_per_round, filename="test_acc.csv")
        save_metrics(args, test_loss_per_round, filename="test_loss.csv")
        # save_metrics(args, train_acc_per_round, filename="train_acc.csv")
        # save_metrics(args, train_loss_per_round, filename="train_loss.csv")

    elif args.alg == "moon":
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(
            args.net_config, args.dropout_p, args.n_parties, args
        )
        global_models, global_model_meta_data, global_layer_type = init_nets(
            args.net_config, 0, 1, args
        )
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        test_acc_per_round, test_loss_per_round = [], []

        for comm_round in range(args.comm_round):
            start_time = time.time()
            logger.info("in comm comm_round:" + str(comm_round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[: int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if comm_round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            global_train_net_moon(
                nets,
                selected,
                args,
                net_dataidx_map,
                test_dl=test_dl_global,
                global_model=global_model,
                prev_model_pool=old_nets_pool,
                comm_round=comm_round,
                device=device,
            )
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [
                len(net_dataidx_map[r]) / total_data_points for r in selected
            ]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            # logger.info("global n_training: %d" % len(train_dl_global))
            # logger.info("global n_test: %d" % len(test_dl_global))

            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(
            #     global_model, train_dl_global, moon_model=True, device=device
            # )
            test_acc, conf_matrix, test_loss = compute_accuracy(
                global_model,
                test_dl_global,
                get_confusion_matrix=True,
                moon_model=True,
                device=device,
            )

            # train_acc_per_round.append(train_acc)
            test_acc_per_round.append(test_acc)
            # train_loss_per_round.append(train_loss)
            test_loss_per_round.append(test_loss)

            # logger.info(">> Global Model Train accuracy: %f" % train_acc)
            logger.info(">> Global Model Test accuracy: %f" % test_acc)

            elapsed_time = round(time.time() - start_time)

            logger_custom.info(
                f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}"
            )
            print(f"Round: {comm_round} | Elapse Time: {elapsed_time} | Test Acc: {test_acc} | Test Loss: {test_loss}")

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets

        # save metrics to file
        save_metrics(args, test_acc_per_round, filename="test_acc.csv")
        save_metrics(args, test_loss_per_round, filename="test_loss.csv")
        # save_metrics(args, train_acc_per_round, filename="train_acc.csv")
        # save_metrics(args, train_loss_per_round, filename="train_loss.csv")

    elif args.alg == "local_training":
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(
            args.net_config, args.dropout_p, args.n_parties, args
        )
        arr = np.arange(args.n_parties)
        local_train_net(
            nets, arr, args, net_dataidx_map, test_dl=test_dl_global, device=device
        )

    elif args.alg == "all_in":
        nets, local_model_meta_data, layer_type = init_nets(
            args.net_config, args.dropout_p, 1, args
        )
        n_epoch = args.epochs
        nets[0].to(device)
        # trainacc, testacc = local_train_net(
        local_train_net(
            args,
            0,
            nets[0],
            train_dl_global,
            test_dl_global,
            n_epoch,
            args.lr,
            args.optimizer,
            device=device,
        )

        # logger.info("All in test acc: %f" % testacc)
