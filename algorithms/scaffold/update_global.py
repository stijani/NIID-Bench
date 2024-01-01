import sys
from math import *
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from algorithms.scaffold.update_local import local_train_net_scaffold
sys.path.append('/home/stijani/projects/phd/paper-2/phd-paper2-code/modules')
from custom_image_dataset import CustomImageDataset


def global_train_net_scaffold(
    nets,
    selected,
    global_model,
    c_nets,
    c_global,
    args,
    net_dataidx_map,
    test_dl=None,
    device="cpu",
):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
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

        c_nets[net_id].to(device)

        train_dl_local = DataLoader(CustomImageDataset(net_dataidx_map[net_id], args.dataset, args.use_aug), batch_size=args.batch_size, shuffle=True)
        c_delta_para = local_train_net_scaffold(
            args,
            net_id,
            net,
            global_model,
            c_nets[net_id],
            c_global,
            train_dl_local,
            test_dl,
            args.num_local_steps,
            args.lr,
            args.optimizer,
            device=device,
        )

        ### c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        # logger.info("net %d final test acc %f" % (net_id, testacc))
        # avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == "torch.LongTensor":
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == "torch.cuda.LongTensor":
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            # print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    # avg_acc /= len(selected)
    # if args.alg == "local_training":
    #     logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list