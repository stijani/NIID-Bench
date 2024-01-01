import sys
from math import *
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from algorithms.fedprox.update_local import local_train_net_fedprox
sys.path.append('/home/stijani/projects/phd/paper-2/phd-paper2-code/modules')
from custom_image_dataset import CustomImageDataset


def global_train_net_fedprox(
    nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"
):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info(
            "Training network %s. n_training: %d" % (str(net_id), len(dataidxs))
        )
        net.to(device)
        train_dl_local = DataLoader(CustomImageDataset(net_dataidx_map[net_id], args.dataset, args.use_aug), batch_size=args.batch_size, shuffle=True)

        local_train_net_fedprox(
            args,
            net_id,
            net,
            global_model,
            train_dl_local,
            test_dl,
            args.num_local_steps,
            args.lr,
            args.optimizer,
            args.mu,
            device=device,
        )
    nets_list = list(nets.values())
    return nets_list

