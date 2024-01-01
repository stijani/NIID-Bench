import sys
from math import *
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from algorithms.fedavg.update_local import local_train_net
sys.path.append('/home/stijani/projects/phd/paper-2/phd-paper2-code/modules')
from custom_image_dataset import CustomImageDataset


def global_train_net(nets, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)
        train_dl_local = DataLoader(CustomImageDataset(net_dataidx_map[net_id], args.dataset, args.use_aug), batch_size=args.batch_size, shuffle=True)
        local_train_net(args, net_id, net, train_dl_local, test_dl, args.num_local_steps, args.lr, args.optimizer, device=device)

    nets_list = list(nets.values())
    return nets_list