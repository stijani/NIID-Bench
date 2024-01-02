import sys
from math import *
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from algorithms.fedavg.update_local import local_train_net
sys.path.append('/home/stijani/projects/phd/paper-2/phd-paper2-code/modules')
from custom_image_dataset import CustomImageDataset


def global_train_net(net, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0
    trained_state_dicts = {}

    for client_id in selected:
        client_net = copy.deepcopy(net)
        # if client_id not in selected:
        #     continue
        dataidxs = net_dataidx_map[client_id]

        logger.info("Training network %s. n_training: %d" % (str(client_id), len(dataidxs)))
        # move the model to cuda device:
        client_net.to(device)
        train_dl_local = DataLoader(CustomImageDataset(net_dataidx_map[client_id], args.dataset, args.use_aug), batch_size=args.batch_size, shuffle=True)
        local_train_net(args, client_id, client_net, train_dl_local, test_dl, args.num_local_steps, args.lr, args.optimizer, device=device)
        trained_state_dicts[client_id] = copy.deepcopy(client_net.state_dict())
        del client_net
    return trained_state_dicts
