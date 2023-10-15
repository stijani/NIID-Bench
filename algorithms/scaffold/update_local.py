import torch.optim as optim
import torch.nn as nn
from math import *
from itertools import cycle
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *


def local_train_net_scaffold(
    args,
    net_id,
    net,
    global_model,
    c_local,
    c_global,
    train_dataloader,
    test_dataloader,
    num_local_steps,
    lr,
    args_optimizer,
    device="cpu",
):
    logger.info("Training network %s" % str(net_id))

    # train_acc, train_loss = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix, test_loss = compute_accuracy(
    #     net, test_dataloader, get_confusion_matrix=True, device=device
    # )

    # logger.info(">> Pre-Training Training accuracy: {}".format(train_acc))
    # logger.info(">> Pre-Training Test accuracy: {}".format(test_acc))

    if args_optimizer == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=args.reg,
        )
    elif args_optimizer == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=args.reg,
            amsgrad=True,
        )
    elif args_optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            momentum=args.rho,
            weight_decay=args.reg,
        )
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    sum_local_loss = 0.0
    train_dataloader_ = cycle(train_dataloader[0])
    for step in range(num_local_steps):
        x, target = next(train_dataloader_)
        x, target = x.to(device), target.to(device)

        optimizer.zero_grad()
        x.requires_grad = True
        target.requires_grad = False
        target = target.long()

        out = net(x)
        loss = criterion(out, target)

        loss.backward()
        optimizer.step()

        net_para = net.state_dict()
        for key in net_para:
            net_para[key] = net_para[key] - args.lr * (
                c_global_para[key] - c_local_para[key]
            )
        net.load_state_dict(net_para)

        cnt += 1
        sum_local_loss += loss.item()

    logger.info(
        f"Client: {net_id} | Averaged Local Loss: {round(sum_local_loss/num_local_steps, 2)}"
    )

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = (
            c_new_para[key]
            - c_global_para[key]
            + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        )
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    # train_acc, train_loss = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix, test_loss = compute_accuracy(
    #     net, test_dataloader, get_confusion_matrix=True, device=device
    # )

    # logger.info(">> Training accuracy: %f" % train_acc)
    # logger.info(">> Test accuracy: %f" % test_acc)

    ### net.to('cpu')
    logger.info(" ** Training complete **")
    return c_delta_para