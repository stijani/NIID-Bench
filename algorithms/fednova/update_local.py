import torch.optim as optim
import torch.nn as nn
from math import *
from itertools import cycle
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *


def local_train_net_fednova(
    args,
    net_id,
    net,
    global_model,
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

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        momentum=args.rho,
        weight_decay=args.reg,
    )
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    tau = 0

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

        tau = tau + 1

        sum_local_loss += loss.item()
    logger.info(
        f"Client: {net_id} | Averaged Local Loss: {round(sum_local_loss/num_local_steps, 2)}"
    )

    global_model.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model.to(device)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        # norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key], a_i)
    # train_acc, train_loss = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix, test_loss = compute_accuracy(
    #     net, test_dataloader, get_confusion_matrix=True, device=device
    # )

    # logger.info(">> Training accuracy: %f" % train_acc)
    # logger.info(">> Test accuracy: %f" % test_acc)

    ### net.to('cpu')
    logger.info(" ** Training complete **")
    return a_i, norm_grad