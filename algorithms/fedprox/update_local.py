import torch.optim as optim
import torch.nn as nn
from math import *
from itertools import cycle
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *


def local_train_net_fedprox(
    args,
    net_id,
    net,
    global_net,
    train_dataloader,
    test_dataloader,
    num_local_steps,
    lr,
    args_optimizer,
    mu,
    device="cpu",
):
    logger.info("Training network %s" % str(net_id))
    logger.info("n_training: %d" % len(train_dataloader))
    logger.info("n_test: %d" % len(test_dataloader))

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

    # cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    # for epoch in range(epochs):
    # epoch_loss_collector = []
    # for batch_idx, (x, target) in enumerate(train_dataloader):
    sum_local_loss = 0.0
    train_dataloader_ = cycle(train_dataloader)
    for step in range(num_local_steps):
        x, target = next(train_dataloader_)
        x, target = x.to(device), target.to(device)

        optimizer.zero_grad()
        x.requires_grad = True
        target.requires_grad = False
        target = target.long()

        out = net(x)
        loss = criterion(out, target)

        # for fedprox
        fed_prox_reg = 0.0
        for param_index, param in enumerate(net.parameters()):
            fed_prox_reg += (mu / 2) * torch.norm(
                (param - global_weight_collector[param_index])
            ) ** 2
        loss += fed_prox_reg

        loss.backward()
        optimizer.step()

        # cnt += 1
        sum_local_loss += loss.item()

    logger.info(
        f"Client: {net_id} | Averaged Local Loss: {round(sum_local_loss/num_local_steps, 2)}"
    )

    # train_acc, train_loss = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix, test_loss = compute_accuracy(
    #     net, test_dataloader, get_confusion_matrix=True, device=device
    # )

    # logger.info(">> Training accuracy: %f" % train_acc)
    # logger.info(">> Test accuracy: %f" % test_acc)

    ### net.to('cpu')
    logger.info(" ** Training complete **")
    # return train_acc, test_acc