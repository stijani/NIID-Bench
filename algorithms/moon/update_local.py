import torch.optim as optim
import torch.nn as nn
from math import *
from itertools import cycle
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *


def local_train_net_moon(
    net_id,
    net,
    global_net,
    previous_nets,
    train_dataloader,
    test_dataloader,
    num_local_steps,
    lr,
    args_optimizer,
    mu,
    temperature,
    args,
    comm_round,
    device="cpu",
):
    logger.info("Training network %s" % str(net_id))

    # train_acc, train_loss = compute_accuracy(
    #     net, train_dataloader, moon_model=True, device=device
    # )
    # test_acc, conf_matrix, test_loss = compute_accuracy(
    #     net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device
    # )

    # logger.info(">> Pre-Training Training accuracy: {}".format(train_acc))
    # logger.info(">> Pre-Training Test accuracy: {}".format(test_acc))

    # conloss = ContrastiveLoss(temperature)

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
            momentum=0.9,
            weight_decay=args.reg,
        )

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    if args.loss != "l2norm":
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()
    # oppsi_nets = copy.deepcopy(previous_nets)
    # for net_id, oppsi_net in enumerate(oppsi_nets):
    #     oppsi_w = oppsi_net.state_dict()
    #     prev_w = previous_nets[net_id].state_dict()
    #     for key in oppsi_w:
    #         oppsi_w[key] = 2*global_w[key] - prev_w[key]
    #     oppsi_nets.load_state_dict(oppsi_w)
    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    # for epoch in range(epochs):
    sum_local_loss = 0.0
    sum_local_loss1 = 0.0
    sum_local_loss2 = 0.0
    train_dataloader_ = cycle(train_dataloader)
    for step in range(num_local_steps):
        x, target = next(train_dataloader_)
        x, target = x.to(device), target.to(device)

        optimizer.zero_grad()
        x.requires_grad = True
        target.requires_grad = False
        target = target.long()

        _, pro1, out = net(x)
        _, pro2, _ = global_net(x)
        if args.loss == "l2norm":
            loss2 = mu * torch.mean(torch.norm(pro2 - pro1, dim=1))

        elif args.loss == "only_contrastive" or args.loss == "contrastive":
            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            for previous_net in previous_nets:
                previous_net.to(device)
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                # previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).to(device).long()

            # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

            loss2 = mu * criterion(logits, labels)

        if args.loss == "only_contrastive":
            loss = loss2
        else:
            loss1 = criterion(out, target)
            loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        cnt += 1
        sum_local_loss += loss.item()
        sum_local_loss1 += loss1.item()
        sum_local_loss2 += loss2.item()

    local_loss = round(sum_local_loss / num_local_steps, 2)
    local_loss1 = round(sum_local_loss1 / num_local_steps, 2)
    local_loss2 = round(sum_local_loss2 / num_local_steps, 2)
    logger.info(
        f"Client: {net_id} | Averaged Local Loss: Loss: {local_loss} Loss1: {local_loss1} Loss2: {local_loss2}"
    )

    if args.loss != "l2norm":
        for previous_net in previous_nets:
            ### previous_net.to('cpu')
            pass
    # train_acc, train_loss = compute_accuracy(
    #     net, train_dataloader, moon_model=True, device=device
    # )
    # test_acc, conf_matrix, test_loss = compute_accuracy(
    #     net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device
    # )

    # logger.info(">> Training accuracy: %f" % train_acc)
    # logger.info(">> Test accuracy: %f" % test_acc)
    ### net.to('cpu')
    logger.info(" ** Training complete **")
    # return train_acc, test_acc