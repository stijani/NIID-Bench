import torch.optim as optim
import torch.nn as nn
from math import *
from itertools import cycle
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *


def local_train_net(args, net_id, net, train_dataloader, test_dataloader, num_local_steps, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # train_acc, train_loss = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix, test_loss = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    sum_local_loss = 0.0
    train_dataloader_ = cycle(train_dataloader[0])
    for  step in range(num_local_steps):
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

        sum_local_loss += loss.item()

    logger.info(f'Client: {net_id} | Averaged Local Loss: {round(sum_local_loss/num_local_steps, 2)}')

    # train_acc, train_loss = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix, test_loss = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Training accuracy: %f' % train_acc)
    # logger.info('>> Test accuracy: %f' % test_acc)

    ### net.to('cpu')
    logger.info(' ** Training complete **')
    #return train_acc, test_acc