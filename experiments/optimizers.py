#!/usr/bin/env python

import torch as th
import torch.optim

def get_optimizer(args, params):
    if 'sgd' in args.opt:
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
