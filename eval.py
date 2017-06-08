#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
import randopt as ro

import torch as th
from torch.autograd import Variable

from tqdm import tqdm
from experiments import problems
from utils import parse_args

args = parse_args()

def test(model, data, loss):
    model.eval()
    error = 0.0
    accuracy = 0.0
    for X, y in data:
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y, volatile=True)
        pred = model(X)
        error += loss(pred, y).data[0]
    return error / len(data), accuracy

if __name__ == '__main__':
    model, (train_set, test_set), loss, opt, num_epochs = problems[args.task](args)
    error = test(model, test_set, loss)
    print('The local validation error is: ', error)
