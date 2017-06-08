#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
import randopt as ro

import torch as th
from torch.autograd import Variable

from tqdm import tqdm
from experiments.problems  problems

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
