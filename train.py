#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
import randopt as ro
from tqdm import tqdm

import torch as th
from torch.autograd import Variable

from experiments import problems
from utils import parse_args, save_checkpoint, reset_parameters
from eval import test


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, data, loss, opt, args):
    model.train()
    total_error = 0.0
    print_inter = len(data) > 10000

    for i, (X, y) in tqdm(enumerate(data), total=len(data), leave=False):
        if args.cuda:
            X = X.cuda()
            y = y.cuda(async=True)
        X, y = Variable(X), Variable(y)
        output = model(X)
        error = loss(output, y)
        opt.zero_grad()
        error.backward()
        if args.clip_grad > 0.0:
            th.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
        opt.step()
        total_error += error.data[0]
        if print_inter and (i + 1) % 100 == 0:
            print('Intermediate loss: ', total_error/i)
    return total_error / len(data)

if __name__ == '__main__':

    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducible experiments
    RNG_SEED = 1234
    np.random.seed(RNG_SEED)
    th.manual_seed(RNG_SEED)
    if args.cuda:
        th.cuda.manual_seed(RNG_SEED)
    
    model, (train_set, test_set), loss, opt, num_epochs = problems[args.task](args)

    exp_name = args.task
    exp = ro.Experiment(args.task, {})

    train_errors = []
    test_errors = []

    for epoch in xrange(num_epochs):
        adjust_learning_rate(opt, epoch, args)
        error = train(model, train_set, loss, opt, args)
        train_errors.append(error)
        print('-' * 20, ' ', args.task, ' Epoch ', epoch, ' ', '-' * 20)
        print('Train error: ', error)
        error = test(model, test_set, loss)
        test_errors.append(error)
        print('Test error: ', error)
        print('/n')

        save_checkpoint({
                'model': model.state_dict(),
                'epoch': epoch,
                'exp': exp_name,
                'train_errors': train_errors,
                'test_errors': test_errors,
                }, pre=args.task + '_')

    info = {
        'train_errors': train_errors,
        'test_errors': test_errors,
    }
    info.update(dict(args.keyvalues))
    exp.add_result(test_errors[-1], info)

    # Save the model weights
    path = os.path.abspath(os.path.curdir)
    path = os.path.join(path, 'trained_models')
    path = os.path.join(path, args.task + '_' + args.save + '.pth.tar')
    th.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'exp': exp_name,
            'train_errors': train_errors,
            'test_errors': test_errors,
            }, path)
