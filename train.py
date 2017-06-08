#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
import randopt as ro

import torch as th
from torch.autograd import Variable

from tqdm import tqdm
from experiments.problems import problems

parser = argparse.ArgumentParser(
    description='Distriuted Optimization Experiment')
parser.add_argument(
    '--task', type=str, default='mnist', help='Task to train on.')
parser.add_argument(
    '--opt', type=str, default='sgd', help='Optimizer')
parser.add_argument(
    '--bsz', type=int, default=64, help='Batch size')
parser.add_argument(
    '--epochs', type=int, default=10, help='Batch size')
parser.add_argument(
    '--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument(
    '--momentum', type=float, default=0.9, help='Momentum constant')
parser.add_argument(
    '--lr_decay', type=int, default=100, help='Learning rate decay')
parser.add_argument(
    '--clip_grad', type=float, default=0.0, help='Gradient norm clipping')
parser.add_argument(
    '--no-cuda', action='store_true', default=False, help='Train on GPU')
args = parser.parse_args()
args.cuda = not args.no_cuda

# Set random seed for reproducible experiments
RNG_SEED = 1234
th.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
if args.cuda:
    if size == 2:
        th.cuda.set_device(2*rank)
    else:
        th.cuda.set_device((rank) % th.cuda.device_count())
    th.cuda.manual_seed(RNG_SEED)


def save_checkpoint(checkpoint, pre=''):
    filename = pre + 'checkpoint.pth.tar'
    th.save(checkpoint, filename)


def train(model, data, loss, opt):
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
    
    model, (train_set, test_set), loss, opt, num_epochs = problems[args.task](args)

    exp_name = args.task + '_' + args.pre + '_' + str(size) + 'replicas'
    exp = ro.Experiment(args.task + '_' + args.pre + '_' + str(size) + 'replicas', {})

    for epoch in xrange(num_epochs):
        adjust_learning_rate(opt, epoch)
        error = train(model, train_set, loss, opt)
        train_errors.append(error)
        print('-' * 20, ' ', opt.name, ' Epoch ', epoch, ' ', '-' * 20)
        print('Train error: ', error)
        error = test(model, test_set, loss)
        test_errors.append(error)
        print('Test error: ', error)
        print('/n')

        save_checkpoint({
                'model': model.state_dict(),
                'epoch': epoch,
                'exp': exp_name,
                'opt': opt.name,
                'train_errors': train_errors,
                'test_errors': test_errors,
                'test_acc': test_acc,
                }, pre=args.task + '_')

    info = {
        'train_errors': train_errors,
        'test_errors': test_errors,
        'test_acc': test_acc,
    }
    info.update(args)
    exp.add_result(test_errors[-1], info)
