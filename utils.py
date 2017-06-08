#!/usr/bin/env python

import argparse

def parse_args():
    args = parse_args()

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
    return args

def reset_parameters(model):
    for p in model.parameters():
        if hasattr(p, 'reset_parameters'):
            p.reset_parameters()


def save_checkpoint(checkpoint, pre=''):
    filename = pre + 'checkpoint.pth.tar'
    th.save(checkpoint, filename)


