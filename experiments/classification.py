#!/usr/bin/env pybon

import os

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms

from .datasets import DataPartitioner, RandomRotate, TestImageFolder
from .optimizers import get_optimizer

CERVIX_PATH = '/media/seba-1511/OCZ/cervical_cancer/mini/'


class Net(nn.Module):

    def __init__(self, model):
        super(Net, self).__init__()
        self.parameters = model.parameters
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x)


def get_classification(args):
    cuda = args.cuda

    model = models.resnet34()
    # model = models.alexnet()
    # model = models.resnet50()
    # model = models.resnet152()
    model = Net(model)
    model = nn.DataParallel(model)

    kwargs = {'num_workers': 10, 'pin_memory': True}
    if cuda:
        model.cuda()
    else:
        kwargs = {}

    bsz = args.bsz

    # train_dir = os.path.join(CERVIX_PATH, 'additional')
    train_dir = os.path.join(CERVIX_PATH, 'train')
    valid_dir = os.path.join(CERVIX_PATH, 'valid')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0])

    train_data = datasets.ImageFolder(train_dir,
                                      transform=transforms.Compose([
                                          transforms.RandomSizedCrop(256),
                                          RandomRotate(45),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize,
                                      ]))

    train_set = th.utils.data.DataLoader(
        train_data, batch_size=bsz, shuffle=True, **kwargs)

    valid_data = datasets.ImageFolder(valid_dir,
                                      transform=transforms.Compose([
                                          transforms.Scale(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize,
                                      ]))

    valid_set = th.utils.data.DataLoader(
        valid_data, batch_size=bsz, shuffle=False, **kwargs)

    # loss = nn.CrossEntropyLoss()
    loss = nn.NLLLoss()

    if cuda:
        loss = loss.cuda()

    opt = get_optimizer(args, model.parameters())

    num_epochs = args.epochs

    return model, (train_set, valid_set), loss, opt, num_epochs

def get_classification_test(args):
    if args.cuda:
        kwargs = {'num_workers': 10, 'pin_memory': True}
    else:
        kwargs = {}
    bsz = 512
    test_dir = os.path.join(CERVIX_PATH, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0])

    test_data = TestImageFolder(test_dir,
                                  transform=transforms.Compose([
                                      transforms.Scale(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize,
                                  ]))
    test_set = th.utils.data.DataLoader(test_data, batch_size=bsz, shuffle=False, **kwargs)
    return test_set
