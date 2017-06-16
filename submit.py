#!/usr/bin/env python

import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn

from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34
import torch.nn.functional as F

import os

from experiments import problems, tests
from utils import parse_args

from experiments.datasets import TestImageFolder


def create_submission(model, data, submission_name="test_submit.csv"):

    fileNamesTotal = []
    predsTotal = []

    for (input, fileNames) in data:
        input = V(input)
        preds = model(input)

        predsTotal.append(preds[:, 0:3])
        fileNamesTotal += fileNames

    predsTotal = th.cat(predsTotal)
    predsTotal = predsTotal.data.numpy()
    names = np.array(fileNamesTotal).reshape(len(fileNamesTotal), 1)
    t1p = predsTotal[:, 0].reshape(len(fileNamesTotal), 1)
    t2p = predsTotal[:, 1].reshape(len(fileNamesTotal), 1)
    t3p = predsTotal[:, 2].reshape(len(fileNamesTotal), 1)
    submission = pd.DataFrame(data=np.concatenate((names, t1p, t2p, t3p), axis=1), columns=[
                              'image_name', 'Type_1', 'Type_2', 'Type_3'])
    submission.to_csv(submission_name, index=False)


if __name__ == "__main__":
    args = parse_args()
    loader = tests[args.task](args)

    model, (train_set, test_set), loss, opt, num_epochs = problems[args.task](args)

    # Load checkpoint
    path = os.path.abspath(os.path.curdir)
    path = os.path.join(path, 'trained_models')
    path = os.path.join(path, args.weights)
    checkpoint = th.load(path)
    model.load_state_dict(checkpoint['model'])
    model.test()
    create_submission(model, loader)
