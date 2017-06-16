#!/usr/bin/env python

import os
import torch as th
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import numpy as np
from numpy.random import uniform, normal
from random import Random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RandomRotate(object):
    def __init__(self, angle, expand=0):
        angle = abs(angle)
        if isinstance(angle, int):
            self.min_angle = -angle
            self.max_angle = angle
        else:
            self.min_angle = angle[0]
            self.max_angle = angle[1]
        self.expand = expand

    def __call__(self, img):
        angle = uniform(self.min_angle, self.max_angle)
        return img.rotate(angle, expand=self.expand)


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]


    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class TestImageFolder(ImageFolder):

    """
    Root image folder must have images inside subfolders.
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        path = os.path.basename(path)
        return img, path
