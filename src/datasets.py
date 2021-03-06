"""
datasets.py

@author mmosse19
@version July 2019
"""
import os
import json
import copy
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from generate import (generate_worlds, NUM1, NUM2)
from utils import data_file_name

# Note: there are about 11,972 pnts in train_mnist that are 3s/4s

# split on all data: test/(train + validate)
TEST_PORTION = .2
TRAINVAL_PORTION = 1 - TEST_PORTION

# split on train data: train/validate
VAL_PORTION = .2
TRAIN_PORTION = 1.0 - VAL_PORTION

# number of square images to generate
DATA_LEN = 11000

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")
CAUSAL_MNIST_DIR = os.path.join(DATA_DIR, 'causal_mnist')

class CausalMNIST(Dataset):
    def __init__(self, split, mnist, using_gan, root=CAUSAL_MNIST_DIR, transform=True, train_on_mnist=False):
        super(CausalMNIST, self).__init__()
        cf = True

        self.root = root
        self.mnist = copy.deepcopy(mnist)
        self.split = split
        self.train_on_mnist = train_on_mnist
        self.img_transform = transforms.ToTensor()

        # if not test, then get length, to split train dataset into train/val
        if (split != "test"):
            mnist_data_len = len(self.mnist["labels"])
            assert(mnist_data_len == len(self.mnist["digits"]))

        # set self.length, self.mnist
        if (split == "train"):
            self.length = int(DATA_LEN*TRAINVAL_PORTION*TRAIN_PORTION)

            start, end = 0, int(mnist_data_len*TRAIN_PORTION)
            self.mnist["labels"] = self.mnist["labels"][start:end]
            self.mnist["digits"] = self.mnist["digits"][start:end]

        elif (split == "validate"):
            self.length = int(DATA_LEN*TRAINVAL_PORTION*VAL_PORTION)

            start, end = int(mnist_data_len*TRAIN_PORTION), mnist_data_len
            self.mnist["labels"] = self.mnist["labels"][start:end]
            self.mnist["digits"] = self.mnist["digits"][start:end]
        elif (split == "test"):
            self.length = int(DATA_LEN*TEST_PORTION)
        else:
            raise RuntimeError("CausalMNIST was expecting split to be 'train', 'validate', or 'test'.")

        # retrieve square images, using self.mnist and self.length
        flags = "{}{}".format("-cf" if cf else "", "-tr" if transform else "")
        file_name = data_file_name(prefix = "scenario", suffix = split + flags)
        if (os.path.isfile(file_name)):
            scenarios = np.load(file_name, allow_pickle=True)
            print("retrieved {} worlds from file.".format(split))
        else:
            scenarios = generate_worlds(self.mnist, n=self.length, cf=cf, transform=transform)
            np.save(file_name, scenarios)
            print("generated {} worlds.".format(split))

        self.imgs = [self.img_transform(Image.fromarray(pt[0])) for pt in scenarios]
        self.labels = [pt[1] for pt in scenarios]
        self.label_kinds = set(self.labels)
        # note: these are for the labels used by the sklearn classifier, no the labels used for log reg
        self.label_conds = [[str(NUM1), str(NUM2)], [str(NUM1)], [str(NUM2)], []] 
        self.label_nums = np.arange(0.0, float(len(self.label_conds)), 1.0)

    def __getitem__(self, index):
        if (self.train_on_mnist):
            return self.mnist["digits"][index][np.newaxis,...], self.mnist["labels"][index], None
        else:
            return self.imgs[index], self.labels[index], "causal" in self.labels[index]

    def __len__(self):
        if (self.train_on_mnist):
            return len(self.mnist["digits"])
        else:
            return self.length

    def label_to_num(self, label):
        for i in range(len(self.label_conds)):
            cond = self.label_conds[i]
            if (all(s in label for s in cond)):
                return self.label_nums[i]

    def np_train_data(self):
        x = np.asarray([np.asarray(img) for img in self.imgs])
        y = np.asarray([self.label_to_num(label) for label in self.labels])
        return x, y

