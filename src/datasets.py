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

from generate import generate_worlds
from utils import data_file_name

# Note: there are about 11,972 pnts in train_mnist that are 3s/4s

# split on all data: test/(train + validate)
TEST_PORTION = .2
TRAINVAL_PORTION = 1 - TEST_PORTION

# split on train data: train/validate
VAL_PORTION = .2
TRAIN_PORTION = 1.0 - VAL_PORTION

# number of square images to generate
GAN_DATA_LEN = 11000
LOG_REG_DATA_LEN = 1200

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")
CAUSAL_MNIST_DIR = os.path.join(DATA_DIR, 'causal_mnist')

class CausalMNIST(Dataset):
    def __init__(self, split, mnist, using_gan, root=CAUSAL_MNIST_DIR,
        cf=False, transform=True, train_on_mnist=False):
        super(CausalMNIST, self).__init__()
        self.root = root
        self.mnist = copy.deepcopy(mnist)
        self.split = split
        self.train_on_mnist = train_on_mnist
        self.img_transform = transforms.ToTensor()

        causal_data_len = GAN_DATA_LEN if using_gan else LOG_REG_DATA_LEN

        # if not test, then get length, to split train dataset into train/val
        if (split != "test"):
            mnist_data_len = len(self.mnist["labels"])
            assert(mnist_data_len == len(self.mnist["digits"]))

        # set self.length, self.mnist
        if (split == "train"):
            self.length = int(causal_data_len*TRAINVAL_PORTION*TRAIN_PORTION)

            start, end = 0, int(mnist_data_len*TRAIN_PORTION)
            self.mnist["labels"] = self.mnist["labels"][start:end]
            self.mnist["digits"] = self.mnist["digits"][start:end]

        elif (split == "validate"):
            self.length = int(causal_data_len*TRAINVAL_PORTION*VAL_PORTION)

            start, end = int(mnist_data_len*TRAIN_PORTION), mnist_data_len
            self.mnist["labels"] = self.mnist["labels"][start:end]
            self.mnist["digits"] = self.mnist["digits"][start:end]
        elif (split == "test"):
            self.length = int(causal_data_len*TEST_PORTION)
        else:
            raise RuntimeError("CausalMNIST was expecting split to be 'train', 'validate', or 'test'.")

        # retrieve square images, using self.mnist and self.length
        file_name = data_file_name(prefix = "scenario", suffix = split)
        if (os.path.isfile(file_name)):
            print("retrieving {} worlds from file...".format(split))
            scenarios = np.load(file_name)
            print("retrieved {} worlds from file.".format(split))
        else:
            print("generating {} worlds...".format(split))
            scenarios = generate_worlds(self.mnist, n=self.length, cf=cf, transform=transform)
            np.save(file_name, scenarios)
            print("generated {} worlds.".format(split))

        self.imgs = [self.img_transform(Image.fromarray(pt[0])) for pt in scenarios]
        self.labels = [pt[1] for pt in scenarios]

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
