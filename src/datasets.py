"""
datasets.py

@author mmosse19
@version July 2019
"""
import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from generate import generate_worlds

# Note: there are about 11,972 pnts in train_mnist that are 3s/4s

# split on all data: test/(train + validate)
TEST_PORTION = .2
TRAINVAL_PORTION = 1 - TEST_PORTION

# split on train data: train/validate
VAL_PORTION = .2
TRAIN_PORTION = 1.0 - VAL_PORTION

# number of square images to generate
GAN_DATA_LEN = 8000
LOG_REG_DATA_LEN = 1200

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")
CAUSAL_MNIST_DIR = os.path.join(DATA_DIR, 'causal_mnist')

class CausalMNIST(Dataset):
	def __init__(self, split, mnist, using_gan, root=CAUSAL_MNIST_DIR,
		channels=1, classes=None, target_trials_only=False, cf=False, transform=True):
		super(CausalMNIST, self).__init__()
		self.root = root
		self.mnist = mnist
		self.split = split
		self.img_transform = transforms.ToTensor()

		causal_data_len = GAN_DATA_LEN if using_gan else LOG_REG_DATA_LEN

		"""
		start, end = 0, int(mnist_data_len*(TRAIN_PORTION+VAL_PORTION))
		self.length = int(causal_data_len*(VAL_PORTION + TRAIN_PORTION))
		"""
		# if not test, then get length, to split train dataset into train/val
		if (split != "test"):
			mnist_data_len = len(self.mnist["labels"])
			assert(mnist_data_len == len(self.mnist["digits"]))

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

		print("generating {} worlds...".format(split))
		scenarios = generate_worlds(self.mnist, n=self.length, cf=cf, transform=transform)
		self.imgs = [self.img_transform(Image.fromarray(scenarios[i][0])) for i in range(self.length)]
		print("generated {} worlds.".format(split))
		
		# TODO: consider not restricting labels like this
		self.labels = ["causal" in pt[1] for pt in scenarios]

	def __getitem__(self, index):
		return self.imgs[index], self.labels[index]

	def __len__(self):
		return self.length