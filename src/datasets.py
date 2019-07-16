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

TEST_PORTION = .2
TRAIN_PORTION = .65
VAL_PORTION = 1.0 - TEST_PORTION - TEST_PORTION

# number of square images to generate
GAN_DATA_SIZE = 7000
LOG_REG_DATA_SIZE = 1200

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")
CAUSAL_MNIST_DIR = os.path.join(DATA_DIR, 'causal_mnist')

class CausalMNIST(Dataset):
	def __init__(self, split, mnist, gan_size, root=CAUSAL_MNIST_DIR,
		channels=1, classes=None, target_trials_only=False, cf=False, transform=True):
		super(CausalMNIST, self).__init__()
		self.root = root
		self.mnist = mnist
		self.split = split
		self.img_transform = transforms.ToTensor()

		total_data_size = GAN_DATA_SIZE if gan_size else LOG_REG_DATA_SIZE
		data_len = len(self.mnist["labels"])
		assert(data_len == len(self.mnist["digits"]))

		if (split == "train"):
			start, end = 0, int(data_len*TRAIN_PORTION)
			self.length = int(LOG_REG_DATA_SIZE*TRAIN_PORTION)

		elif (split == "validate"):
			start, end = int(data_len*TRAIN_PORTION), int(data_len*(TRAIN_PORTION+VAL_PORTION))
			self.length = int(LOG_REG_DATA_SIZE*TEST_PORTION)

		elif (split == "test"):
			start, end = int(data_len*(TRAIN_PORTION+VAL_PORTION)), data_len
			self.length = int(LOG_REG_DATA_SIZE*TEST_PORTION)
		else:
			raise RuntimeError("CausalMNIST was expecting split to be 'train', 'validate', or 'test'.")

		self.mnist["labels"] = self.mnist["labels"][start:end]
		self.mnist["digits"] = self.mnist["digits"][start:end]

		breakpoint()
		print("generating {} worlds...".format(split))
		scenarios = generate_worlds(self.mnist, n=self.length, cf=cf, transform=transform)
		self.imgs = [self.img_transform(Image.fromarray(scenarios[i][0])) for i in range(self.length)]
		print("generated {} worlds.".format(split))
		
		# TODO: consider not restricting labels like this
		self.labels = ["causal" in pt[1] for pt in scenarios]
		breakpoint()

	def __getitem__(self, index):
		return self.imgs[index], self.labels[index]

	def __len__(self):
		return self.length