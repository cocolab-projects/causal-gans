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

TRAIN_SET_SZ_GAN = 5500  	# set to 5500 for GAN
VAL_SET_SZ_GAN = 1000		# set to 1000 later
TEST_SET_SZ_GAN = 1000

TRAIN_SET_SZ_LOG_REG = 1000
VAL_SET_SZ_LOG_REG = 500
TEST_SET_SZ_LOG_REG= 500

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")
CAUSAL_MNIST_DIR = os.path.join(DATA_DIR, 'causal_mnist')

class CausalMNIST(Dataset):
	def __init__(self, split, mnist, gan_size, root=CAUSAL_MNIST_DIR,
		channels=1, classes=None, target_trials_only=False, cf=False, transform=True):
		super(CausalMNIST, self).__init__()
		self.root = root
		self.mnist = mnist
		self.img_transform = transforms.ToTensor()

		if (split == "train"):
			self.length = TRAIN_SET_SZ_GAN if gan_size else TRAIN_SET_SZ_LOG_REG
		elif (split == "validate"):
			self.length = VAL_SET_SZ_GAN if gan_size else VAL_SET_SZ_LOG_REG
		elif (split == "test"):
			self.length = TEST_SET_SZ_GAN if gan_size else TEST_SET_SZ_LOG_REG
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