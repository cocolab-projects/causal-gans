import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from generate import mnist_dir_setup, generate_worlds

TRAIN_SET_SZ = 5000
TEST_SET_SZ = 1000
VAL_SET_SZ = 1500

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")
CAUSAL_MNIST_DIR = os.path.join(DATA_DIR, 'causal_mnist')

class CausalMNIST(Dataset):
	# later: add vocab
	def __init__(self, root=CAUSAL_MNIST_DIR, split="train",
		channels=1, classes=None, target_trials_only=False):
		super(CausalMNIST, self).__init__()
		self.root = root
		self.train = train
		self.mnist = mnist_dir_setup(train)
		self.img_transform = transforms.ToTensor()

		if (split == "train"):
			self.length = TRAIN_SET_SZ
		elif (split == "test"):
			self.length = TEST_SET_SZ
		else (split == "validate")
			self.length = VAL_SET_SZ
		else:
			raise RuntimeError("CausalMNIST was expecting split to be 'train', 'test', or 'validate'.")
		
		self.imgs = [self.img_transform(Image.fromarray(generate_worlds(self.mnist)[0])) for i in range(self.length)]
		self.labels = ["causal" in utt for i in range(self.length)]

	def __getitem__(self, index):
		return img, label

	def __len__(self):
		return self.length