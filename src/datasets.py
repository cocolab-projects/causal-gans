import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from generate import mnist_dir_setup, generate_worlds

TRAIN_SET_SZ = 50000
TEST_SET_SZ = 10000

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")
CAUSAL_MNIST_DIR = os.path.join(DATA_DIR, 'causal_mnist')

class CausalMNIST(Dataset):
	# later: add vocab
	def __init__(self, root=CAUSAL_MNIST_DIR, train=True,
		channels=1, classes=None, target_trials_only=False):
		super(CausalMNIST, self).__init__()
		self.root = root
		self.train = train
		self.length = TRAIN_SET_SZ if train else TEST_SET_SZ
		self.mnist = mnist_dir_setup(train)

		self.img_transform = transforms.ToTensor()

	def __getitem__(self, index):
		img, utt = generate_worlds(self.mnist)
		img = Image.fromarray(img)
		img = self.img_transform(img)
		label = "causal" in utt

		return img, label

	def __len__(self):
		return self.length