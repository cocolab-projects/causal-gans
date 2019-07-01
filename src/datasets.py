import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from generate import mnist_dir_setup, generate_worlds

DATA_SIZE = 100

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")
CAUSAL_MNIST_DIR = os.path.join(DATA_DIR, 'causal_mnist')

class CausalMNIST(Dataset):
	# later: add vocab
	def __init__(self, root=CAUSAL_MNIST_DIR, train=True,
		channels=1, classes=None, target_trials_only=False):
		super(CausalMNIST, self).__init__()
		self.root = root
		self.train = train
		self.mnist = mnist_dir_setup(train)
		self.images, self.labels = [], []

		assert len(self.images) == len(self.raw_utterances)

		self.img_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])

	def __getitem__(self, index):
		img, utt = generate_worlds(mnist)
		img = Image.fromarray(img)
		img = self.img_transform(img)
		label = causal_utt(utt)

		self.images.append(img)
		self.labels.append(label)

		assert len(self.images) == len(self.raw_utterances)

	def __len__(self)
		return len(self.images)

	def causal_utt(utt):
		return "causal" in utt