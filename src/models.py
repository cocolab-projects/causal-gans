import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 28 # why not 28*2?

class WeakClassifier(nn.Module):
	def __init__(self, channels=1, img_size=IMG_SIZE, n_classes=3):
		super(WeakClassifier, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(channels*img_size**2, 16),
			nn.ReLU(inplace=True),
			nn.Linear(16, n_classes)
		)

	# ask about this
	def forward(self, img):
		logits = self.net(img)
		return logits