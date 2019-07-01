import numpy as np

import torch 
import torch.nn as nn

from datasets

IMG_SIZE = 28 # why not 28*2?

class LogisticRegression(nn.Module):
	def __init__(self, channels=1, img_size=IMG_SIZE, n_classes=3):
		super(LogisticRegression, self).__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)

	# ask about this
	def forward(self, img):
		outputs = self.linear(img)
		return outputs

