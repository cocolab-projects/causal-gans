import numpy as np

import torch 
import torch.nn as nn

IMG_SIZE = 64

class LogisticRegression(nn.Module):
	def __init__(self, channels=1, input_dim=IMG_SIZE, output_dim=1):
		super(LogisticRegression, self).__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, img):
		outputs = self.linear(img)
		return outputs