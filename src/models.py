import numpy as np

import torch 
import torch.nn as nn

from generate import TOTAL_NUM_WORLDS

IMG_SIZE = 64

class LogisticRegression(nn.Module):
    def __init__(self, cf = False, channels=1, input_dim=IMG_SIZE**2, output_dim=1):
        super(LogisticRegression, self).__init__()
        if (cf):
            input_dim *=TOTAL_NUM_WORLDS
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, img):
        batch_sz = img.size(0)
        img = img.view(batch_sz, -1)
        outputs = self.linear(img)
        return torch.sigmoid(outputs)