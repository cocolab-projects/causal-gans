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

# ask Mike about this later
class Classifier(nn.Module):
    def __init__(self, channels=1, img_size=32, n_filters=64, n_classes=3):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, n_filters, 2, 2, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        cout = gen_32_conv_output_dim(img_size)
        # NOTE: 3 classes (blank, 3, or 4)
        self.fc = nn.Linear(n_filters * 4 * cout**2, n_classes)
        self.cout = cout
        self.n_filters = n_filters