import numpy as np

import torch 
import torch.nn as nn

from generate import (TOTAL_NUM_WORLDS, IMG_DIM)

class LogisticRegression(nn.Module):
    def __init__(self, cf=True, channels=1, input_dim=IMG_DIM**2, output_dim=1):
        super(LogisticRegression, self).__init__()
        if (cf): input_dim *=TOTAL_NUM_WORLDS
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, img):
        batch_sz = img.size(0)
        img = img.view(batch_sz, -1)
        outputs = self.linear(img)
        return torch.sigmoid(outputs)

class Generator(nn.Module):
    def __init__(self, latent_dim, cf=True, wass=False, n_channels =1, img_height=IMG_DIM, img_width=IMG_DIM):
        super(Generator, self).__init__()

        self.optimizer = torch.optim.RMSprop if wass else torch.optim.Adam

        if (cf): img_height *= TOTAL_NUM_WORLDS
        self.img_shape = (n_channels, img_height, img_width)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, cf=True, wass=False, input_dim=IMG_DIM, n_channels =1, img_height=IMG_DIM, img_width=IMG_DIM):
        super(Discriminator, self).__init__()

        self.optimizer = torch.optim.RMSprop if wass else torch.optim.Adam

        # if we're including counterfactual imgs in training, they're concatenated, so we need to increase height
        if (cf): img_height *= TOTAL_NUM_WORLDS

        self.img_shape = (n_channels, img_height, img_width)
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity