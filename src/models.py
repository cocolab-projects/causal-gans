"""
models.py

Credit for generator/discriminator goes to eriklindernoren
Credit for inference net goes to mhw32

@author mmosse19
@version July 2019
"""
import numpy as np

import torch 
import torch.nn as nn

from utils import get_conv_output_dim
from generate import (TOTAL_NUM_WORLDS, MNIST_IMG_DIM)

IMG_DIM = MNIST_IMG_DIM*2

class LogisticRegression(nn.Module):
    def __init__(self, cf=True, channels=1, input_dim=IMG_DIM**2, output_dim=1):
        super(LogisticRegression, self).__init__()
        if (cf): input_dim *=TOTAL_NUM_WORLDS
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.criterion = torch.nn.BCELoss()

    def forward(self, img):
        batch_sz = img.size(0)
        img = img.view(batch_sz, -1)
        outputs = self.linear(img)
        return torch.sigmoid(outputs)

class ConvGenerator(nn.Module):
    def __init__(self, latent_dim, wass=False, train_on_mnist=False):
        super(ConvGenerator, self).__init__()
        self.optimizer = torch.optim.RMSprop if wass else torch.optim.Adam
        self.n_channels = 1

        # get dim of image (only supports square images)
        img_height = IMG_DIM
        if (train_on_mnist): img_height = MNIST_IMG_DIM

        self.init_size = img_height // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128*self.init_size**2),
            nn.BatchNorm1d(128*self.init_size**2, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, self.n_channels, 5, stride=1, padding=2),
            nn.Tanh())

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class ConvDiscriminator(nn.Module):
    def __init__(self, wass=False, train_on_mnist=False, attach_inference=False, z_dim=0):
        super(ConvDiscriminator, self).__init__()

        self.optimizer = torch.optim.RMSprop if wass else torch.optim.Adam
        self.criterion = torch.nn.BCELoss()
        self.n_channels=1
        self.attach_inference=attach_inference
        
        self.model = nn.Sequential(
            nn.Conv2d(self.n_channels, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        dim = 2048 if train_on_mnist else 8192
        if (attach_inference): dim += z_dim
        
        # The height and width of downsampled image
        self.adv_layer = nn.Sequential( 
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid())  # note: we apply sigmoid here

    def forward(self, img, z=None):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        if (attach_inference): torch.cat((out, z), dim=1)
        validity = self.adv_layer(out)
        return validity

class InferenceNet(nn.Module):
    def __init__(self, channels, img_size, z_dim):
        super(InferenceNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True),
            # -
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1, inplace=True))
        # The height and width of downsampled image
        self.fc = nn.Linear(2048, z_dim * 2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        z_params = self.fc(out)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

        return z_mu, z_logvar
