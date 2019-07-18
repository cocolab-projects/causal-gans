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

from generate import (TOTAL_NUM_WORLDS, IMG_DIM)

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
        self.criterion = torch.nn.BCELoss()

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

class ConvGenerator(nn.Module):
    def __init__(self, latent_dim, cf=True, wass=False, n_channels=1, img_height=IMG_DIM, img_width=IMG_DIM):
        super(ConvGenerator, self).__init__()
        self.optimizer = torch.optim.RMSprop if wass else torch.optim.Adam
        if (cf): img_height *= TOTAL_NUM_WORLDS  # DEPRECATED? this wont work with this arch
        assert img_height == img_width, "only square images supported for now"  # block non-square like cf=True

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
            nn.Conv2d(128, n_channels, 5, stride=1, padding=2),
            nn.Tanh())

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class ConvDiscriminator(nn.Module):
    def __init__(self, cf=True, wass=False, input_dim=IMG_DIM, n_channels=1, img_height=IMG_DIM, img_width=IMG_DIM):
        super(ConvDiscriminator, self).__init__()

        assert img_height == img_width, "only square images supported for now"
        self.optimizer = torch.optim.RMSprop if wass else torch.optim.Adam
        self.criterion = torch.nn.BCELoss()
       
        # @mmosse19, is this every necessary... we dont want to train a GAN given CFs?
        if (cf): img_height *= TOTAL_NUM_WORLDS

        self.model = nn.Sequential(
            nn.Conv2d(n_channels, 128, 3, stride=1, padding=1),
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
        self.adv_layer = nn.Sequential( 
            nn.Linear(2048, 512),  # fix me to not hardcode to 2048
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid())  # note: we apply sigmoid here
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
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
