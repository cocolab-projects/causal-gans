"""
train.py

Credit for GAN code goes to eriklindernoren

@author mmosse19
@version July 2019
"""
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.utils as utils
from torchvision.utils import save_image

from datasets import CausalMNIST
from models import (LogisticRegression, Generator, Discriminator)
from utils import (AverageMeter, save_checkpoint, free_params, frozen_params, to_percent)

# single pass over the data
def log_reg_loop(loader, model, mode, pbar=None):
    loss_meter = AverageMeter()
    causal_loss_meter = AverageMeter()
    
    if (mode != "test"):
        criterion = torch.nn.BCELoss()

    for i, (imgs, labels) in enumerate(loader):
        imgs,labels = imgs.to(device), labels.to(device).float()
        outputs = model(imgs)

        # if testing for accuracy, round outputs; else add dim to labels
        if (mode == "test"):
            outputs = np.rint(outputs.numpy().flatten())
            labels = labels.numpy()
        else:
            labels = labels.unsqueeze(1)

        # calculate loss
        if (mode == "test"):
            loss = (outputs == labels)
            loss_amt = np.mean(loss)

            causal_indices = np.where(labels == 1)
            causal_loss = (outputs[causal_indices] == labels[causal_indices])
            causal_loss_amt = np.mean(causal_loss)
            causal_loss_meter.update(causal_loss_amt, len(causal_indices))
        else:
            loss = criterion(outputs,labels)
            loss_amt = loss.item()

        # update loss meters
        loss_meter.update(loss_amt, loader.batch_size)

        if (mode == "train"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss_meter.avg})
            pbar.update()

    return loss_meter.avg, causal_loss_meter.avg

def log_reg_run_epoch(loader, model, mode, epoch=0):
    if (mode == "train"):
        model.train()
        pbar = tqdm(total=len(loader))
        avg_loss, _ = log_reg_loop(loader, model, mode, pbar)
        pbar.close()
    else:
        model.eval()
        with torch.no_grad():
            avg_loss, avg_causal_loss = log_reg_loop(loader, model, mode)

    if (mode=="test"):
            print('====> test accuracy: {}%'.format(to_percent(avg_loss)))
            print('====> test accuracy on causal pics: {}%'.format(to_percent(avg_causal_loss)))
    elif epoch % 20 == 0:
            print('====> {} epoch: {}\tloss: {:.4f}'.format(mode, epoch, avg_loss))
    return avg_loss

def load_checkpoint(folder='./', filename='model_best.pth.tar'):
    checkpoint = torch.load(folder + filename)
    epoch = checkpoint['epoch']
    track_loss = checkpoint['track_loss']
    model = checkpoint['model']
    return epoch, track_loss, model

def nearby(pertburbation, z, eps):
    return False

def latent_cf(perturbation, z):
    return z

# z has shape (batch-size, latent_dim)
def latent_cfs(z, eps):
    latent_dim = z.size(1)
    cfs = []

    component_dist_weights = [.5, .5]
    prior_mean = 0
    prior_scale = 1
    for i in range(latent_dim):
        posterior_mean = z[:,i]
        posterior_scale = unnkown
        means = [prior_mean, posterior_mean]
        scales = [prior_scale, posterior_scale]

        perturbation = pyro.distributions.MixtureOfDiagNormals(means, scales, component_dist_weights)
        while nearby(perturbation, z, eps):
            perturbation = pyro.distributions.MixtureOfDiagNormals(means, scales, component_dist_weights)
        cfs.append(latent_cf(perturbation, z))
    return cfs

def get_loss_d(wass, discriminator, imgs, gen_imgs, valid, fake):
    imgs = imgs.type(Tensor)
    if (wass):
        return -torch.mean(discriminator(imgs)) + torch.mean(discriminator(gen_imgs))
    else:
        with torch.no_grad():
            real_loss = torch.nn.BCELoss(discriminator(imgs), valid)
            fake_loss = torch.nn.BCELoss(discriminator(gen_imgs), fake)
            return (real_loss + fake_loss) / 2

def get_loss_g(wass, discriminator, gen_imgs, valid):
    if (wass):
        return -torch.mean(discriminator(gen_imgs))
    else:
        return torch.nn.BCELoss(discriminator(gen_imgs), valid)

def clip_discriminator(discriminator, clip_value):
    for p in discriminator.parameters():
        p.data.clamp_(-clip_value,clip_value)

def handle_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str,
                        help='where to save checkpoints',default="./")
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [default: 42]')
    parser.add_argument('--resample_eps', type=float, default=1e-3,
                        help='epsilon ball to resample z')
    # for learning
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size [default=64]')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate [default: 2e-4]')
    parser.add_argument('--lr_d', type=float, default=2e-5,
                        help='discriminator learning rate [default: 2e-5]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs [default: 50]')
    parser.add_argument('--cuda', action='store_true',
                        help='Enable cuda')
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")

    # for GANs
    parser.add_argument("--latent_dim", type=int, default=5,
                        help="dimensionality of the latent space")
    parser.add_argument("--sample_interval", type=int, default=500,
                        help="interval betwen image samples")
    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for discriminator per iter")
    return parser.parse_args()

# todo: consider preprocessing (center pixels around 0, ensure |value| \leq 1)
if __name__ == "__main__":
    # args
    args = handle_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # train, validate, test
    cf = False
    transform = False
    wass = True
    attach_classifier = False

    train_dataset = CausalMNIST(split="train", cf=cf, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # GAN
    generator = Generator(latent_dim=args.latent_dim, cf=cf, wass=wass)
    discriminator = Discriminator(cf=cf, wass=wass)
    optimizer_g = generator.optimizer(generator.parameters(), lr=args.lr)
    optimizer_d = discriminator.optimizer(discriminator.parameters(), lr=args.lr)

    # necessary?
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    for epoch in range(int(args.epochs)):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs,labels = imgs.to(device), labels.to(device)

            # adversarial ground truths
            with torch.no_grad():
                valid = Tensor(imgs.size(0), 1).fill_(1.0)
                fake = Tensor(imgs.size(0), 1).fill_(0.0)

            # generate images      
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
            with torch.no_grad():
                gen_imgs = generator(z)

            # train discriminator
            loss_d = get_loss_d(wass, discriminator, imgs, gen_imgs, valid, fake)
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # clip discriminator if wass
            if (wass): clip_discriminator(discriminator, args.clip_value)

            # train generator (and log_reg)
            if not wass or i % args.n_critic == 0:
                gen_imgs = generator(z)

                loss_g = get_loss_g(wass, discriminator, gen_imgs, valid)
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

                # later: replace with pbar, integrate with run method above
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.epochs, i, len(train_loader), loss_d.item(), loss_g.item())
                )

            batches_done = epoch * len(train_loader) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], "%d.png" % batches_done, nrow=5)

    # LOGISTIC REGRESSION
    """
    log_reg = LogisticRegression(cf)
    log_reg = log_reg.to(device)
    optimizer = torch.optim.Adam(log_reg.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    valid_dataset = CausalMNIST(split="validate", cf=cf)
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)

    # train and validate, keeping track of best loss in validation
    best_loss = float('inf')
    track_loss = np.zeros((args.epochs, 2))

    for epoch in range(int(args.epochs)):

        train_loss = log_reg_run_epoch(train_loader, log_reg, "train", epoch)
        validate_loss = log_reg_run_epoch(valid_loader, log_reg, "validate", epoch)

        is_best = validate_loss < best_loss
        best_loss = min(validate_loss, best_loss)
        track_loss[epoch, 0] = train_loss
        track_loss[epoch, 1] = validate_loss
        
        save_checkpoint({
            'epoch': epoch,
            'model': log_reg.state_dict(),
            'optimizer': optimizer.state_dict(),
            'track_loss': track_loss,
            'cmd_line_args': args,
            'seed': args.seed
        }, is_best, folder = args.out_dir)
        # np.save(os.path.join(args.out_dir, 'loss.npy'), track_loss)

    # test
    test_dataset = CausalMNIST(split='test',cf=cf)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    test_model = LogisticRegression(cf)
    _,_,state_dict = load_checkpoint(folder=args.out_dir)
    test_model.load_state_dict(state_dict)
    run_epoch(test_loader, test_model, "test")
    """
