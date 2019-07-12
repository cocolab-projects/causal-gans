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
from utils import (LossTracker, AverageMeter, save_checkpoint, free_params, frozen_params, to_percent, viewable_img)

CLASSIFIER_LOSS_WT = 1.0

# ARGUMENTS; CHECKPOINTS AND PROGRESS WHILE TRAINING

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

def load_checkpoint(folder='./', filename='model_best.pth.tar'):
    checkpoint = torch.load(folder + filename)
    epoch = checkpoint['epoch']
    loss_tracker = checkpoint['loss_tracker']
    classifier = checkpoint['classifier']
    GAN = checkpoint['GAN']
    return epoch, loss_tracker, classifier, GAN

def record_progress(epoch, epochs, batch, num_batches, loss_tracker):
    losses = [kind + ": " + str(loss_tracker.loss_kinds[kind].avg) for kind in loss_tracker.kinds]
    np.savetxt("./losses.txt", losses)
    progress = "[Epoch %d/%d] [Batch %d/%d]. Losses (running avg): %s" % (epoch, epochs, batch, num_batches, str(losses))
    print(progress)

# COMPUTING LOSS FOR GAN (DISCRIMINATOR AND GENERATOR)

def get_loss_d(wass, discriminator, imgs, gen_imgs, valid, fake):
    if (wass):
        return -torch.mean(discriminator(imgs)) + torch.mean(discriminator(gen_imgs))
    else:
        real_loss = torch.nn.BCELoss(discriminator(imgs), valid)
        fake_loss = torch.nn.BCELoss(discriminator(gen_imgs), fake)
        return (real_loss + fake_loss) / 2

def get_loss_g(wass, discriminator, gen_imgs, valid):
    if (wass):
        return -torch.mean(discriminator(gen_imgs))
    else:
        return torch.nn.BCELoss(discriminator(gen_imgs), valid)

# CLAMPING DISCRIMINATOR for GAN

def clip_discriminator(discriminator, clip_value):
    for p in discriminator.parameters():
        p.data.clamp_(-clip_value,clip_value)

# LOGISTIC REGRESSION TRAINING

# single pass over all data
def log_reg_loop(loader, model, mode, optimizer, pbar=None):
    loss_meter = AverageMeter()
    loss_tracker = LossTracker()
    
    if (mode != "test"):
        criterion_log_reg = torch.nn.BCELoss()
    else:
        causal_loss_meter = AverageMeter()

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
            loss_tracker.update("causal_loss", causal_loss_amt, len(causal_indices))
        else:
            loss = criterion_log_reg(outputs,labels)
            loss_amt = loss.item()

        loss_tracker.update(mode + "_loss", loss_amt, loader.batch_size)

        if (mode == "train"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        record_progress(epoch, args.epochs, i, len(train_loader), loss_tracker)

    return loss_meter.avg, causal_loss_meter.avg

# sets model appropriately, calls log_reg_loop, and reports results
def log_reg_run_epoch(loader, model, mode, tracker=None, epoch=0, optimizer = None):
    if (mode == "train"):
        model.train()
        assert(optimizer is not None)
        avg_loss, _ = log_reg_loop(loader, model, mode, optimizer)
    else:
        model.eval()
        with torch.no_grad():
            avg_loss, avg_causal_loss = log_reg_loop(loader, model, mode, None)

    if (mode=="test"):
            print('====> test accuracy: {}%'.format(to_percent(avg_loss)))
            print('====> test accuracy on causal pics: {}%'.format(to_percent(avg_causal_loss)))
    elif epoch % 20 == 0:
            print('====> {} epoch {}\tloss: {:.4f}'.format(mode, epoch, avg_loss))
    return avg_loss

# test log reg
def test_log_reg_from_checkpoint(args, cf):
    test_dataset = CausalMNIST(split='test',cf=cf)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    test_model = LogisticRegression(cf)
    _,_,classifier,GAN = load_checkpoint(folder=args.out_dir)
    test_model.load_state_dict(classifier)
    log_reg_run_epoch(test_loader, test_model, "test")

# train/test/validate log reg
def log_reg(train_loader, args, cf):
    model = LogisticRegression(cf).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    valid_dataset = CausalMNIST(split="validate", cf=cf)
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)

    tracker = LossTracker()

    for epoch in range(int(args.epochs)):
        train_loss = log_reg_run_epoch(train_loader, model, "train", tracker, epoch, optimizer)
        validate_loss = log_reg_run_epoch(valid_loader, model, "validate", tracker, epoch)
            
        tracker.best_loss = min(validate_loss, tracker.best_loss)
        tracker.update("train_loss_log_reg", epoch, train_loss.item())
        tracker.update("validate_loss_log_reg", epoch, validate_loss.item())

        save_checkpoint({
            'epoch': epoch,
            'classifier': model.state_dict(),
            'GAN': 0,
            'loss_tracker': tracker,
            'cmd_line_args': args,
            'seed': args.seed
        }, tracker.best_loss == validate_loss, folder = args.out_dir)

    test_log_reg_from_checkpoint(args, cf)

# MAIN: TRAINS GANs
if __name__ == "__main__":
    # args
    args = handle_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cf = False
    transform = True

    train_dataset = CausalMNIST(split="train", cf=cf, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # train linear model alone
    log_reg(train_loader=train_loader, args=args, cf=cf)

    """
    # train GANs
    wass = True
    attach_classifier = True

    generator = Generator(latent_dim=args.latent_dim, cf=cf, wass=wass)
    discriminator = Discriminator(cf=cf, wass=wass)
    # optimizer is set by constructor depending on the value of wass
    optimizer_g = generator.optimizer(generator.parameters(), lr=args.lr)
    optimizer_d = discriminator.optimizer(discriminator.parameters(), lr=args.lr)

    if (attach_classifier):
        log_reg = LogisticRegression(cf).to(device)
        optimizer_log_reg = torch.optim.Adam(log_reg.parameters(), lr=args.lr, betas=(args.b1, args.b2))

        valid_dataset = CausalMNIST(split="validate", cf=cf)
        valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)

    loss_tracker = LossTracker()

    # train (and validate, if attach_classifier)
    for epoch in range(int(args.epochs)):
        if (attach_classifier):
            log_reg.train()
            criterion_log_reg = torch.nn.BCELoss()

            loss_meter = AverageMeter()
            causal_loss_meter = AverageMeter()

        for i, (imgs, labels) in enumerate(train_loader):
            batch_size = imgs.size(0)

            imgs,labels = imgs.to(device), labels.to(device)

            # adversarial ground truths
            with torch.no_grad():
                valid = torch.ones(imgs.size(0), 1, device=device)
                fake = torch.zeros(imgs.size(0), 1, device=device)

            # generate images      
            z = torch.randn(batch_size, args.latent_dim, device=device)

            with torch.no_grad():
                gen_imgs = generator(z)

            # train discriminator
            loss_d = get_loss_d(wass, discriminator, imgs, gen_imgs, valid, fake)
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            loss_tracker.update("train_loss_d", epoch, loss_d, batch_size)

            # clip discriminator if wass
            if (wass): clip_discriminator(discriminator, args.clip_value)

            # train generator (and log_reg if necessary)
            # we do this unconditionally if GAN and periodically if WGAN
            if not wass or i % args.n_critic == 0:
                gen_imgs = generator(z)
                optimizer_g.zero_grad()

                loss_g = get_loss_g(wass, discriminator, gen_imgs, valid)
                loss_tracker.update("train_loss_g", epoch, loss_g, batch_size)
                total_loss = loss_g

                if (attach_classifier):
                    outputs = log_reg(imgs)
                    loss_log_reg = criterion_log_reg(outputs, labels.float().unsqueeze(1))
                    loss_log_reg_amt = loss_log_reg.item()
                    loss_tracker.update("train_loss_log_reg", epoch, loss_log_reg_amt, batch_size)

                    loss += CLASSIFIER_LOSS_WT*loss_log_reg
                    optimizer_log_reg.zero_grad()

                total_loss.backward()
                optimizer_g.step()

                if (attach_classifier):
                    optimizer_log_reg.step()

                # is this the best place for this?
                record_progress(epoch, args.epochs, i, len(train_loader), loss_tracker)

            batches_done = epoch * len(train_loader) + i
            if batches_done % args.sample_interval == 0:
                save_image(viewable_img(gen_imgs.data[:25]), "%d.png" % batches_done, nrow=5)
        
        if (attach_classifier):
            validate_loss = log_reg_run_epoch(valid_loader, log_reg, "validate", epoch)
            
            loss_tracker.best_loss = min(validate_loss, loss_tracker.best_loss)
            loss_tracker.update("validate_loss_log_reg", epoch, validate_loss)
            
            save_checkpoint({
                'epoch': epoch,
                'classifier': log_reg.state_dict(),
                'GAN': generator.state_dict(),
                'loss_tracker': loss_tracker,
                'cmd_line_args': args,
                'seed': args.seed
            }, loss_tracker.best_loss == validate_loss, folder = args.out_dir)

    # test
    test_log_reg_from_checkpoint(args, cf)
    """
