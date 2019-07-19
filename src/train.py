"""
train.py

Much credit for GAN training goes to eriklindernoren, mhw32

TODO:
        (3) add generated cfs

@author mmosse19
@version July 2019
"""
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import chain

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as utils
from torchvision.utils import save_image

from torch.utils.data.dataset import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transform

from generate import mnist_dir_setup
from datasets import (CausalMNIST)
from models import (LogisticRegression, ConvGenerator, ConvDiscriminator, InferenceNet)
from utils import (LossTracker, AverageMeter, save_checkpoint, free_params, frozen_params, to_percent, viewable_img, reparameterize)

CLASSIFIER_LOSS_WT = 1.0
SUPRESS_PRINT_STATEMENTS = True 

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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size [default=64]')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate [default: 2e-4]')
    parser.add_argument('--lr_d', type=float, default=1e-5,
                        help='discriminator learning rate [default: 1e-5]')
    parser.add_argument('--epochs', type=int, default=101,
                        help='number of training epochs [default: 51]')
    parser.add_argument('--cuda', action='store_true',
                        help='Enable cuda')
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")

    # for GANs
    parser.add_argument("--latent_dim", type=int, default=40,
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
    tracker = checkpoint['tracker']
    classifier = checkpoint['classifier']
    GAN = checkpoint['GAN']
    return epoch, tracker, classifier, GAN

def record_progress(epoch, epochs, batch_num, num_batches, tracker, kind, amt, batch_size):
    # update tracker
    tracker.update(epoch, kind, amt, batch_size)

    # print avg for current epoch
    loss = tracker[kind][epoch].avg
    progress = "[epoch {}/{}]\t[batch {}/{}]\t[{} (epoch running avg):\t\t{}]".format(epoch+1, epochs, batch_num+1, num_batches, kind, loss)
    
    if (batch_num % 30 == 0) and not SUPRESS_PRINT_STATEMENTS: print(progress)

    # save tracker avgs (for each epoch) to file
    num_epochs_with_data = len(tracker[kind])
    data = [tracker[kind][e].avg for e in range(num_epochs_with_data)]
    np.savetxt("./progress_" + kind + ".txt", data)

# TRAINING

def descend(optimizers, loss):
    for optimizer in optimizers: optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizers: optimizer.step()

# COMPUTING LOSS FOR GAN (DISCRIMINATOR AND GENERATOR)

def get_loss_d(wass, discriminator, x, x_g, valid, fake, attach_inference, z_prior, z_inf):
    if (wass):
        return -torch.mean(discriminator(x)) + torch.mean(discriminator(x_g))
    elif (attach_inference):
        pred_fake = discriminator(x_g, z_prior)
        pred_real = discriminator(x, z_inf)
        return torch.mean(F.softplus(-pred_real)) + torch.mean(F.softplus(pred_fake))
    else:
        real_loss = discriminator.criterion(discriminator(x), valid)
        fake_loss = discriminator.criterion(discriminator(x_g), fake)
        return (real_loss + fake_loss) / 2

def get_loss_g(wass, discriminator, x, x_g, valid, attach_inference, z_prior, z_inf):
    if (wass):
        return -torch.mean(discriminator(x_g))
    elif(attach_inference):
        pred_fake = discriminator(x_g, z_prior)
        pred_real = discriminator(x, z_inf)
        return torch.mean(F.softplus(pred_real)) + torch.mean(-F.softplus(pred_fake))
    else:
        print("wass and attach_inference are both false")
        return discriminator.criterion(discriminator(x_g), valid)

# CLAMPING DISCRIMINATOR FOR GAN

def clip_discriminator(discriminator):
    for p in discriminator.parameters():
        p.data.clamp_(-args.clip_value,args.clip_value)

# TRAINING FOR LOG REG

# single pass over all data
def log_reg_run_batch(batch_num, num_batches, imgs, labels, model, mode, epoch, epochs, tracker, optimizer=None):
    labels = labels.float()

    if ("train" in mode):
        model.train()
    else:
        model.eval()

    outputs = model(imgs)

    # if testing for accuracy, round outputs; else add dim to labels
    if (mode == "test"):
        outputs = np.rint(outputs.numpy().flatten())
        labels = labels.numpy()
    else:
        labels = labels.unsqueeze(1)

    # calculate loss, update tracker
    if (mode == "test"):
        # find loss on images labeled '1' (i.e., causal images)
        causal_indices = np.where(labels == 1)
        causal_loss = (outputs[causal_indices] == labels[causal_indices])
        causal_loss_amt = np.mean(causal_loss)
        tracker.update(epoch, mode + "_causal_accuracy", causal_loss_amt, len(causal_indices))

        # find loss
        loss = (outputs == labels)
        loss_amt = np.mean(loss)
    else:
        # find loss
        loss = model.criterion(outputs,labels)
        loss_amt = loss.item()

    # compute and check batch_size
    batch_size = imgs.size(0) # don't use labels.size(0), bc if "test", labels are edited above

    # update tracker
    record_progress(epoch, epochs, batch_num, num_batches, tracker, mode + "_loss_c", loss_amt, batch_size)

    # follow gradient
    if (mode == "train"): descend([optimizer], loss)
    return loss

def log_reg_run_all_batches(loader, model, mode, epoch, epochs, tracker, optimizer):
    for batch_num, (imgs, labels) in enumerate(loader):
        if (mode == "train"):
            log_reg_run_batch(batch_num, len(loader), imgs, labels, model, mode, epoch, epochs, tracker, optimizer)
        else:
            with torch.no_grad():
                log_reg_run_batch(batch_num, len(loader), imgs, labels, model, mode, epoch, epochs, tracker)

# model is log reg model
def log_reg_run_epoch(loader, model, mode, epoch, epochs, tracker, optimizer = None, generator_state=None):
    # run all batches
    log_reg_run_all_batches(loader, model, mode, epoch, epochs, tracker, optimizer)
    
    # get avg loss for this epoch, save best loss if validating
    avg_loss = tracker[mode + "_loss_c"][epoch].avg

    if (mode == "validate"):
        tracker.best_loss = min(avg_loss, tracker.best_loss)
                
        save_checkpoint({
            'epoch': epoch,
            'classifier': model.state_dict(),
            'GAN': generator_state,
            'tracker': tracker,
            'cmd_line_args': args,
            'seed': args.seed
        }, tracker.best_loss == avg_loss, folder = args.out_dir)

    # report loss
    if epoch % 10 == 0:
        print('====> {} loss for log reg \t(epoch {}):\t {:.4f}'.format(mode, epoch+1, avg_loss))
    if (mode=="test"):
        avg_causal_loss = tracker[mode + "_causal_accuracy"][epoch].avg
        print('====> test accuracy for log reg on causal pics: {}%'.format(to_percent(avg_causal_loss)))
    return avg_loss

# train/test/validate log reg
def run_log_reg(train_loader, valid_loader, test_loader, args, cf, tracker):
    model = LogisticRegression(cf).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(int(args.epochs)):
        train_loss = log_reg_run_epoch(train_loader, model, "train", epoch, args.epochs, tracker, optimizer)
        validate_loss = log_reg_run_epoch(valid_loader, model, "validate", epoch, args.epochs, tracker)

    test_log_reg_from_checkpoint(test_loader, tracker, args, cf)

# test log reg
def test_log_reg_from_checkpoint(test_loader, tracker, args, cf):
    test_model = LogisticRegression(cf)
    _,_,classifier,GAN = load_checkpoint(folder=args.out_dir)
    test_model.load_state_dict(classifier)
    log_reg_run_epoch(test_loader, test_model, "test", 0, 0, tracker)

def get_causal_mnist_dataset(mode, cf, transform, mnist):
    data = CausalMNIST(split=mode, mnist=mnist, cf=cf, transform=transform)
    return data

def get_causal_mnist_loaders(using_gan, cf, transform, train_on_mnist):
    train_mnist = mnist_dir_setup(test=False)
    test_mnist = mnist_dir_setup(test=True)

    train = CausalMNIST("train", train_mnist, using_gan, cf=cf, transform=transform, train_on_mnist=train_on_mnist)
    valid = CausalMNIST("validate", train_mnist, using_gan, cf=cf, transform=transform, train_on_mnist=train_on_mnist)
    test = CausalMNIST("test", test_mnist, using_gan, cf=cf, transform=transform, train_on_mnist=train_on_mnist)

    train_loader = DataLoader(train, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(valid, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test, shuffle=True, batch_size=args.batch_size)

    return train_loader, valid_loader, test_loader

def save_images_from_g(generator, epoch, wass, latent_dim, batch_size):
    with torch.no_grad():
        # z ~ N(0,1)
        z = torch.randn(batch_size, latent_dim, device=device)

        # generate (batch_size) images
        gen_imgs = generator(z)

        n_imgs_per_row = 5
        n_images = n_imgs_per_row**2

        if epoch % 5 == 0:
            title = "{}GAN after {} epochs.png".format("W" if wass else "", format(epoch, "04"))
            save_image(gen_imgs.data[:n_images], title, nrow=n_imgs_per_row, normalize=True)

if __name__ == "__main__":
    # external args
    args = handle_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # internal args
    cf = False
    transform = True
    using_gan = True
    train_on_mnist = False

    # set up data loaders, loss tracker
    
    train_loader, valid_loader, test_loader = get_causal_mnist_loaders(using_gan, cf, transform, train_on_mnist)
    
    tracker = LossTracker()

    # Option 1: linear classifier alone
    if (not using_gan):
        run_log_reg(train_loader, valid_loader, test_loader, args, cf, tracker)
        breakpoint()    # to prevent GAN from training

    # Option 2: GAN, with the option to attach a linear classifier
    # TODO: wass and attach_inference can't work together; loss is computed differently
    wass = False
    attach_classifier = False
    attach_inference = True

    # setup models and optimizers
    generator = ConvGenerator(args.latent_dim, wass, train_on_mnist).to(device)
    inference_net = InferenceNet(1, 64, args.latent_dim).to(device)
    discriminator = ConvDiscriminator(wass, train_on_mnist, attach_inference, args.latent_dim).to(device)
    classifier = LogisticRegression(cf).to(device)

    generator.train()
    discriminator.train()
    inference_net.train()
    classifier.train()
    
    optimizer_g = generator.optimizer(chain(generator.parameters(),
                                            inference_net.parameters()),
                                            lr=args.lr)
    optimizer_d = discriminator.optimizer(discriminator.parameters(), lr=args.lr_d)
    optimizer_c = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # train (and validate, if attach_classifier)
    for epoch in range(args.epochs):
        pbar = tqdm(total = len(train_loader))
        # train
        for batch_num, (imgs, labels) in enumerate(train_loader):
            x, labels = imgs.to(device), labels.to(device)
            # x.shape = (64, 1, 64, 64)
            batch_size = x.size(0)

            optimizer_d.zero_grad()

            frozen_params(generator)
            frozen_params(inference_net)
            frozen_params(classifier)
            free_params(discriminator)

            # adversarial ground truths
            valid = torch.ones(x.size(0), 1, device=device)
            fake = torch.zeros(x.size(0), 1, device=device)

            # z_prior ~ N(0,1)
            z_prior = torch.randn(batch_size, args.latent_dim, device=device)
            
            # define q(z|x)
            z_inf_mu, z_inf_logvar = inference_net(x) # note: x may not be appropriate shape

            # z_inf ~ q(z|x)
            z_inf = reparameterize(z_inf_mu, z_inf_logvar)

            # x ~ p(x | z_prior)
            x_g = generator(z_prior)

            # train discriminator
            loss_d = get_loss_d(wass, discriminator, x, x_g, valid, fake, attach_inference, z_prior, z_inf)
            descend([optimizer_d], loss_d)
            record_progress(epoch, args.epochs, batch_num, len(train_loader), tracker, "train_loss_d", loss_d.item(), batch_size)

            # clip discriminator if wass
            if (wass): clip_discriminator(discriminator)

            # train generator (and classifier if necessary); execute unconditionally if GAN and periodically if WGAN
            if not wass or batch_num % args.n_critic == 0:
                optimizer_g.zero_grad()

                free_params(generator)
                free_params(inference_net)
                free_params(classifier)
                frozen_params(discriminator)

                # x_g ~ p(x|z_prior)
                x_g = generator(z_prior)

                loss_g = get_loss_g(wass, discriminator, x, x_g, valid, attach_inference, z_prior, z_inf)
                record_progress(epoch, args.epochs, batch_num, len(train_loader), tracker, "train_loss_g", loss_g.item(), batch_size)
                total_loss = loss_g
                optimizers = [optimizer_g]

                if (attach_classifier):
                    if (attach_inference):
                        
                    loss_c = log_reg_run_batch(batch_num, len(train_loader), x, labels, classifier, "train(+GAN)", epoch, args.epochs, tracker, optimizer_c)
                    total_loss += CLASSIFIER_LOSS_WT*loss_c
                    optimizers.append(optimizer_c)
                
                descend(optimizers, total_loss)
                tracker.update(epoch, "train_loss_total", total_loss.item(), batch_size)

            pbar.update()
        pbar.close()
        # finished training for epoch; print train loss, output images
        print('====> total train loss \t(epoch {}):\t {:.4f}'.format(epoch+1, tracker["train_loss_total"][epoch].avg))
        save_images_from_g(generator, epoch+1, wass, args.latent_dim, args.batch_size)
        
        # validate (if attach_classifier); this saves a checkpoint if the loss was especially good
        if (attach_classifier):
            validate_loss = log_reg_run_epoch(valid_loader, classifier, "validate", epoch, args.epochs, tracker, generator.state_dict())

    # test
    if (attach_classifier):
        test_log_reg_from_checkpoint(test_loader, tracker, args, cf)
