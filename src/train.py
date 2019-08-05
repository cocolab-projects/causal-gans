"""
train.py

Much credit for GAN training goes to eriklindernoren, mhw32

@author mmosse19
@version July 2019
"""
# general
import os
import copy
from itertools import chain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# torch
import torch
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transform
from torchvision.utils import save_image
import torchvision.utils as utils

# from this dir
from generate import (mnist_dir_setup, NUM1, NUM2, IMG_DIM)
from datasets import (CausalMNIST)
from models import (LogisticRegression, ConvGenerator, ConvDiscriminator, InferenceNet)
from utils import (LossTracker, AverageMeter, save_checkpoint, free_params, frozen_params, to_percent, viewable_img, reparameterize, latent_cfs, args_to_string)

DATA_DIR = "/mnt/fs5/mmosse19/causal-gans/progress"
MAX_CLASS_WT = 1.0
SUPRESS_PRINT_STATEMENTS = True
LOSS_KINDS = {  "causal_loss": lambda utt: "causal" in utt,                     # find loss on images labeled '1' (i.e., causal images)
                "both_nums_loss": lambda utt: str(NUM1) in utt and str(NUM2) in utt}   # find loss on all images that contain both NUM1 and NUM2

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
    parser.add_argument('--transform', action='store_false',
                        help='apply nonlinear transform to causal images')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size [default=64]')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate [default: 2e-4]')
    parser.add_argument('--lr_d', type=float, default=1e-5,
                        help='discriminator learning rate [default: 1e-5]')
    parser.add_argument('--epochs', type=int, default=51,
                        help='number of training epochs [default: 101]')
    parser.add_argument('--cuda', action='store_false',
                        help='Enable cuda')
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    # when not using GAN
    parser.add_argument("--human_cf", action='store_true')

    # when using GN
    parser.add_argument("--latent_dim", type=int, default=4,
                        help="dimensionality of the latent space")
    parser.add_argument("--sample_interval", type=int, default=500,
                        help="interval betwen image samples")
    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")
    parser.add_argument('--n_critic', type=int, default=5,
                        help="number of training steps for discriminator per iter")
    parser.add_argument('--wass', action='store_true',
                        help="use WGAN instead of GAN")
    parser.add_argument("--train_on_mnist", action='store_true',
                        help="train on MNIST instead of CMNIST")
    parser.add_argument("--using_gan", action='store_true')
    parser.add_argument("--attach_classifier", action='store_true')
    parser.add_argument("--attach_inference", action='store_true')
    parser.add_argument("--cf_inf", action='store_true')
    parser.add_argument("--sample_from", type=str, default="post")
    parser.add_argument("--gradual_wt", action='store_false')

    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    if (args.cf_inf):
        args.attach_classifier = True
        args.attach_inference = True
    if (args.wass or args.attach_inference or args.attach_classifier):
        args.using_gan = True
    if (args.using_gan):
        args.human_cf = False
    return args

def load_checkpoint(folder='./', filename='model_best.pth.tar'):
    checkpoint = torch.load(folder + filename)
    return checkpoint['epoch'], checkpoint['classifier_state'], checkpoint['generator_state'], checkpoint['inference_net_state'], checkpoint['tracker'], checkpoint['cached_args']

def record_progress(epoch, epochs, batch_num, num_batches, tracker, kind, amt, batch_size, arg_str):
    # update tracker
    tracker.update(epoch, kind, amt, batch_size)

    # print avg for current epoch
    loss = tracker[kind][epoch].avg
    progress = "[epoch {}/{}]\t[batch {}/{}]\t[{} (epoch running avg):\t\t\t{}]".format(epoch+1, epochs, batch_num+1, num_batches, kind, loss)
    
    if (batch_num % 30 == 0) and not SUPRESS_PRINT_STATEMENTS: print(progress)

    # save tracker avgs (for each epoch) to file
    num_epochs_with_data = len(tracker[kind])
    data = [tracker[kind][e].avg for e in range(num_epochs_with_data)]
    np.savetxt(os.path.join(DATA_DIR, "progress_{}_{}.txt".format(kind, arg_str)), data)

# TRAINING

def descend(optimizers, loss):
    for optimizer in optimizers: optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizers: optimizer.step()

# COMPUTING LOSS FOR GAN (DISCRIMINATOR AND GENERATOR)

def get_loss_d(wass, discriminator, x, x_g, valid, fake, attach_inference, z_prior, z_inf):
    if (attach_inference):
        pred_fake = discriminator(x_g, z_prior)
        pred_real = discriminator(x, z_inf)
        return torch.mean(F.softplus(-pred_real)) + torch.mean(F.softplus(pred_fake))
    elif (wass):
        return -torch.mean(discriminator(x)) + torch.meang(discriminator(x_g))
    else:
        real_loss = discriminator.criterion(discriminator(x), valid)
        fake_loss = discriminator.criterion(discriminator(x_g), fake)
        return (real_loss + fake_loss) / 2

def get_loss_g(wass, discriminator, x, x_g, valid, attach_inference, z_prior, z_inf):
    if (attach_inference):
        pred_fake = discriminator(x_g, z_prior)
        pred_real = discriminator(x, z_inf)
        return torch.mean(F.softplus(pred_real)) + torch.mean(-F.softplus(pred_fake))
    elif (wass):
        return -torch.mean(discriminator(x_g))
    else:
        print("wass and attach_inference are both false")
        return discriminator.criterion(discriminator(x_g), valid)

# CLAMPING DISCRIMINATOR FOR GAN

def clip_discriminator(discriminator):
    for p in discriminator.parameters():
        p.data.clamp_(-args.clip_value,args.clip_value)

# TRAINING FOR LOG REG

def test_loss(loss_kind, outputs, utts, labels, condition, tracker, epoch):
    relevant_indices = np.where([condition(u) for u in utts])
    loss = (outputs[relevant_indices] == labels[relevant_indices])
    loss_amt = np.mean(loss)
    tracker.update(epoch, "test_" + loss_kind, loss_amt, len(relevant_indices))

# single pass over all data
def log_reg_run_batch(batch_num, num_batches, imgs, utts, labels, model, mode, epoch, epochs, tracker, arg_str, optimizer=None):
    labels = labels.float()

    if ("train" in mode):
        model.train()
    else:
        model.eval()

    outputs = model(imgs)

    # if testing for accuracy, round outputs; else add dim to labels
    if (mode == "test"):
        outputs = np.rint(outputs.cpu().numpy().flatten())
        labels = labels.cpu().numpy()
    else:
        labels = labels.unsqueeze(1)

    # calculate loss, update tracker
    if (mode == "test"):
        # find loss on images labeled '1' (i.e., causal images)
        for loss_kind, condition in LOSS_KINDS.items():
            test_loss(loss_kind, outputs, utts, labels, condition, tracker, epoch)

        # find total loss
        loss = (outputs == labels)
        loss_amt = np.mean(loss)
    else:
        # find loss
        loss = model.criterion(outputs,labels)
        loss_amt = loss.item()

    # compute and check batch_size
    batch_size = imgs.size(0) # don't use labels.size(0), bc if "test", labels are edited above

    # update tracker
    record_progress(epoch, epochs, batch_num, num_batches, tracker, mode + "_loss_c", loss_amt, batch_size, arg_str)

    # follow gradient
    if (mode == "train"): descend([optimizer], loss)
    return loss

def log_reg_run_all_batches(loader, model, mode, epoch, epochs, tracker, arg_str, optimizer, generator, inference_net, sample_from):
    for batch_num, (x, utts, labels) in enumerate(loader):
        x, utts, labels = x.to(device), utts, labels.to(device)
        x_to_classify = x
        if (model.cf and x_to_classify.shape[3] == IMG_DIM):
            # define q(z|x)
            z_inf_mu, z_inf_logvar = inference_net(x)

            # z_inf ~ q(z|x)
            z_inf = reparameterize(z_inf_mu, z_inf_logvar)

            # x_to_classify ~ q(x | cf(z_inf))
            x_to_classify = combine_x_cf(x, z_inf, z_inf_mu, torch.exp(0.5*z_inf_logvar), sample_from, generator)

        if (mode == "train"):
            log_reg_run_batch(batch_num, len(loader), x_to_classify, utts, labels, model, mode, epoch, epochs, tracker, arg_str, optimizer)
        else:
            with torch.no_grad():
                log_reg_run_batch(batch_num, len(loader), x_to_classify, utts, labels, model, mode, epoch, epochs, tracker, arg_str)

# model is log reg model
def log_reg_run_epoch(loader, model, mode, epoch, epochs, tracker, arg_str, optimizer = None, generator=None, inference_net=None, sample_from=None):
    # get generator state
    generator_state = generator.state_dict() if generator else None
    inference_net_state = inference_net.state_dict() if inference_net else None

    # run all batches
    log_reg_run_all_batches(loader, model, mode, epoch, epochs, tracker, arg_str, optimizer, generator, inference_net, sample_from)
    
    # get avg loss for this epoch, save best loss if validating
    avg_loss = tracker[mode + "_loss_c"][epoch].avg

    if (mode == "validate"):
        tracker.best_loss = min(avg_loss, tracker.best_loss)
                
        save_checkpoint({
            'epoch': epoch,
            'classifier_state': model.state_dict(),
            'generator_state': generator_state,
            'inference_net_state': inference_net_state,
            'tracker': tracker,
            'cached_args': args,
        }, tracker.best_loss == avg_loss, folder = args.out_dir)

    # report loss
    if (mode=="test"):
        print('====> test_total loss: \t\t\t {}%'.format(to_percent(1.0-avg_loss)))
        for loss_kind in LOSS_KINDS:
            avg_loss = tracker["test_" + loss_kind][epoch].avg
            print('====> test_{} loss:\t\t {}%'.format(loss_kind, to_percent(1.0-avg_loss)))
    else:
        print('====> {} loss log reg \t(epoch {}):\t\t {:.4f}'.format(mode, epoch+1, avg_loss))
    return avg_loss

# train/test/validate log reg
def run_log_reg(train_loader, valid_loader, test_loader, args, cf, tracker):
    model = LogisticRegression(cf).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(int(args.epochs)):
        log_reg_run_epoch(train_loader, model, "train", epoch, args.epochs, tracker, args_to_string(args), optimizer=optimizer)
        log_reg_run_epoch(valid_loader, model, "validate", epoch, args.epochs, tracker, args_to_string(args))

    test_log_reg_from_checkpoint(test_loader, tracker, args.out_dir, cf, args_to_string(args))

# test log reg
def test_log_reg_from_checkpoint(test_loader, tracker, out_dir, cf, arg_str, sample_from=None):
    epoch, classifier_state, generator_state, inference_net_state, tracker, cached_args = load_checkpoint(folder=out_dir)

    test_model = LogisticRegression(cf).to(device)
    test_model = LogisticRegression(cf).to(device)
    generator = ConvGenerator(cached_args.latent_dim, cached_args.wass, cached_args.train_on_mnist).to(device)
    inference_net = InferenceNet(1, 64, cached_args.latent_dim).to(device)

    test_model.load_state_dict(classifier_state)
    if (generator_state): generator.load_state_dict(generator_state)
    if (inference_net_state): inference_net.load_state_dict(inference_net_state)

    log_reg_run_epoch(test_loader, test_model, "test", 0, 0, tracker, arg_str, generator=generator, inference_net=inference_net, sample_from=sample_from)

def get_causal_mnist_loaders(using_gan, cf, transform, train_on_mnist):
    train_mnist = mnist_dir_setup(test=False)
    test_mnist = mnist_dir_setup(test=True)

    train = CausalMNIST("train", train_mnist, using_gan, cf=cf, transform=transform, train_on_mnist=train_on_mnist)
    valid = CausalMNIST("validate", train_mnist, using_gan, cf=cf, transform=transform, train_on_mnist=train_on_mnist)
    test = CausalMNIST("test", test_mnist, using_gan, cf=cf, transform=transform, train_on_mnist=train_on_mnist)

    train_loader = DataLoader(train, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(valid, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test, shuffle=True, batch_size=args.batch_size)
    
    print("filled data loaders.")

    return train_loader, valid_loader, test_loader

def save_images_from_g(generator, epoch, wass, latent_dim, batch_size, arg_str):
    with torch.no_grad():
        # z ~ N(0,1)
        z = torch.randn(batch_size, latent_dim, device=device)

        # generate (batch_size) images
        gen_imgs = generator(z)

        n_imgs_per_row = 5
        n_images = n_imgs_per_row**2

        if epoch % 5 == 0:
            title = "{}GAN after {} epochs {}.png".format("W" if wass else "", format(epoch, "04"), arg_str)
            title = os.path.join(DATA_DIR, title)
            save_image(gen_imgs.data[:n_images], title, nrow=n_imgs_per_row, normalize=True)

def combine_x_cf(x, z_inf, z_inf_mu, z_inf_sigma, sample_from, generator):
    latent_dim = z_inf.size(1)
    x_to_classify = [[img.squeeze()] for img in x]
    for dim in range(latent_dim):
        # z_cf = z_inf, with one dim ~ sample_from
        z_cf = latent_cfs(dim, z_inf, z_inf_mu, z_inf_sigma, sample_from)

        # x_cf ~ p(x | z_cf)
        x_cf = generator(z_cf)

        # append x_cf to x_to_classify
        for i, img in enumerate(x_to_classify):
            x_to_classify[i].append(x_cf[0].squeeze())

    x_to_classify = [torch.cat(cfs, 0).unsqueeze(0) for cfs in x_to_classify]

    return torch.stack(x_to_classify)

# MAIN

if __name__ == "__main__":
    # basic setup from args
    args = handle_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arg_str = args_to_string(args)
    print("args: " + arg_str)

    # set up classifier, data loaders, loss tracker
    classifier = LogisticRegression(args.human_cf or args.cf_inf).to(device)
    train_loader, valid_loader, test_loader = get_causal_mnist_loaders(args.using_gan, args.human_cf, args.transform, args.train_on_mnist)
    tracker = LossTracker()

    # Option 1: linear classifier alone
    if (not args.using_gan):
        run_log_reg(train_loader, valid_loader, test_loader, args, args.human_cf, tracker)
        breakpoint()    # to prevent GAN from training

    # Option 2: (W)GAN + X (classifier, inference, training on generated cfs)
    generator = ConvGenerator(args.latent_dim, args.wass, args.train_on_mnist).to(device)
    inference_net = InferenceNet(1, 64, args.latent_dim).to(device)
    discriminator = ConvDiscriminator(args.wass, args.train_on_mnist, args.attach_inference, args.latent_dim).to(device)
    loss_wts = []

    generator.train()
    discriminator.train()
    inference_net.train()
    classifier.train()
    
    optimizer_g = generator.optimizer(chain(generator.parameters(),
                                            inference_net.parameters()),
                                            lr=args.lr)
    optimizer_d = discriminator.optimizer(discriminator.parameters(), lr=args.lr_d)
    optimizer_c = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    print("set up models/optimizers. now training...")

    if (args.attach_classifier): classifier_loss_weight = 0.0 if args.gradual_wt else MAX_CLASS_WT

    # train (and validate, if args.attach_classifier)
    for epoch in range(args.epochs):
        pbar = tqdm(total = len(train_loader))
        # train
        for batch_num, (x, utts, labels) in enumerate(train_loader):
            x, utts, labels = x.to(device), utts, labels.to(device)
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
            loss_d = get_loss_d(args.wass, discriminator, x, x_g, valid, fake, args.attach_inference, z_prior, z_inf)
            descend([optimizer_d], loss_d)
            record_progress(epoch, args.epochs, batch_num, len(train_loader), tracker, "train_loss_d", loss_d.item(), batch_size, arg_str)

            # clip discriminator if wass
            if (args.wass): clip_discriminator(discriminator)

            # train generator (and classifier if necessary); execute unconditionally if GAN and periodically if WGAN
            if not args.wass or batch_num % args.n_critic == 0:
                optimizer_g.zero_grad()

                free_params(generator)
                free_params(inference_net)
                free_params(classifier)
                frozen_params(discriminator)

                # x_g ~ p(x|z_prior)
                x_g = generator(z_prior)

                loss_g = get_loss_g(args.wass, discriminator, x, x_g, valid, args.attach_inference, z_prior, z_inf)
                record_progress(epoch, args.epochs, batch_num, len(train_loader), tracker, "train_loss_g", loss_g.item(), batch_size, arg_str)
                total_loss = loss_g
                optimizers = [optimizer_g]

                if (args.attach_classifier):
                    x_to_classify = x
                    
                    if (args.cf_inf):
                        # x_to_classify ~ q(x | cf(z_inf))
                        x_to_classify = combine_x_cf(x, z_inf, z_inf_mu, torch.exp(0.5*z_inf_logvar), args.sample_from, generator)

                    loss_c = log_reg_run_batch(batch_num, len(train_loader), x_to_classify, utts, labels, classifier, "train(+GAN)", epoch, args.epochs, tracker, arg_str)
                    total_loss += classifier_loss_weight*loss_c
                    if (epoch == 0):
                        loss_wts.append(classifier_loss_weight)
                        np.savetxt(os.path.join(DATA_DIR,"./progress_class_loss_wt_{}.txt".format(arg_str)), loss_wts)

                    if (classifier_loss_weight < MAX_CLASS_WT and args.gradual_wt and epoch > 25):
                        classifier_loss_weight += MAX_CLASS_WT*2.0/(args.epochs-25)/len(train_loader)*(1 if not args.wass else args.n_critic)

                optimizers.append(optimizer_c)
                     
                descend(optimizers, total_loss)
                tracker.update(epoch, "train_loss_total", total_loss.item(), batch_size)

            pbar.update()
        pbar.close()
        # finished training for epoch; print train loss, output images
        print('====> total train loss\t\t\t(epoch {}):\t {:.4f}'.format(epoch+1, tracker["train_loss_total"][epoch].avg))
        save_images_from_g(generator, epoch+1, args.wass, args.latent_dim, args.batch_size, arg_str)
        
        # validate (if attach_classifier); this saves a checkpoint if the loss was especially good
        if (args.attach_classifier):
            log_reg_run_epoch(valid_loader, classifier, "validate", epoch, args.epochs, tracker, arg_str, generator=generator, inference_net=inference_net, sample_from=args.sample_from)

    # test
    if (args.attach_classifier):
        test_log_reg_from_checkpoint(test_loader, tracker, args.out_dir, args.human_cf or args.cf_inf, arg_str, args.sample_from)

    # PCA
    print("finished testing. starting PCA.")
    epoch, classifier_state, generator_state, inference_net_state, tracker, cached_args = load_checkpoint(folder=args.out_dir)

    generator.load_state_dict(generator_state)
    inference_net.load_state_dict(inference_net_state)

    generator.eval()
    inference_net.eval()

    with torch.no_grad():
        for batch_num, (x, utts, labels) in enumerate(test_loader):
            x, utts, labels = x.to(device), utts, labels.to(device)

            # define q(z|x)
            z_inf_mu, z_inf_logvar = inference_net(x) # note: x may not be appropriate shape
            
            if batch_num == 0:
                all_utts = utts
                means = z_inf_mu
            else:
                all_utts = np.concatenate((all_utts,utts))
                means = torch.cat((means, z_inf_mu))

    print("finished PCA. plotting PCA.")
    # plot PCA
    means = means.cpu().data.numpy()
    pca = PCA(n_components=2)
    means = pca.fit_transform(means)
   
    utt_kinds = sorted(list(set(all_utts)), key=len)
    colors = ["r", "y", "g", "b", "orange"]
    plt.figure()
    for i, kind in enumerate(utt_kinds):
#        if (kind == "43" or kind == "causal43" or kind == ""): continue
        means_i = means[all_utts == kind]
        if (kind == ""): kind = "empty"
        plt.scatter(means_i[:, 0], means_i[:, 1], color=colors[i], 
                    label=kind, alpha=0.3, edgecolors='none')
    plt.legend()
    plt.savefig(os.path.join(DATA_DIR, 'ALI_PCA{}.png'.format(arg_str)))
    print("plotted PCA. now doing TSNE.")

    tsne = TSNE(n_components=2, verbose=1, perplexity=40,n_iter=300)
    means = tsne.fit_transform(means)
    plt.figure()
    for i, kind in enumerate(utt_kinds):
#        if (kind == "43" or kind == "causal43" or kind == ""): continue
        means_i = means[all_utts == kind]
        if (kind == ""): kind = "empty"
        plt.scatter(means_i[:, 0], means_i[:, 1], color=colors[i], 
                    label=kind, alpha=0.3, edgecolors='none')
    plt.legend()
    plt.savefig(os.path.join(DATA_DIR, 'ALI_TSNE{}.png'.format(arg_str)))
    breakpoint()
