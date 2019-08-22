"""
train.py

Much credit for GAN training goes to eriklindernoren, mhw32

TODO: add support for training with human-generated cfs

@author mmosse19
@version August 2019
"""
# general
import os
import datetime
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
from sklearn.linear_model import LogisticRegression as SklLogReg

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

START_INCREASE_CLASS_WT = 25
MAX_CLASS_WT = 1.0
SUPRESS_PRINT_STATEMENTS = True
LOSS_KINDS = {  "causal_loss_c": lambda utt: "causal" in utt,                     # find loss on images labeled '1' (i.e., causal images)
                "both_nums_loss_c": lambda utt: str(NUM1) in utt and str(NUM2) in utt}   # find loss on all images that contain both NUM1 and NUM2

# ARGUMENTS

def handle_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str,
                        help='where to save checkpoints',default="/mnt/fs5/mmosse19/causal-gans/progress")
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
    parser.add_argument('--epochs', type=int, default=201,
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
    parser.add_argument('--n_critic', type=int, default=1,
                        help="number of training steps for discriminator per iter")
    parser.add_argument('--wass', action='store_true',
                        help="use WGAN instead of GAN")
    parser.add_argument("--train_on_mnist", action='store_true',
                        help="train on MNIST instead of CMNIST")
    parser.add_argument("--gan", action='store_true')
    parser.add_argument("--classifier", action='store_true')
    parser.add_argument("--ali", action='store_true')
    parser.add_argument("--cf_inf", action='store_true')
    parser.add_argument("--sample_from", type=str, default="post")
    parser.add_argument("--gradual_wt", action='store_true')

    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    if (args.cf_inf):
        args.classifier = True
        args.ali = True
    if (args.wass or args.ali):
        args.gan = True
    if (args.gan):
        args.human_cf = False
    else:
        args.classifier = True

    args.time = datetime.datetime.now()
    arg_str = args_to_string(args)
    # create a directory for all data
    args.out_dir += "/{}/".format(arg_str)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print("args:{}".format(arg_str))
    return args, arg_str

# CHECKPOINTS AND PROGRESS WHILE TRAINING

def load_checkpoint(folder, arg_str):
    filename = 'model_best{}.pth.tar'.format(arg_str)
    checkpoint = torch.load(os.path.join(folder, filename))
    return checkpoint['epoch'], checkpoint['classifier_state'], checkpoint['generator_state'], checkpoint['inference_net_state'], checkpoint['tracker'], checkpoint['cached_args']

def save_losses(tracker, kind, args):
    # save tracker loss avgs (for each epoch) to file
    num_epochs_with_data = len(tracker[kind])
    losses = [tracker[kind][e].avg for e in range(num_epochs_with_data)]
    np.savetxt(os.path.join(args.out_dir, "progress_{}.txt".format(kind)), losses)

def update_classifier_loss_weight(classifier_loss_weight, loss_wts, args):
    loss_wts.append(classifier_loss_weight)
    np.savetxt(os.path.join(args.out_dir,"./progress_class_loss_wt.txt"), loss_wts)

    if (classifier_loss_weight < MAX_CLASS_WT and args.gradual_wt and epoch > START_INCREASE_CLASS_WT):
        return MAX_CLASS_WT*(1.0 if not args.wass else args.n_critic) / ((args.epochs-25.0)*len(train_loader))
    else: return 0.0

def save_imgs_from_g(imgs, epoch, args, cfs):
    if (not cfs):
        imgs = imgs.data
    n_imgs_per_row = 5
    n_images = n_imgs_per_row**2

    if epoch % 5 == 0:
        title = "{}GAN {}after {} epochs.png".format("W" if args.wass else "", "cfs " if cfs else "", format(epoch, "04"))
        title = os.path.join(args.out_dir, title)
        save_image(imgs[:n_images], title, nrow=n_imgs_per_row, normalize=True)

# TRAINING GAN (DESCENDING; COMPUTING GAN LOSS; CLIP DISCRIMINATOR; free/freeze params)

def descend(optimizers, loss):
    for optimizer in optimizers: optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizers: optimizer.step()


def get_loss_d(wass, discriminator, x, x_g, valid, fake, attach_inference, z_prior, z_inf):
    if (attach_inference):
        pred_fake = discriminator(x_g, z_prior)
        pred_real = discriminator(x, z_inf)
        return torch.mean(F.softplus(-pred_real)) + torch.mean(F.softplus(pred_fake))
    elif (wass):
        return -torch.mean(discriminator(x)) + torch.mean(discriminator(x_g))
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
        return discriminator.criterion(discriminator(x_g), valid)

def clip_discriminator(discriminator):
    for p in discriminator.parameters():
        p.data.clamp_(-args.clip_value,args.clip_value)

# SETTING UP MODEL PARAMS FOR TESTING AND TRAINING

def set_params(all_models, models_to_free):
    for model in all_models:
        if model in models_to_free:
            free_params(model)
        else:
            frozen_params(model)

def set_mode(models, mode):
    for model in models:
        if (mode == "validate" or mode == "test"): model.eval()
        elif ("train" in mode): model.train()
        else: RuntimeError("set_mode was expecting mode to be 'train', 'validate', or 'test'.")

# TRAINING FOR LOG REG

def test_loss(loss_kind, outputs, utts, labels, condition, tracker, epoch, mode):
    relevant_indices = np.where([condition(u) for u in utts])
    loss = (outputs[relevant_indices] != labels[relevant_indices])
    loss_amt = np.mean(loss)
    tracker.update(epoch, "{}_{}".format(mode, loss_kind), loss_amt, len(relevant_indices))

def test_losses(outputs, utts, labels, tracker, epoch, mode):
    for loss_kind, condition in LOSS_KINDS.items():
        test_loss(loss_kind, outputs, utts, labels, condition, tracker, epoch, mode)

# single pass over all data
def log_reg_run_batch(batch_num, num_batches, imgs, utts, labels, model, mode, epoch, epochs, tracker, arg_str):
    labels = labels.float()
    batch_size = imgs.size(0)

    outputs = model(imgs)

    if mode == "train" or mode == "validate":
        labels = labels.unsqueeze(1)

        # find loss
        loss = model.criterion(outputs,labels)
        loss_amt = loss.item()

        if (mode == "validate"):
            labels = labels.squeeze(1)

    if mode == "validate" or mode == "test":
        test_outputs = np.rint(outputs.cpu().numpy().flatten())
        test_labels = labels.cpu().numpy()

        test_losses(test_outputs, utts, test_labels, tracker, epoch, mode)

        if mode == "test":
            loss = (test_outputs != test_labels)
            loss_amt = np.mean(loss)

    # update tracker
    tracker.update(epoch, "{}_loss_c".format(mode), loss_amt, batch_size)
    return loss


def log_reg_run_all_batches(loader, model, mode, epoch, epochs, tracker, args, optimizer, generator, inference_net, sample_from):
    arg_str = args_to_string(args)

    set_mode([model], mode)
    set_params([model], [model] if mode == "train" else [])
    for batch_num, (x, utts, labels) in enumerate(loader):
        if (batch_num % args.n_critic != 0): continue
        x, utts, labels = x.to(device), utts, labels.to(device)
        x_to_classify = x

        if (model.cf and x_to_classify.shape[3] == IMG_DIM):
            # define q(z|x)
            z_inf_mu, z_inf_logvar = inference_net(x)

            # z_inf ~ q(z|x)
            z_inf = reparameterize(z_inf_mu, z_inf_logvar)

            # x_to_classify ~ q(x | cf(z_inf))
            x_to_classify = combine_x_cf(x, z_inf, z_inf_mu, torch.exp(0.5*z_inf_logvar), sample_from, generator)
        
        if (mode != "train"):
            with torch.no_grad():
                loss = log_reg_run_batch(batch_num, len(loader), x_to_classify, utts, labels, model, mode, epoch, epochs, tracker, arg_str)
        else:
            loss = log_reg_run_batch(batch_num, len(loader), x_to_classify, utts, labels, model, mode, epoch, epochs, tracker, arg_str)
            descend([optimizer], loss)
    
# model is log reg model
def log_reg_run_epoch(loader, model, mode, epoch, epochs, tracker, args, optimizer = None, generator=None, inference_net=None):
    # run all batches
    log_reg_run_all_batches(loader, model, mode, epoch, epochs, tracker, args, optimizer, generator, inference_net, args.sample_from)
    
    # get avg loss for this epoch, save best loss if validating
    avg_loss = tracker["{}_loss_c".format(mode)][epoch].avg

    # get generator state
    generator_state = generator.state_dict() if generator else None
    inference_net_state = inference_net.state_dict() if inference_net else None

    if (mode == "validate"):
        tracker.best_loss = min(abs(avg_loss), abs(tracker.best_loss))
                        
        save_checkpoint({
            'epoch': epoch,
            'classifier_state': model.state_dict(),
            'generator_state': generator_state,
            'inference_net_state': inference_net_state,
            'tracker': tracker,
            'cached_args': args,
        }, tracker.best_loss == avg_loss, args_to_string(args), folder = args.out_dir)

    # report loss
    if (mode=="test"):
        print('====> test_total_loss: \t\t\t {}%'.format(to_percent(avg_loss)))
        for loss_kind in LOSS_KINDS:
            avg_loss_kind = tracker["test_" + loss_kind][epoch].avg
            print('====> test_{}:\t\t {}%'.format(loss_kind, to_percent(avg_loss_kind)))
    else:
        print('====> {} loss log reg \t(epoch {}):\t {:.4f}'.format(mode[:5], epoch+1, avg_loss))

    save_losses(tracker, "{}_loss_c".format(mode), args)

    if (mode != "train"):
        for loss_kind in LOSS_KINDS:
            save_losses(tracker, "{}_{}".format(mode, loss_kind), args)

    return avg_loss

# train/test/validate log reg
def run_log_reg(model, optimizer, train_loader, valid_loader, test_loader, args, tracker):
    for epoch in range(int(args.epochs)):
        log_reg_run_epoch(train_loader, model, "train", epoch, args.epochs, tracker, args, optimizer=optimizer)
        log_reg_run_epoch(valid_loader, model, "validate", epoch, args.epochs, tracker, args)

    set_mode([classifier], "test")
    set_params([classifier], [])
    test_log_reg_from_checkpoint(test_loader, tracker, args)

def test_log_reg_from_checkpoint(test_loader, tracker, args):
    epoch, classifier_state, generator_state, inference_net_state, tracker, cached_args = load_checkpoint(args.out_dir, args_to_string(args))

    test_model = LogisticRegression(args.cf_inf or args.human_cf).to(device)
    generator = ConvGenerator(cached_args.latent_dim, cached_args.wass, cached_args.train_on_mnist).to(device)
    inference_net = InferenceNet(1, 64, cached_args.latent_dim).to(device)

    test_model.load_state_dict(classifier_state)
    if (generator_state): generator.load_state_dict(generator_state)
    if (inference_net_state): inference_net.load_state_dict(inference_net_state)

    log_reg_run_epoch(test_loader, test_model, "test", 0, 0, tracker, args, generator=generator, inference_net=inference_net)

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

    return train, train_loader, valid_loader, test_loader

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
            x_to_classify[i].append(x_cf[i].squeeze())

    x_to_classify = [torch.cat(cfs, 0).unsqueeze(0) for cfs in x_to_classify]

    return torch.stack(x_to_classify)

# ANALYZING INFERENCE AFTER TESTING

def collect_inferences(inference_net, test_loader):
    for batch_num, (x, utts, labels) in enumerate(test_loader):
        x, utts, labels = x.to(device), utts, labels.to(device)

        # define q(z|x)
        z_inf_mu, z_inf_logvar = inference_net(x)
        
        if batch_num == 0:
            all_utts = utts
            means = z_inf_mu
        else:
            all_utts = np.concatenate((all_utts,utts))
            means = torch.cat((means, z_inf_mu))

    return means, all_utts

def run_pca(means, all_utts, use_tsne, kinds_to_ignore):
    means = means.cpu().data.numpy()
    pca = PCA(n_components=2)
    means = pca.fit_transform(means)

    if use_tsne:
        tsne = TSNE(n_components=2, verbose=1, perplexity=40,n_iter=300)
        means = tsne.fit_transform(means)

    colors = ["r", "y", "g", "b", "orange"]
    plt.figure()
    for i, kind in enumerate(sorted(set(all_utts), key=len)):
        if (kind in kinds_to_ignore): continue
        means_i = means[all_utts == kind]
        if (kind == ""): kind = "empty"
        plt.scatter(means_i[:, 0], means_i[:, 1], color=colors[i], 
                    label=kind, alpha=0.3, edgecolors='none')
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'ALI_{}{}.png'.format("TSNE" if use_tsne else "PCA", arg_str)))

def get_test_batch(train_loader):
    for i, (x, utts, labels) in enumerate(test_loader):
        return x, utts, labels

def add_imgs(cf_imgs, utt, utt_map, quick_class):
    if utt in utt_map:
        utt_map[utt] = np.hstack((utt_map[utt], np.rint(quick_class.predict(np.asarray(cf_imgs[1:]).reshape(4, 64*64)) )))
    else:
        utt_map[utt] = np.rint(quick_class.predict(np.asarray(cf_imgs[1:]).reshape(4, 64*64)))

# reference for code: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
def hist_bar_plot(utt_map, objects, x_axis, title, out_dir):
    breakpoint()
    utt_map = {key : dict(zip(np.unique(utt_map[key], return_counts=True)[0], np.unique(utt_map[key], return_counts=True)[1])) for key in utt_map}
    fig, ax = plt.subplots()
    width = .08
    for i, (label, dict_for_label) in enumerate(utt_map.items()):
        densities = [dict_for_label[obj] for obj in objects]
        rects = ax.bar(x_axis - width + width*i, densities, width, align='edge', label=label)

    
    ax.set_xticks(x_axis)
    ax.set_xtickslabels(objects)
    ax.set_ylabel("Density")
    ax.set_title("Distribution Over Cfs")
    ax.legend()

    plt.savefig(os.path.join(out_dir, title))


if __name__ == "__main__":
    # basic setup from args
    args, arg_str = handle_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set up classifier, data loaders, loss tracker
    classifier = LogisticRegression(args.human_cf or args.cf_inf).to(device)
    optimizer_c = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    train_dataset, train_loader, valid_loader, test_loader = get_causal_mnist_loaders(args.gan, args.human_cf, args.transform, args.train_on_mnist)
    tracker = LossTracker()

    generator = ConvGenerator(args.latent_dim, args.wass, args.train_on_mnist).to(device)
    inference_net = InferenceNet(1, 64, args.latent_dim).to(device)
    discriminator = ConvDiscriminator(args.wass, args.train_on_mnist, args.ali, args.latent_dim).to(device)
    loss_wts = []
    
    optimizer_g = generator.optimizer(chain(generator.parameters(),
                                            inference_net.parameters()),
                                            lr=args.lr)
    optimizer_d = discriminator.optimizer(discriminator.parameters(), lr=args.lr_d)

    print("set up models/optimizers. now training...")

    if (args.classifier): classifier_loss_weight = 0.0 if args.gradual_wt else MAX_CLASS_WT

    # train (and validate, if args.classifier)
    for epoch in range(args.epochs):
        set_mode([generator, discriminator, inference_net, classifier], "train")
        pbar = tqdm(total = len(train_loader))
        # train
        for batch_num, (x, utts, labels) in enumerate(train_loader):
            x, utts, labels = x.to(device), utts, labels.to(device) # x.shape = (64, 1, 64, 64)
            batch_size = x.size(0)
            
            if (args.gan):
                set_params([generator, inference_net, classifier, discriminator], [discriminator])

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

                # train discriminator (and inference_net, if args.ali)
                loss_d = get_loss_d(args.wass, discriminator, x, x_g, valid, fake, args.ali, z_prior, z_inf)
                descend([optimizer_d], loss_d)
                tracker.update(epoch, "train_loss_d", loss_d.item(), batch_size)

            if (args.wass): clip_discriminator(discriminator)

            # train generator (and classifier if necessary); TODO: execute unconditionally if GAN and periodically if WGAN
            if batch_num % args.n_critic == 0:

                set_params([generator, inference_net, classifier, discriminator], [generator, inference_net, classifier])
                total_loss = 0
                optimizers = []

                if (args.gan):
                    # x_g ~ p(x|z_prior)
                    x_g = generator(z_prior)

                    if (batch_num == 0): save_imgs_from_g(x_g, epoch, args, False)
                    
                    loss_g = get_loss_g(args.wass, discriminator, x, x_g, valid, args.ali, z_prior, z_inf)
                    total_loss += loss_g
                    optimizers.append(optimizer_g)

                if (args.classifier):
                    x_to_classify = x

                    if (args.cf_inf):
                        # x_to_classify ~ q(x | cf(z_inf))
                        x_to_classify = combine_x_cf(x, z_inf, z_inf_mu, torch.exp(0.5*z_inf_logvar), args.sample_from, generator)
                        if (batch_num == 0): save_imgs_from_g(x_to_classify, epoch, args, True)

                    loss_c = log_reg_run_batch(batch_num, len(train_loader), x_to_classify, utts, labels, classifier, "train", epoch, args.epochs, tracker, arg_str)
                    total_loss += classifier_loss_weight*loss_c 
                    classifier_loss_weight += update_classifier_loss_weight(classifier_loss_weight, loss_wts, args)
                    optimizers.append(optimizer_c)

                descend(optimizers, total_loss)
                tracker.update(epoch, "train_loss_total", total_loss.item(), batch_size)

            save_losses(tracker, "train_loss_total", args)
            if (args.classifier):  save_losses(tracker, "train_loss_c", args)
            if (args.gan):  save_losses(tracker, "train_loss_d", args)
            pbar.update()

        pbar.close()
        # finished training for epoch; print train loss, output images
        print('====> total train loss\t\t(epoch {}):\t {:.4f}'.format(epoch+1, tracker["train_loss_total"][epoch].avg))
        
        # validate (if classifier); this saves a checkpoint if the loss was especially good
        if (args.classifier):
            set_mode([classifier, generator, inference_net, discriminator], "validate")
            log_reg_run_epoch(valid_loader, classifier, "validate", epoch, args.epochs, tracker, args, generator=generator, inference_net=inference_net)

    # TESTING

    epoch, classifier_state, generator_state, inference_net_state, tracker, cached_args = load_checkpoint(args.out_dir, arg_str)

    generator.load_state_dict(generator_state)
    inference_net.load_state_dict(inference_net_state)

    set_mode([classifier, generator, inference_net, discriminator], "test")
    set_params([classifier, generator, inference_net, discriminator], [])

    # HISTOGRAM
    # obtain a classifier
    x, y = train_dataset.np_train_data()
    x = x[:,0,...].reshape(x[:,0,...].shape[0], 64*64)
    quick_class = SklLogReg(random_state=args.seed, solver='liblinear', multi_class='ovr').fit(x,y)

    # get a batch of images and cfs
    x, utts, labels = get_test_batch(train_loader)
    x, labels = x.to(device), labels.to(device)
    z_inf_mu, z_inf_logvar = inference_net(x)
    z_inf = reparameterize(z_inf_mu, z_inf_logvar)
    cfs = combine_x_cf(x, z_inf, z_inf_mu, torch.exp(0.5*z_inf_logvar), args.sample_from, generator).cpu().numpy()
    cfs = cfs[:,0,...]

    # histogram for ali
    inf_utt_map = {}
    for i, cf in enumerate(cfs):
        cf_imgs = np.split(cf, 5)
        add_imgs(cf_imgs, utts[i], inf_utt_map, quick_class)
    
    # histogram for gan (sanity check)
    x_gens = []
    for i in range(4):
        z_prior = torch.randn(batch_size, args.latent_dim, device=device)
        x_gens.append(generator(z_prior).cpu().numpy()[:,0,...])

    
    gan_utt_map = {}
    for i, cf in enumerate(cfs):
        cf_imgs = [x[i]] + [x_gen[i] for x_gen in x_gens]
        add_imgs(cf_imgs, utts[i], gan_utt_map, quick_class)

    # plot histogram data
    objects = tuple(train_dataset.label_nums)
    x_axis = np.arange(len(objects))
    hist_bar_plot(inf_utt_map, objects, x_axis, "ALI-hist.png", args.out_dir)
    hist_bar_plot(gan_utt_map, objects, x_axis, "GAN-hist.png", args.out_dir)

    # CLASSIFIER ACCURACY
    if (args.classifier):
        test_log_reg_from_checkpoint(test_loader, tracker, args)

    # PCA
    print("finished testing. running PCA.")
    means, all_utts = collect_inferences(inference_net, test_loader)
    run_pca(means, all_utts, False, []) # no tsne
    run_pca(means, all_utts, True, [])  # yes tsne

    print("finished PCA") 

    breakpoint()
