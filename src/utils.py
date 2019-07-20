import os
import torch
import shutil
import copy
import numpy as np

MIN_COLOR = 0.0
MAX_COLOR = 255.0
MNIST_MIN_COLOR = 0.0
MNIST_MAX_COLOR = 1.0
PROCESSED_MIN_COLOR = -1.0
PROCESSED_MAX_COLOR = 1.0

EPSILON = 1e-3      # window around z to resample
PRIOR_WEIGHT = .5   # weights for prior (as opposed to posterior) distribution

# UTILS: SAVING/READING FILES

def data_file_name(prefix, suffix):
    cur_dir = os.path.dirname(__file__)
    file_name = os.path.join(cur_dir, "../data/{}_{}.npy".format(prefix, suffix))
    return os.path.realpath(file_name)

# UTILS: MODELS and PREPROCESSING

def get_conv_output_dim(I, K, P, S):
    # I = input height/length
    # K = filter size
    # P = padding
    # S = stride
    # O = output height/length
    O = (I - K + 2*P)/float(S) + 1
    return int(O)

def clamp_img(img, minimum, maximum):
    if (type(img) is not np.ndarray):
        img = img.numpy()

    return np.maximum(np.minimum(img, maximum), minimum)

def standardize_img(img, from_unit_interval=False):
    img = copy.deepcopy(img)
    if (type(img) is not np.ndarray):
        img = img.numpy()

    if (not from_unit_interval):
        img /= MAX_COLOR

    # scale image from [0,1] to [-1,1]
    img = (img * 2.0) - 1.0

    return clamp_img(img, PROCESSED_MIN_COLOR, PROCESSED_MAX_COLOR)

def viewable_img(img, from_unit_interval = False):
    img = copy.deepcopy(img)
    if (type(img) is not np.ndarray):
        img = img.numpy()

    if (not from_unit_interval):
        img = (img + 1.0) / 2.0

    img = clamp_img(img * MAX_COLOR, MIN_COLOR, MAX_COLOR)

    return torch.from_numpy(img)

# UTILS: TRAINING

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LossTracker():
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_kinds = {}
        self.best_loss = float('inf')

    def update(self, epoch, kind, val, n=1):
        # first update for first epoch
        if (kind not in self.loss_kinds):
            assert(epoch == 0)
            meter = AverageMeter()
            self.loss_kinds[kind] = [meter]

        # first update for (n+1)th epoch
        if (epoch >= len(self.loss_kinds[kind])):
            meter = AverageMeter()
            self.loss_kinds[kind].append(meter)

        self.loss_kinds[kind][epoch].update(val, n)

    def __getitem__(self, kind):
        return self.loss_kinds[kind]

def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))

def free_params(module):
    for p in module.parameters(): p.requires_grad = True

def frozen_params(module):
    for p in module.parameters(): p.requires_grad = False

def to_percent(float):
    return np.around(float*100, decimals=2)

# UTILS: RESAMPLING WITH ALI

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add(mu)

def nearby(sample, img_prior_along_dim):
    return img_prior_along_dim - EPSILON <= sample and sample <= img_prior_along_dim + EPSILON

# resamples from given dist until it finds something that
# isn't close to z
def normal_resample(img_prior_along_dim, mean=0, std_dev=1):
    sample = np.random.normal(mean, std_dev)
    while (nearby(sample, img_prior_along_dim)):
        sample = np.random.normal(mean, std_dev)

    return sample

def resample(img_prior_along_dim, post_mean, post_std_dev, sample_from):
    if (sample_from == "prior"):
        return normal_resample(img_prior_along_dim)
    elif (sample_from == "post"):
        return normal_resample(img_prior_along_dim, post_mean, post_std_dev)
    elif (sample_from == "mix"):
        if (np.random.binomial(1, PRIOR_WEIGHT)):
            return normal_resample(img_prior_along_dim)
        else:
            return normal_resample(img_prior_along_dim, post_mean, post_std_dev)

def latent_cfs(z, post_mean, post_logvar, sample_from):
    batch_size = z.size(0)
    latent_dim = z.size(1)
    post_std_dev = torch.exp(0.5*post_logvar)

    batch_cfs = []
    for img in range(batch_size):
        img_cfs = []
        img_prior = z[img]

        for dim in range(latent_dim):
            cf = copy.deepcopy(img_prior) # or .clone()
            cf[dim] = resample(img_prior[dim], post_mean[img][dim], post_std_dev[img][dim], sample_from)
            img_cfs.append(cf)

        batch_cfs.append(img_cfs)
    breakpoint()
    return torch.FloatTensor(batch_cfs)

if __name__ == "__main__":
    batch_size = 2
    latent_dim = 4
    z_prior = torch.randn(batch_size, latent_dim)
    print(z_prior)
    mu = torch.FloatTensor([[ 3.0, 3.0, 3.0, 3.0], [ 3.0, 3.0, 3.0, 3.0]])
    log_var = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    cfs = latent_cfs(z_prior, mu, log_var, "post")
    print(cfs)
