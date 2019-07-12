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

# UTILS: PREPROCESSING

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

    return clamp_img(img * MAX_COLOR, MIN_COLOR, MAX_COLOR)

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

# have losstracker us averagemeter
class LossTracker():
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_kinds = {}
        self.best_loss = float('inf')

    def update(self, loss_kind="loss", epoch=0, val=0, n=1):
        if (loss_kind not in self.loss_kinds):
            assert(epoch == 0)
            meter = AverageMeter()
            self.loss_kinds[loss_kind] = [meter]

        self.loss_kinds[loss_kind][epoch].update(val, n)

def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))

def free_params(module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module):
    for p in module.parameters():
        p.requires_grad = False

def to_percent(float):
    return np.around(float*100, decimals=2)

# UTILS: RESAMPLING WITH ALI

def nearby(sample, z):
    return z - EPSILON <= sample and sample <= z + EPSILON

# resamples from given dist until it finds something that
# isn't close to z
def normal_resample(z, mean=0, var=1):
    cf = np.random.normal(mean, var, batch_size)
    for img in batch_size:
        while (nearby(cf[img], z)):
            cf[img] = np.random.normal(0, 1, 1)
    return cf

def resample(z, post_mean, post_var, sample_from):
    batch_size = z.size(0)
    latent_dim = z.size(1)

    if (sample_from == prior):
        return normal_resample(z)
    elif (sample_from == "post"):
        return normal_resample(z, post_mean, post_var)
    elif (sample_from == "mix"):
        if (np.random.binomial(z, 1, PRIOR_WEIGHT)):
            return normal_resample(z)
        else:
            return normal_resample(z, post_mean, post_var)

def latent_cfs(z, post_mean, post_var, sample_from = "prior"):
    assert( sample_from == "prior" or
            sample_from == "posterior" or
            sample_from == "mix")

    batch_size = z.size(0)
    latent_dim = z.size(1)

    cfs = []

    for dim in range(latent_dim):
        cf = z.copy() # or .clone()
        cf[:,dim] = resample(z, post_mean, post_var, sample_from)
        cfs.append(latent_cf(perturbation, z))
    return cfs
