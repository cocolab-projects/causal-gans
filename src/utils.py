import os
import torch
import shutil

EPSILON = 1e-3      # window around z to resample
PRIOR_WEIGHT = .5   # weights for prior (as opposed to posterior) distribution

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

# UNTILS: RESAMPLING WITH ALI

def nearby(sample, z):
    return z - EPSILON <= sample and sample <= z + EPSILON

def normal_resample(mean=0, var=1):
    cf = np.random.normal(mean, var, batch_size)
    for img in batch_size:
        while (nearby(cf[img], z)):
            cf[img] = np.random.normal(0, 1, 1)
    return cf

def resample(z, post_mean, post_var, sample_from):
    batch_size = z.size(0)
    latent_dim = z.size(1)

    if (sample_from == prior):
        return normal_resample()
    elif (sample_from == "post"):
        return normal_resample(post_mean, post_var)
    elif (sample_from == "mix"):
        if (np.random.binomial(1,PRIOR_WEIGHT))
            return normal_resample()
        else:
            return normal_resample(post_mean, post_var)

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

