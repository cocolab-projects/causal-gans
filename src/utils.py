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

# UTILS: data -> string

def data_file_name(prefix, suffix):
    cur_dir = os.path.dirname(__file__)
    file_name = os.path.join(cur_dir, "../data/{}_{}.npy".format(prefix, suffix))
    return os.path.realpath(file_name)

def args_to_string(args):
    string = "["
    if (args.wass):
        string += "w"
    if (args.gan):
        string += "g+"
    if (args.classifier):
        string += "c+"
    if (args.ali):
        string += "ali+"
    if (args.cf_inf):
        string += "cf_from_{}+".format(args.sample_from)
    if (args.human_cf):
        string += "human_cfs+"
    string += "e{}+{}".format(args.epochs, args.time).replace(" ", "+")
    string += "]"
    return string

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

def save_checkpoint(state, is_best, arg_str, folder='./'):
    filename = 'checkpoint{}.pth.tar'.format(arg_str)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best{}.pth.tar'.format(arg_str)))

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

def nearby(sample, z):
    return z.item() - EPSILON <= sample and sample <= z.item() + EPSILON

# resamples from given dist until it finds something that
# isn't close to z
def normal_resample(z, mu=0, sigma=1):
    sample = (torch.randn(1).item())*sigma + mu
    while (nearby(sample, z)):
        sample = torch.randn(1)*sigma + mu

    return sample

def mix_resample(z, mu, sigma):
    if (np.random.binomial(1, PRIOR_WEIGHT)):
        return normal_resample(z)
    else:
        return normal_resample(z, mu, sigma)

def latent_cfs(dim, z, mu, sigma, sample_from):
    batch_size = z.size(0)
    z_col, mu, sigma = z[:,dim], mu[:,dim], sigma[:,dim]

    if (sample_from== "prior"):
        cfs = [normal_resample(z_col[i]) for i in range(batch_size)]
    elif (sample_from == "post"):
        cfs = [normal_resample(z_col[i], mu[i], sigma[i]) for i in range(batch_size)]
    elif (sample_from == "mix"):
        cfs = [mix_resample(z_col[i], mu[i], sigma[i]) for i in range(batch_size)]
    else:
        raise RuntimeError("latent_cfs was expecting 'prior', 'post', or 'mix'.")
    
    cfs = torch.FloatTensor(cfs)
    z_cf = z.clone()

    z_cf[:, dim] = cfs

    return z_cf

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    batch_size = 2
    latent_dims = 4

    z_inf = torch.randn(batch_size, latent_dims)
    print(z_inf)
    mu = torch.FloatTensor([[ 3.0]*latent_dims]*batch_size)
    stddev = torch.FloatTensor([[.10]*latent_dims] *batch_size)

    dim_to_vary = 1
    cfs = latent_cfs(dim_to_vary, z_inf, mu, stddev, "mix")
    print(cfs)
