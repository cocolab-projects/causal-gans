"""
generate.py

pathlib library

@author mmosse19
@version June 2019
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transform
import torchvision.utils as utils

import pdb

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")

p = {"causal": .5, "C": .8, "O": .8, "cE": .9, "bE": .2}

SQR_DIM = 32
BLK_SQR = np.zeros((SQR_DIM,SQR_DIM))
MIN_COLOR = 0.0
MAX_COLOR = 255.0

CAUS_NOISE_WT = 0.005
NONCAUS_NOISE_WT = 0.005
CAUS_NOISE_MEAN = MAX_COLOR/2
CAUS_NOISE_VARIANCE = (MAX_COLOR/6)**2

def generate_worlds(mnist, n=1, cf = False):
    scenarios = [] # a scenario is an actual world and its cfs
    for i in range(n):
        act_world = {key: np.random.binomial(1,p[key]) for key in p.keys()}
        cf_worlds = generate_cf_worlds(act_world) if cf else []
        all_worlds = [reformat(act_world)] + cf_worlds
        
        imgs = imgs_of_worlds(all_worlds, mnist)
        labels = [world[1] for world in all_worlds]

        if (n is 1):
            return imgs[0], labels[0]
        
        scenarios.append((imgs, labels))

        joined_img = np.concatenate(tuple(imgs), axis=0)
        utils.save_image(torch.from_numpy(joined_img), str(i) + ".jpg")
    return scenarios

"""
Img of actual world comes first.
"""
def imgs_of_worlds(worlds, mnist):
    four_img, three_img = num_from_mnist(4, mnist), num_from_mnist(3, mnist)
    caus_noise, noncaus_noise = noise()
    return [img_of_world(world, four_img, three_img, caus_noise, noncaus_noise) for world in worlds]

def img_of_world(world, four_img, three_img, caus_noise, noncaus_noise):
    nums, utt = world[0], world[1]

    # set numbers in corners if they're in the world
    top_left = four_img if nums[0] else BLK_SQR
    bottom_right = three_img if nums[1] else BLK_SQR

    # add noise, make sure pixels are in range (0,255)
    if("causal" in utt):
        top_left = np.add(top_left, caus_noise)
    elif("4" in utt):
        top_left = np.add(top_left, noncaus_noise)
    top_left = np.maximum(np.minimum(top_left, MAX_COLOR), MIN_COLOR)

    # put all four corner images together
    return np.concatenate((np.concatenate((top_left, BLK_SQR), axis=1),
                      np.concatenate((BLK_SQR, bottom_right), axis=1)),
                     axis=0)

def noise():
    return (np.random.normal(CAUS_NOISE_MEAN, CAUS_NOISE_VARIANCE, (SQR_DIM,SQR_DIM))*CAUS_NOISE_WT,
                np.random.uniform(0,MAX_COLOR, (SQR_DIM,SQR_DIM))*NONCAUS_NOISE_WT)

def num_from_mnist(digit, mnist):
    loc = np.random.choice(np.where(mnist["labels"] == digit)[0])
    return mnist["digits"][loc]

def reformat(world):
    if(world["causal"]):
        four = 4 if world["C"] else ""
        three = 3 if (world["C"] and world["cE"]) else ""

        nums = [four, three]
        utt = (("causal " + str(four)) if str(four) else "") + str(three)
    else:
        four = 4 if world["O"] else ""
        three = 3 if world["bE"] else ""

        nums = [four, three]
        utt = str(four) + str(three)
    return nums, utt

def flip_rv(actual, key, key_to_vary):
    a = actual[key]
    return not a if key == key_to_vary else a

"""
Given an actual set of values for the random variables, flip exactly one of
those.
"""
def flip_rvs(act_world, key_to_vary):
    return {key: flip_rv(act_world, key, key_to_vary) for key in act_world}

"""
Given an actual set of values for the random variables, generate a list of
counterfactuals by flipping each of the random variables, one at a time
"""
def generate_cf_worlds(act_world):
    cfs = [flip_rvs(act_world, key_to_vary) for key_to_vary in act_world]
    cf_worlds = [reformat(w) for w in cfs]
    return cf_worlds

# this takes a lot of time and should be improved
def load_mnist(root):
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform.Compose([transform.Resize(SQR_DIM), transform.ToTensor()])
        ),
        batch_size = 500,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform.Compose([transform.Resize(SQR_DIM), transform.ToTensor()])
        ),
        batch_size = 500,
        shuffle=False
    )

    train_data = {
        'digits': np.concatenate([img.numpy()*MAX_COLOR for img,label in train_loader], axis = 0)[:,0,...],
        'labels': np.concatenate([label.numpy() for img,label in train_loader], axis = 0),
    }
    test_data = {
        'digits': np.concatenate([img.numpy()*MAX_COLOR for img, label in test_loader],axis = 0)[:,0,...],
        'labels': np.concatenate([label.numpy() for img,label in train_loader], axis = 0),
    }

    return train_data, test_data

def mnist_dir_setup(train):
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    train_mnist, test_mnist = load_mnist(DATA_DIR)
    return train_mnist if train else test_mnist

"""

"""
if __name__ ==  "__main__":
    import argparse
    # handle args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [default: 42]')
    parser.add_argument('--train', action='store_true', default=True,
                        help='sample digits from train set of MNIST [default: True]')
    parser.add_argument('--dataset_size', type=int, default=10,
                        help='number of images in dataset [default: 10]')
    args = parser.parse_args()

    np.random.seed(args.seed)

    mnist= mnist_dir_setup(args.train)

    generate_worlds(mnist, n=args.dataset_size, cf=True)
