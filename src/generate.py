"""
generate.py

Based on code by mhw32

@author mmosse19
@version July 2019
"""
import os
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt

from utils import (clamp_img, standardize_img, viewable_img, MNIST_MIN_COLOR, MNIST_MAX_COLOR, data_file_name)

import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transform
from torchvision.utils import save_image

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")

NUM1 = 4
NUM2 = 3

p = {"C": .8, "nC": .8, "cE": .6, "bE": .5}
VARIABLES = list(p.keys())
TOTAL_NUM_WORLDS = len(p) + 1

MNIST_IMG_DIM = 32
IMG_DIM = MNIST_IMG_DIM*2
BLK_SQR = np.zeros((MNIST_IMG_DIM,MNIST_IMG_DIM))

def warp(img, A, B):
    new_img = copy.deepcopy(img)
    for old_x in range(MNIST_IMG_DIM):
        for y in range(MNIST_IMG_DIM):
            x = int(old_x + A *np.sin(2.0 * np.pi  * y / B))
            new_img[x][y] = img[old_x][y]
    return new_img

def generate_worlds(mnist, n, cf = False, transform=True):
    # TODO: remove useless typing
    num1_locs = np.where(mnist["labels"] == np.asarray([NUM1]))[0]
    num2_locs = np.where(mnist["labels"] == np.asarray([NUM2]))[0]

    scenarios = [] # a scenario is an actual world and its cfs
    for i in range(n):
        act_world = {key: np.random.binomial(1,p[key]) for key in p.keys()}

        if (cf):
            cf_worlds = generate_cf_worlds(act_world)
            all_worlds = [act_world] + cf_worlds    # act_world comes first
            imgs = imgs_of_worlds(all_worlds, mnist, transform, num1_locs, num2_locs)
            img = np.concatenate(tuple(imgs), axis=1)
            label = reformat(act_world)[1]
        else:
            all_worlds = [act_world]
            img = imgs_of_worlds([act_world], mnist, transform, num1_locs, num2_locs)[0]
            label = reformat(act_world)[1]
        
        scenarios.append((img, label))

        # save_image(torch.from_numpy(joined_img), str(i) + ".jpg")
    return scenarios

def num_from_mnist(mnist, locs):
    loc = np.random.choice(locs)
    return mnist["digits"][loc]

def imgs_of_worlds(worlds, mnist, transform, num1_locs, num2_locs):
    num1_img = num_from_mnist(mnist, num1_locs)
    num2_img = num_from_mnist(mnist, num2_locs)    
    return [img_of_world(world, num1_img, num2_img, transform) for world in worlds]

# note: BLK_SQR is all 0's; assumes no preprocessing (else, BLK_SQR would be all -1's)
def img_of_world(world, num1_img, num2_img, transform):
    nums, utt = reformat(world)[0], reformat(world)[1]

    # set numbers in corners if they're in the world
    top_left = num1_img if nums[0] else BLK_SQR
    bottom_right = num2_img if nums[1] else BLK_SQR
   
    # apply nonlinear transformations; note that pixels are in interval [0,1]
    if (transform) and (world["C"] and world["cE"]):
        top_left = warp(top_left, 1, 1)
        top_left = clamp_img(top_left, MNIST_MIN_COLOR, MNIST_MAX_COLOR)

    # put all four corner images together
    img = np.concatenate((np.concatenate((top_left, BLK_SQR), axis=1),
                      np.concatenate((BLK_SQR, bottom_right), axis=1)),
                     axis=0)

    # move image from [0,1] to [-1,1]
    return standardize_img(img, from_unit_interval=True)

def reformat(world):
    num1 = NUM1 if world["C"] or world["nC"] else ""
    num2 = NUM2 if ((world["C"] and world["cE"]) or world["bE"]) else ""
    utt = ("causal" if (world["C"] and world["cE"]) else "") + str(num1) + str(num2)
    return [num1, num2], utt

def flip_rv(actual, key, key_to_vary):
    return not actual[key] if key == key_to_vary else actual[key]

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
    return [flip_rvs(act_world, key_to_vary) for key_to_vary in act_world]

# only grabs imgs/labels if they're num1 or num2
def pick_imgs(loader):
    data  = {'digits': [], 'labels': []}
    for imgs,labels in loader:
        for img, label in zip(imgs[:,0,...], labels):
            if (label == NUM1 or label == NUM2):
                data['digits'].append(img.numpy())
                data['labels'].append(label.numpy())
    return data

# images' pixels are in interval [0,1]
def load_mnist(root, test):
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform.Compose([transform.Resize(MNIST_IMG_DIM), transform.ToTensor()])
        ),
        batch_size = 500,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(
            root=root,
            train=False,
            download=True,
            transform=transform.Compose([transform.Resize(MNIST_IMG_DIM), transform.ToTensor()])
        ),
        batch_size = 500,
        shuffle=False
    )
    

    (loader, data_kind) = (test_loader, "test") if test else (train_loader, "train/validate")

    return pick_imgs(loader)

# might be worth using pathlib
def mnist_dir_setup(test):
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    # get cur dir
    data_kind = "test" if test else "train-validate"
    file_name = data_file_name(prefix="mnist", suffix = data_kind)

    if (os.path.isfile(file_name)):
        data = np.load(file_name, allow_pickle=True).item()
        print("retrieved {} mnist data from file.".format(data_kind))
        return data
    else:
        data = load_mnist(DATA_DIR, test)
        np.save(file_name, data)
        print("retrieved " + data_kind + " mnist data from online.")

    return data

if __name__ ==  "__main__":
    import argparse
    # handle args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [default: 42]')
    parser.add_argument('--test', action='store_true', default=False,
                        help='sample digits from test set of MNIST [default: False]')
    args = parser.parse_args()

    np.random.seed(args.seed)

    mnist= mnist_dir_setup(args.test)
